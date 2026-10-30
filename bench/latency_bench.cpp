// Tail-latency benchmark: mpmc::BoundedMpmcQueue vs Boost.Lockfree.
//
// Three latency distributions are collected for each queue / thread config:
//   enqueue  - duration of the successful try_enqueue/push call
//   dequeue  - duration of the successful try_dequeue/pop call
//   e2e      - producer timestamp -> consumer receipt (includes queueing time)
//
// Failed calls (queue full / empty) are counted but not timed, so enqueue and
// dequeue latencies reflect the cost of the operation itself under contention
// (CAS retries, cache-line transfers), not backpressure.
//
// In paced mode (--rate), producers send on a fixed schedule and stamp each
// message with its *intended* send time, so e2e latency is free of
// coordinated omission: a stalled producer shows up as latency, not as
// silently missing samples.

#include "../mpmc_queue.hpp"

#include <boost/lockfree/bounded_ticket_queue.hpp>
#include <boost/lockfree/queue.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#include <pthread.h>
#include <sys/qos.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

// ---------------------------------------------------------------------------
// Timer: raw hardware counter, converted to ns after the run.
// ---------------------------------------------------------------------------

namespace timer {

#if defined(__APPLE__)
// Reading cntvct_el0 directly is not reliable on recent macOS; use the
// OS-provided counter (24 MHz on Apple Silicon, ~41.7 ns per tick).
inline std::uint64_t now() { return mach_absolute_time(); }
inline double ticks_per_ns() {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return static_cast<double>(tb.denom) / tb.numer;
}
#elif defined(__aarch64__)
inline std::uint64_t now() {
    std::uint64_t v;
    asm volatile("isb; mrs %0, cntvct_el0" : "=r"(v)::"memory");
    return v;
}
inline double ticks_per_ns() {
    std::uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return static_cast<double>(f) / 1e9;
}
#elif defined(__x86_64__) || defined(_M_X64)
inline std::uint64_t now() {
    unsigned aux;
    return __rdtscp(&aux);
}
inline double ticks_per_ns() {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    std::uint64_t c0 = now();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto t1 = clk::now();
    std::uint64_t c1 = now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return static_cast<double>(c1 - c0) / ns;
}
#else
inline std::uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline double ticks_per_ns() { return 1.0; }
#endif

}  // namespace timer

// ---------------------------------------------------------------------------
// Queue adapters
// ---------------------------------------------------------------------------

struct Msg {
    std::uint64_t ts;
    std::uint64_t seq;
};

struct MineQueue {
    static constexpr const char* name = "mpmc::BoundedMpmcQueue";
    mpmc::BoundedMpmcQueue<Msg> q;
    explicit MineQueue(std::size_t cap) : q(cap) {}
    bool push(const Msg& m) { return q.try_enqueue(m); }
    bool pop(Msg& m) { return q.try_dequeue(m); }
};

// Same queue with each slot on its own cache line.
struct PaddedQueue {
    static constexpr const char* name = "mpmc::BoundedMpmcQueue<pad>";
    mpmc::BoundedMpmcQueue<Msg, true> q;
    explicit PaddedQueue(std::size_t cap) : q(cap) {}
    bool push(const Msg& m) { return q.try_enqueue(m); }
    bool pop(Msg& m) { return q.try_dequeue(m); }
};

// Same algorithm family (Vyukov sequence-per-cell) - the apples-to-apples one.
struct BoostTicketQueue {
    static constexpr const char* name = "boost::bounded_ticket_queue";
    boost::lockfree::bounded_ticket_queue<Msg> q;
    explicit BoostTicketQueue(std::size_t cap) : q(cap) {}
    bool push(const Msg& m) { return q.push(m); }
    bool pop(Msg& m) { return q.pop(m); }
};

// Michael-Scott linked queue over a fixed node pool.
struct BoostMsQueue {
    static constexpr const char* name = "boost::lockfree::queue";
    boost::lockfree::queue<Msg, boost::lockfree::fixed_sized<true>> q;
    explicit BoostMsQueue(std::size_t cap) : q(cap) {}
    bool push(const Msg& m) { return q.bounded_push(m); }
    bool pop(Msg& m) { return q.pop(m); }
};

// ---------------------------------------------------------------------------
// Config / results
// ---------------------------------------------------------------------------

struct Config {
    std::vector<std::pair<int, int>> thread_configs{{1, 1}, {2, 2}, {4, 4}, {1, 4}, {4, 1}};
    std::size_t capacity = 1024;
    std::size_t ops_per_producer = 500000;
    int reps = 7;
    double rate = 0;  // msgs/sec per producer; 0 = closed loop (max contention)
    std::vector<std::string> queues{"mine", "padded", "ticket", "ms"};
    std::string csv;
};

struct Percentiles {
    double p50, p90, p95, p99, p999, max;
};

struct RunResult {
    Percentiles enq, deq, e2e;
    std::uint64_t full_retries, empty_retries;
    double throughput_mops;
};

static Percentiles percentiles(std::vector<std::uint64_t>& v, double tpns) {
    std::sort(v.begin(), v.end());
    auto at = [&](double p) {
        // nearest-rank
        std::size_t idx = static_cast<std::size_t>(p * v.size());
        if (idx >= v.size()) idx = v.size() - 1;
        return v[idx] / tpns;
    };
    return {at(0.50), at(0.90), at(0.95), at(0.99), at(0.999), v.back() / tpns};
}

static void tune_current_thread(int cpu_hint) {
#if defined(__APPLE__)
    // macOS has no hard affinity; ask for P-cores via QoS.
    (void)cpu_hint;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__)
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_hint % n, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu_hint;
#endif
}

// ---------------------------------------------------------------------------
// One run
// ---------------------------------------------------------------------------

template <typename Q>
RunResult run_once(const Config& cfg, int producers, int consumers, double tpns) {
    auto q = std::make_unique<Q>(cfg.capacity);
    const std::size_t n = cfg.ops_per_producer;
    const std::uint64_t total = n * producers;
    // Ticks between sends for each producer in paced mode.
    const std::uint64_t interval =
        cfg.rate > 0 ? static_cast<std::uint64_t>(1e9 / cfg.rate * tpns) : 0;

    std::vector<std::vector<std::uint64_t>> enq(producers), deq(consumers), e2e(consumers);
    std::vector<std::uint64_t> full_retries(producers), empty_retries(consumers);
    // Pre-size and touch every buffer so page faults don't land in the run.
    for (auto& v : enq) v.assign(n, 0);
    for (int c = 0; c < consumers; ++c) {
        deq[c].assign(total, 0);
        e2e[c].assign(total, 0);
    }

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<std::uint64_t> consumed{0};
    std::uint64_t start_tick = 0, end_tick = 0;

    std::vector<std::thread> threads;
    for (int p = 0; p < producers; ++p) {
        threads.emplace_back([&, p] {
            tune_current_thread(p);
            auto& lat = enq[p];
            std::uint64_t retries = 0;
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {
            }
            std::uint64_t next = timer::now();
            for (std::size_t i = 0; i < n; ++i) {
                Msg m;
                if (interval) {
                    while (timer::now() < next) {
                    }
                    m.ts = next;  // intended send time
                    next += interval;
                } else {
                    m.ts = timer::now();
                }
                m.seq = i;
                for (;;) {
                    std::uint64_t t0 = timer::now();
                    bool ok = q->push(m);
                    std::uint64_t t1 = timer::now();
                    if (ok) {
                        lat[i] = t1 - t0;
                        break;
                    }
                    ++retries;
                }
            }
            full_retries[p] = retries;
        });
    }
    for (int c = 0; c < consumers; ++c) {
        threads.emplace_back([&, c] {
            tune_current_thread(producers + c);
            auto& dl = deq[c];
            auto& el = e2e[c];
            std::size_t k = 0;
            std::uint64_t retries = 0;
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {
            }
            Msg m;
            while (consumed.load(std::memory_order_relaxed) < total) {
                std::uint64_t t0 = timer::now();
                bool ok = q->pop(m);
                std::uint64_t t1 = timer::now();
                if (ok) {
                    dl[k] = t1 - t0;
                    el[k] = t1 - m.ts;
                    ++k;
                    consumed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    ++retries;
                }
            }
            dl.resize(k);
            el.resize(k);
            empty_retries[c] = retries;
        });
    }

    while (ready.load() != producers + consumers) {
    }
    start_tick = timer::now();
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    end_tick = timer::now();

    auto merge = [](std::vector<std::vector<std::uint64_t>>& parts) {
        std::vector<std::uint64_t> all;
        std::size_t sz = 0;
        for (auto& v : parts) sz += v.size();
        all.reserve(sz);
        for (auto& v : parts) all.insert(all.end(), v.begin(), v.end());
        return all;
    };
    auto enq_all = merge(enq), deq_all = merge(deq), e2e_all = merge(e2e);
    if (deq_all.size() != total) {
        std::fprintf(stderr, "BUG: consumed %zu of %llu\n", deq_all.size(),
                     static_cast<unsigned long long>(total));
        std::exit(1);
    }

    RunResult r;
    r.enq = percentiles(enq_all, tpns);
    r.deq = percentiles(deq_all, tpns);
    r.e2e = percentiles(e2e_all, tpns);
    r.full_retries = 0;
    r.empty_retries = 0;
    for (auto x : full_retries) r.full_retries += x;
    for (auto x : empty_retries) r.empty_retries += x;
    double secs = (end_tick - start_tick) / tpns / 1e9;
    r.throughput_mops = total / secs / 1e6;
    return r;
}

using Runner = std::function<RunResult(const Config&, int, int, double)>;

struct QueueEntry {
    std::string key;
    std::string name;
    Runner run;
};

static std::vector<QueueEntry> all_queues() {
    return {
        {"mine", MineQueue::name, run_once<MineQueue>},
        {"padded", PaddedQueue::name, run_once<PaddedQueue>},
        {"ticket", BoostTicketQueue::name, run_once<BoostTicketQueue>},
        {"ms", BoostMsQueue::name, run_once<BoostMsQueue>},
    };
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Summary {
    // Median across reps of each percentile; also min/max of P99 across reps.
    Percentiles enq, deq, e2e;
    double e2e_p99_min, e2e_p99_max, deq_p99_min, deq_p99_max;
    double enq_p99_min, enq_p99_max;
    double mops;
    double full_retries, empty_retries;
};

static Percentiles median_of(const std::vector<RunResult>& rs, Percentiles RunResult::*f) {
    auto col = [&](double Percentiles::*g) {
        std::vector<double> v;
        for (auto& r : rs) v.push_back((r.*f).*g);
        return median(v);
    };
    return {col(&Percentiles::p50), col(&Percentiles::p90), col(&Percentiles::p95),
            col(&Percentiles::p99), col(&Percentiles::p999), col(&Percentiles::max)};
}

static Summary summarize(const std::vector<RunResult>& rs) {
    Summary s;
    s.enq = median_of(rs, &RunResult::enq);
    s.deq = median_of(rs, &RunResult::deq);
    s.e2e = median_of(rs, &RunResult::e2e);
    auto range = [&](Percentiles RunResult::*f, double& lo, double& hi) {
        lo = 1e300;
        hi = 0;
        for (auto& r : rs) {
            lo = std::min(lo, (r.*f).p99);
            hi = std::max(hi, (r.*f).p99);
        }
    };
    range(&RunResult::enq, s.enq_p99_min, s.enq_p99_max);
    range(&RunResult::deq, s.deq_p99_min, s.deq_p99_max);
    range(&RunResult::e2e, s.e2e_p99_min, s.e2e_p99_max);
    std::vector<double> m, fr, er;
    for (auto& r : rs) {
        m.push_back(r.throughput_mops);
        fr.push_back(static_cast<double>(r.full_retries));
        er.push_back(static_cast<double>(r.empty_retries));
    }
    s.mops = median(m);
    s.full_retries = median(fr);
    s.empty_retries = median(er);
    return s;
}

static void print_row(const char* metric, const std::string& qname, const Percentiles& p,
                      double p99lo, double p99hi) {
    std::printf("  %-8s %-28s %8.0f %8.0f %8.0f %8.0f %9.0f %10.0f   [%.0f..%.0f]\n", metric,
                qname.c_str(), p.p50, p.p90, p.p95, p.p99, p.p999, p.max, p99lo, p99hi);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    std::size_t start = 0, pos;
    while ((pos = s.find(d, start)) != std::string::npos) {
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    out.push_back(s.substr(start));
    return out;
}

static void usage() {
    std::puts(
        "usage: latency_bench [options]\n"
        "  --threads PxC,...   producer x consumer configs (default 1x1,2x2,4x4,1x4,4x1)\n"
        "  --capacity N        queue capacity (default 1024)\n"
        "  --ops N             messages per producer per rep (default 500000)\n"
        "  --reps N            measured repetitions per config (default 7)\n"
        "  --rate R            paced mode: R msgs/sec per producer (default 0 = closed loop)\n"
        "  --queues a,b,...    subset of: mine,padded,ticket,ms (default all)\n"
        "  --csv FILE          also write per-rep raw percentiles to FILE");
}

static Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--threads") {
            c.thread_configs.clear();
            for (auto& t : split(val(), ',')) {
                auto pc = split(t, 'x');
                c.thread_configs.emplace_back(std::stoi(pc.at(0)), std::stoi(pc.at(1)));
            }
        } else if (a == "--capacity") {
            c.capacity = std::stoull(val());
        } else if (a == "--ops") {
            c.ops_per_producer = std::stoull(val());
        } else if (a == "--reps") {
            c.reps = std::stoi(val());
        } else if (a == "--rate") {
            c.rate = std::stod(val());
        } else if (a == "--queues") {
            c.queues = split(val(), ',');
        } else if (a == "--csv") {
            c.csv = val();
        } else {
            usage();
            std::exit(a == "-h" || a == "--help" ? 0 : 1);
        }
    }
    return c;
}

int main(int argc, char** argv) {
    Config cfg = parse(argc, argv);
    tune_current_thread(0);

    std::vector<QueueEntry> queues;
    for (auto& e : all_queues())
        if (std::find(cfg.queues.begin(), cfg.queues.end(), e.key) != cfg.queues.end())
            queues.push_back(e);

    const double tpns = timer::ticks_per_ns();
    {
        // Timer cost / resolution: smallest nonzero back-to-back delta.
        std::uint64_t best = ~0ull, overhead_sum = 0;
        const int k = 100000;
        for (int i = 0; i < k; ++i) {
            std::uint64_t a = timer::now(), b = timer::now();
            overhead_sum += b - a;
            if (b > a) best = std::min(best, b - a);
        }
        std::printf("timer: %.3f ticks/ns, resolution ~%.1f ns, mean back-to-back %.1f ns\n",
                    tpns, best / tpns, overhead_sum / double(k) / tpns);
    }
    std::printf("capacity=%zu ops/producer=%zu reps=%d mode=%s\n", cfg.capacity,
                cfg.ops_per_producer, cfg.reps,
                cfg.rate > 0 ? ("paced " + std::to_string(cfg.rate) + " msg/s/producer").c_str()
                             : "closed-loop");
    std::printf("All latencies in ns. Each cell = median across reps; [..] = P99 min..max across reps.\n");

    std::FILE* csv = nullptr;
    if (!cfg.csv.empty()) {
        csv = std::fopen(cfg.csv.c_str(), "w");
        std::fprintf(csv,
                     "queue,producers,consumers,rep,metric,p50,p90,p95,p99,p999,max,mops,"
                     "full_retries,empty_retries\n");
    }

    for (auto [P, C] : cfg.thread_configs) {
        std::printf("\n=== %dP x %dC ===\n", P, C);

        // Warm-up rep per queue (discarded).
        for (auto& q : queues) q.run(cfg, P, C, tpns);

        // Interleave queues within each rep so thermal / frequency drift
        // affects all of them equally; rotate the order every rep.
        std::vector<std::vector<RunResult>> results(queues.size());
        for (int rep = 0; rep < cfg.reps; ++rep) {
            for (std::size_t j = 0; j < queues.size(); ++j) {
                std::size_t qi = (j + rep) % queues.size();
                RunResult r = queues[qi].run(cfg, P, C, tpns);
                results[qi].push_back(r);
                if (csv) {
                    auto emit = [&](const char* metric, const Percentiles& p) {
                        std::fprintf(csv, "%s,%d,%d,%d,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%llu,%llu\n",
                                     queues[qi].key.c_str(), P, C, rep, metric, p.p50, p.p90,
                                     p.p95, p.p99, p.p999, p.max, r.throughput_mops,
                                     static_cast<unsigned long long>(r.full_retries),
                                     static_cast<unsigned long long>(r.empty_retries));
                    };
                    emit("enqueue", r.enq);
                    emit("dequeue", r.deq);
                    emit("e2e", r.e2e);
                }
            }
        }

        std::printf("  %-8s %-28s %8s %8s %8s %8s %9s %10s   %s\n", "metric", "queue", "p50", "p90",
                    "p95", "p99", "p99.9", "max", "p99 range");
        std::vector<Summary> sums;
        for (auto& r : results) sums.push_back(summarize(r));
        for (std::size_t i = 0; i < queues.size(); ++i)
            print_row("enqueue", queues[i].name, sums[i].enq, sums[i].enq_p99_min, sums[i].enq_p99_max);
        for (std::size_t i = 0; i < queues.size(); ++i)
            print_row("dequeue", queues[i].name, sums[i].deq, sums[i].deq_p99_min, sums[i].deq_p99_max);
        for (std::size_t i = 0; i < queues.size(); ++i)
            print_row("e2e", queues[i].name, sums[i].e2e, sums[i].e2e_p99_min, sums[i].e2e_p99_max);
        for (std::size_t i = 0; i < queues.size(); ++i)
            std::printf("  %-37s throughput %.2f Mops/s, full-retries %.0f, empty-retries %.0f\n",
                        queues[i].name.c_str(), sums[i].mops, sums[i].full_retries,
                        sums[i].empty_retries);
    }
    if (csv) std::fclose(csv);
}

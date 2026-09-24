#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mpmc {

constexpr std::size_t kCacheLineSize = 128;

template <typename T>
class BoundedMpmcQueue {
    static_assert(std::is_move_constructible<T>::value,
                  "T must be move-constructible");

public:
    explicit BoundedMpmcQueue(std::size_t capacity)
        : capacity_(round_up_pow2(capacity < 2 ? 2 : capacity)),
          mask_(capacity_ - 1),
          buffer_(static_cast<Slot*>(::operator new[](capacity_ * sizeof(Slot)))) {
        for (std::size_t i = 0; i < capacity_; ++i) {
            new (&buffer_[i]) Slot();
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    ~BoundedMpmcQueue() {
        T tmp;
        while (try_dequeue(tmp)) {
        }
        for (std::size_t i = 0; i < capacity_; ++i) {
            buffer_[i].~Slot();
        }
        ::operator delete[](buffer_);
    }

    BoundedMpmcQueue(const BoundedMpmcQueue&) = delete;
    BoundedMpmcQueue& operator=(const BoundedMpmcQueue&) = delete;

    std::size_t capacity() const noexcept { return capacity_; }

    template <typename U>
    bool try_enqueue(U&& item) {
        Slot* slot;
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            slot = &buffer_[pos & mask_];
            std::size_t seq = slot->sequence.load(std::memory_order_acquire);
            std::intptr_t diff =
                static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);

            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(
                        pos, pos + 1,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        slot->construct(std::forward<U>(item));
        slot->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool try_dequeue(T& out) {
        Slot* slot;
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

        for (;;) {
            slot = &buffer_[pos & mask_];
            std::size_t seq = slot->sequence.load(std::memory_order_acquire);
            std::intptr_t diff =
                static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);

            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(
                        pos, pos + 1,
                        std::memory_order_relaxed,
                        std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }

        out = slot->move_out();
        slot->sequence.store(pos + mask_ + 1, std::memory_order_release);
        return true;
    }

private:
    struct Slot {
        std::atomic<std::size_t> sequence;
        typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;

        Slot() : sequence(0) {}
        ~Slot() {}

        template <typename U>
        void construct(U&& value) {
            new (&storage) T(std::forward<U>(value));
        }

        T move_out() {
            T* p = reinterpret_cast<T*>(&storage);
            T result(std::move(*p));
            p->~T();
            return result;
        }
    };

    static std::size_t round_up_pow2(std::size_t v) {
        std::size_t p = 1;
        while (p < v) p <<= 1;
        return p;
    }

    const std::size_t capacity_;
    const std::size_t mask_;
    Slot* const buffer_;

    alignas(kCacheLineSize) std::atomic<std::size_t> enqueue_pos_;
    char pad1_[kCacheLineSize - sizeof(std::atomic<std::size_t>)];
    alignas(kCacheLineSize) std::atomic<std::size_t> dequeue_pos_;
    char pad2_[kCacheLineSize - sizeof(std::atomic<std::size_t>)];
};

}  // namespace mpmc

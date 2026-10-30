BOOST_ROOT ?= $(shell brew --prefix 2>/dev/null || echo /usr/local)
CXX       ?= c++
CXXFLAGS  ?= -std=c++20 -O3 -DNDEBUG -march=native -Wall -Wextra
LDFLAGS   ?= -pthread

latency_bench: latency_bench.cpp ../mpmc_queue.hpp
	$(CXX) $(CXXFLAGS) -I$(BOOST_ROOT)/include $< -o $@ $(LDFLAGS)

clean:
	rm -f latency_bench

.PHONY: clean

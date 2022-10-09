#pragma once

#include <atomic>

namespace coll_cache_lib {
namespace common {

struct AtomicBarrier {
  std::atomic_int counter{0};
  std::atomic_bool flag{false};
  int worker = 0;
  void Wait() {
    int local_f = flag.load();
    int local_counter = counter.fetch_add(1);
    if (local_counter + 1 == worker) {
      counter.store(0);
      flag.store(!local_f);
    } else {
      while (flag.load() == local_f) {}
    }
  }
};

}
}
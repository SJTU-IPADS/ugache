#pragma once

#include <atomic>
#include "common.h"

namespace coll_cache_lib {
namespace common {

class AtomicBarrier : public ExternalBarrierHandler {
  std::atomic_int counter{0};
  std::atomic_bool flag{false};
  int worker = 0;
 public:
  AtomicBarrier(int num_syncher) : worker(num_syncher) {}
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
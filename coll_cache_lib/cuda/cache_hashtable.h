#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

#include "../device.h"
#include "../common.h"
#include "../constant.h"
#include "../logging.h"

namespace coll_cache_lib {
namespace common {
class SimpleHashTable;
constexpr IdType kEmptyPos = 0xffffffff;

struct EmbCacheOff {
  IdType data;
  __host__ __device__ __forceinline__ EmbCacheOff(IdType loc, IdType off) { data = (loc << 28) + off; }
  __host__ __device__ __forceinline__ EmbCacheOff& operator=(const EmbCacheOff & v) { data = v.data; return *this;}
  __host__ __device__ __forceinline__ IdType loc() const { return (data >> 28) & 0x0f; }
  __host__ __device__ __forceinline__ IdType off() const { return data & 0x0fffffff; }
};

using ValType = EmbCacheOff;
struct alignas(unsigned long long) BucketO2N {
  // don't change the position of version and key
  //   which used for efficient insert operation
  IdType state_key;
  ValType val;
};
class CacheEntryManager;
}
}
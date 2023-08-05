#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

#include "../device.h"
#include "../common.h"
#include "../constant.h"
// #include "../logging.h"

namespace coll_cache_lib {
namespace common {
class SimpleHashTable;
class FlatHashTable;
constexpr IdType kEmptyPos = 0xffffffff;
size_t TableSize(const size_t num, const size_t scale = 2);

struct EmbCacheOff {
  IdType data;
  __host__ __device__ __forceinline__ EmbCacheOff(IdType loc, IdType off) { data = (loc << 28) + off; }
  __host__ __device__ __forceinline__ EmbCacheOff& operator=(const EmbCacheOff & v) { data = v.data; return *this;}
  __host__ __device__ __forceinline__ IdType loc() const { return (data >> 28) & 0x0f; }
  __host__ __device__ __forceinline__ IdType off() const { return data & 0x0fffffff; }
  __host__ __device__ __forceinline__ void set_loc(const IdType loc) { data = (data & 0x0fffffff) | (loc << 28); }
  __host__ __device__ __forceinline__ void set_off(const IdType off) { data = (data & 0xf0000000) | (off); }
};

struct IdxStoreAPI {
  __forceinline__ __host__ __device__ IdType src_loc(const IdType idx) const { abort(); }
  __forceinline__ __host__ __device__ IdType src_off(const IdType idx) const { abort(); }
  __forceinline__ __host__ __device__ IdType dst_off(const IdType idx) const { abort(); }
  // __forceinline__ __host__ __device__ void set_src_loc(const IdType idx, const IdType src_loc) { }
  // __forceinline__ __host__ __device__ void set_src_off(const IdType idx, const IdType src_off) { }
  __forceinline__ __host__ __device__ void set_src(const IdType idx, const IdType src_loc, const IdType src_off) { }
  __forceinline__ __host__ __device__ void set_dst_off(const IdType idx, const IdType dst_off) { }

  __forceinline__ __host__ __device__ size_t required_mem(const IdType num_keys) const { abort(); }
  __forceinline__ __host__ __device__ void prepare_mem(uint8_t* ptr, const IdType num_keys) { abort(); }
};

using ValType = EmbCacheOff;
struct alignas(unsigned long long) BucketO2N {
  // don't change the position of version and key
  //   which used for efficient insert operation
  IdType state_key;
  ValType val;
};
struct alignas(unsigned int) BucketFlat {
  // don't change the position of version and key
  //   which used for efficient insert operation
  ValType val;
};
class CacheEntryManager;
}
}
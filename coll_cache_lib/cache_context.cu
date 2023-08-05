#include "common.h"
#include "cpu/cpu_device.h"
#include "cpu/cpu_utils.h"
#include "cuda/cuda_utils.h"
#include "cpu/mmap_cpu_device.h"
#include "run_config.h"
#include "logging.h"
#include "coll_cache/ndarray.h"
#include "coll_cache/optimal_solver_class.h"
// #include "atomic_barrier.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cub/cub.cuh>

#include "facade.h"
#include "timer.h"
#include "cache_context.h"
#include "cuda/cub_sort_wrapper.cuh"
#include "cuda/mps_util.h"
#include "cuda/cache_hashtable.cuh"

#define SWITCH_TYPE(type, Type, ...)      \
  switch(type) {                     \
    case kF32: { typedef float   Type; { __VA_ARGS__ }; break; } \
    case kF64: { typedef double  Type; { __VA_ARGS__ }; break; } \
    case kF16: { typedef short   Type; { __VA_ARGS__ }; break; } \
    case kU8:  { typedef uint8_t Type; { __VA_ARGS__ }; break; } \
    case kI32: { typedef int32_t Type; { __VA_ARGS__ }; break; } \
    case kI64: { typedef int64_t Type; { __VA_ARGS__ }; break; } \
    case kF64_2: { typedef int4 Type; { __VA_ARGS__ }; break; } \
    case kF64_4: { typedef double4 Type; { __VA_ARGS__ }; break; } \
    default: CHECK(false);           \
  }


#define SAM_CUDA_PREPARE_1D(num_item) \
  const size_t num_tiles = RoundUpDiv((num_item), Constant::kCudaTileSize); \
  const dim3 grid(num_tiles); \
  const dim3 block(Constant::kCudaBlockSize);

#define SWITCH_BOOL(expr, alias, ...)             \
  if (expr) {                                     \
    constexpr bool alias=true;   { __VA_ARGS__ }; \
  } else {                                        \
    constexpr bool alias=false;  { __VA_ARGS__ }; \
  }

namespace coll_cache_lib {

using namespace common;
// per-gpu cache handler
#ifdef DEAD_CODE
struct GetIdxHelper {
  SrcKey* src_key;
  DstVal* dst_val;
  int fallback_loc;
  GetIdxHelper(SrcKey* src_key, DstVal* dst_val, int fallback_loc) : src_key(src_key), dst_val(dst_val), fallback_loc(fallback_loc) {}
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
    if (val) {
      src_key[idx]._location_id = val->val.loc();
      dst_val[idx]._src_offset = val->val.off();
      dst_val[idx]._dst_offset = idx;
    } else {
      src_key[idx]._location_id = fallback_loc;
      dst_val[idx]._src_offset = key;
      dst_val[idx]._dst_offset = idx;
    };
  }
};
struct GetIdxHelperMock {
  SrcKey* src_key;
  DstVal* dst_val;
  int fallback_loc;
  size_t empty_feat;

  GetIdxHelperMock(SrcKey* src_key, DstVal* dst_val, int fallback_loc) : src_key(src_key), dst_val(dst_val), fallback_loc(fallback_loc) {
    empty_feat = RunConfig::option_empty_feat;
  }
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
    if (val) {
      src_key[idx]._location_id = val->val.loc();
      dst_val[idx]._src_offset = val->val.off();
      dst_val[idx]._dst_offset = idx;
    } else {
      src_key[idx]._location_id = fallback_loc;
      dst_val[idx]._src_offset = key % (1 << empty_feat);
      dst_val[idx]._dst_offset = idx;
    };
  }
};
#endif

namespace {

struct DataIterAPI {
  template<typename T>
  __forceinline__ __host__ __device__ T* src(const size_t & idx) {}
  template<typename T>
  __forceinline__ __host__ __device__ T* dst(const size_t & idx) {}
};

#ifdef DEAD_CODE
/**
 * @brief dynamic finds source location & offset when extracting feature, by old location & offset array impl
 */
template<typename OffsetIter_T>
struct DataIterMultiLocation {
  const OffsetIter_T _offset_iter = nullptr;
  HashTableEntryLocation* _hash_table_location = nullptr;
  HashTableEntryOffset* _hash_table_offset = nullptr;
  void* _device_cache_data[9] = {nullptr};
  size_t dim;
  DataIterMultiLocation(const OffsetIter_T offset_iter,
        HashTableEntryLocation* location,
        HashTableEntryOffset* offset,
        std::vector<void*> & cache_data,
        size_t dim) :
    _offset_iter(offset_iter), _hash_table_offset(offset), _hash_table_location(location), dim(dim) {
    memcpy(_device_cache_data, cache_data.data(), sizeof(void*) * cache_data.size());
  }
  template<typename T>
  __forceinline__ __host__ __device__ T* operator[](const size_t & idx) const {
    const size_t src_offset = _offset_iter[idx];
    const int location = _hash_table_location[src_offset];
    const auto _remote_raw_data = (T*)_device_cache_data[location];
    const auto offset = _hash_table_offset[src_offset];
    return _remote_raw_data + offset * dim;
  }
};
#endif
/**
 * @brief dynamic finds source location & offset when extracting feature, but more general
 * typical usage is to provide separate loc&off array by lookup cache hashtable in advance.
 */
template<typename IdxStore_T>
struct DataIterMixLoc : DataIterAPI {
  IdxStore_T idx_store;
  void* _device_cache_data[9] = {nullptr};
  void* output;
  size_t dim;
  DataIterMixLoc(const IdxStore_T idx_store,
        std::vector<void*> & cache_data,
        void* output,
        size_t dim) :
    idx_store(idx_store), dim(dim), output(output) {
    memcpy(_device_cache_data, cache_data.data(), sizeof(void*) * cache_data.size());
  }
  template<typename T>
  __forceinline__ __host__ __device__ T* src(const size_t & idx) {
    const auto src_data = (T*)_device_cache_data[idx_store.src_loc(idx)];
    return ((T*)src_data) + idx_store.src_off(idx) * dim;
  }
  template<typename T>
  __forceinline__ __host__ __device__ T* dst(const size_t & idx) {
    return ((T*)output) + idx_store.dst_off(idx) * dim;
  }
};
template<int clique_size>
struct DataIterCliqSrcRR {
  void* _device_cache_data[clique_size] = {nullptr};
  const IdType chunk_size;
  DataIterCliqSrcRR(int _, std::vector<void*> & cache_data, size_t dim, IdType chunk_size)
   : chunk_size(chunk_size) {
    memcpy(_device_cache_data, cache_data.data(), sizeof(void*) * cache_data.size());
  }
  template<typename T>
  __forceinline__ __host__ __device__ T* call(const size_t & key, size_t dim) const {
    return (T*)_device_cache_data[key % clique_size] + (key / clique_size) * dim;
  }
};
template<int clique_size>
struct DataIterCliqSrcChunk {
  void* _device_cache_data[clique_size] = {nullptr};
  const IdType chunk_size;
  DataIterCliqSrcChunk(int _, std::vector<void*> & cache_data, size_t dim, IdType chunk_size)
   : chunk_size(chunk_size) {
    memcpy(_device_cache_data, cache_data.data(), sizeof(void*) * cache_data.size());
  }

  template<typename T>
  __forceinline__ __host__ __device__ T* call(const size_t & key, size_t dim) const {
    return (T*)_device_cache_data[key / chunk_size] + (key % chunk_size) * dim;
  }
};
struct DataIterCliqDst {
  void* data;
  DataIterCliqDst(void* data, size_t dim) : data(data) {}
  template<typename T>
  __forceinline__ __host__ __device__ T* call(const size_t & idx, size_t dim) const {
    return ((T*)data) + idx * dim;
  }
};
template<typename IdxStore_T>
struct DataIterPerLoc : DataIterAPI {
  IdxStore_T idx_store;
  const void* src_data = nullptr;
  void* output = nullptr;
  size_t dim;
  DataIterPerLoc() {}
  DataIterPerLoc(IdxStore_T idx_store, const void* src_data, void* output, size_t dim) : 
    idx_store(idx_store), src_data(src_data), output(output), dim(dim) {}
  template<typename T>
  __forceinline__ __host__ __device__ T* src(const size_t & idx) {
    return ((T*)src_data) + idx_store.src_off(idx) * dim;
  }
  template<typename T>
  __forceinline__ __host__ __device__ T* dst(const size_t & idx) {
    return ((T*)output) + idx_store.dst_off(idx) * dim;
  }
};

template <typename T, typename DataIter_T>
__global__ void extract_data(DataIter_T ptrs,
                             const size_t num_node,
                             size_t dim) {
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_node) {
    size_t col = threadIdx.x;
    T* dst = ptrs.template dst<T>(i);
    const T* src = ptrs.template src<T>(i);
    while (col < dim) {
      dst[col] = src[col];
      col += blockDim.x;
    }
    i += stride;
  }
}

template<int NUM_LINK, typename DataIter_T>
struct ExtractConcurrentParam {
  DataIter_T data_iter_array[NUM_LINK];
  IdType num_node_array[NUM_LINK];
  IdType block_num_prefix_sum[NUM_LINK + 1];
  const IdType* link_mapping;
  const IdType* sub_block_mappling;
  size_t dim;
};

template <int NUM_LINK, typename T, typename DataIter_T>
__global__ void extract_data_concurrent(ExtractConcurrentParam<NUM_LINK, DataIter_T> packed_param) {
  // block -> which link
  // block -> local block idx in this link
  // block -> num block of this link
  const IdType link_idx = packed_param.link_mapping[blockIdx.x];
  const IdType local_block_idx_x = packed_param.sub_block_mappling[blockIdx.x];
  const IdType local_grid_dim_x = packed_param.block_num_prefix_sum[link_idx + 1] - packed_param.block_num_prefix_sum[link_idx];
  size_t i = local_block_idx_x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * local_grid_dim_x;


  const IdType num_node = packed_param.num_node_array[link_idx];
  auto & data_iter = packed_param.data_iter_array[link_idx];

  // if ((packed_param.num_node_array[0] % 20) == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
  //   printf("Block[%d]/[%d], link[%d], Local Block idx=%d, local_grid_dim=%d, block duty size=%d, stride=%d\n",
  //     blockIdx.x, gridDim.x, link_idx, local_block_idx_x, local_grid_dim_x, num_node, stride);
  // }

  // if ((packed_param.num_node_array[0] % 20) == 0 && threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("\n");
  // }

  while (i < num_node) {
    size_t col = threadIdx.x;
    T* dst = data_iter.template dst<T>(i);
    const T* src = data_iter.template src<T>(i);
    while (col < packed_param.dim) {
      dst[col] = src[col];
      col += blockDim.x;
    }
    i += stride;
  }
}

#ifdef DEAD_CODE
struct LocationIter {
  SrcKey* src_key;
  LocationIter() {}
  LocationIter(SrcKey* src_key) : src_key(src_key) {}
  LocationIter(const SrcKey* src_key) : src_key(const_cast<SrcKey*>(src_key)) {}
  __forceinline__ __host__ __device__ int & operator[](const size_t & idx) { return src_key[idx]._location_id; }
  __forceinline__ __host__ __device__ const int & operator[](const size_t & idx) const { return src_key[idx]._location_id; }
};
struct SrcOffIter {
  DstVal* dst_val;
  SrcOffIter() {}
  SrcOffIter(DstVal* dst_val) : dst_val(dst_val) {}
  SrcOffIter(const DstVal* dst_val) : dst_val(const_cast<DstVal*>(dst_val)) {}
  __forceinline__ __host__ __device__ IdType & operator[](const size_t & idx) { return dst_val[idx]._src_offset; }
  __forceinline__ __host__ __device__ const IdType & operator[](const size_t & idx) const { return dst_val[idx]._src_offset; }
};
struct FreeOffIter {
  IdType* off_list;
  FreeOffIter() {}
  FreeOffIter(IdType* off_list) : off_list(off_list) {}
  FreeOffIter(const IdType* off_list) : off_list(const_cast<IdType*>(off_list)) {}
  __forceinline__ __host__ __device__ IdType & operator[](const size_t & idx) { return off_list[idx]; }
  __forceinline__ __host__ __device__ const IdType & operator[](const size_t & idx) const { return off_list[idx]; }
};

struct DstOffIter {
  DstVal* dst_val;
  DstOffIter() {}
  DstOffIter(DstVal* dst_val) : dst_val(dst_val) {}
  DstOffIter(const DstVal* dst_val) : dst_val(const_cast<DstVal*>(dst_val)) {}
  __forceinline__ __host__ __device__ IdType & operator[](const size_t & idx) { return dst_val[idx]._dst_offset; }
  __forceinline__ __host__ __device__ const IdType & operator[](const size_t & idx) const { return dst_val[idx]._dst_offset; }
};
struct DirectOffIter {
  __forceinline__ __host__ __device__ size_t operator[](const size_t & idx) const { return idx; }
};

// used when building cache with empty_feat
struct MockOffIter {
  size_t empty_feat;
  MockOffIter() { empty_feat = RunConfig::option_empty_feat; }
  __forceinline__ __host__ __device__ size_t operator[](const size_t & idx) const { return idx % (1 << empty_feat); }
};
struct MockSrcOffIter {
  size_t empty_feat;
  const IdType* idx_list;
  MockSrcOffIter(const IdType* idx_list) : idx_list(idx_list) { empty_feat = RunConfig::option_empty_feat; }
  __forceinline__ __host__ __device__ size_t operator[](const size_t & idx) const { return idx_list[idx] % (1 << empty_feat); }
};

template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename LocIter_T, typename SrcOffIter_T, typename DstOffIter_T>
__global__ void get_miss_cache_index(
    LocIter_T location_iter, SrcOffIter_T src_offset_iter, DstOffIter_T dst_offset_iter,
    const IdType* nodes, const size_t num_nodes,
    const HashTableEntryLocation* hash_table_location,
    const HashTableEntryOffset* hash_table_offset) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t dst_idx = block_start + threadIdx.x; dst_idx < block_end;
       dst_idx += BLOCK_SIZE) {
    if (dst_idx < num_nodes) {
      const IdType node_id = nodes[dst_idx];
      location_iter[dst_idx] = hash_table_location[node_id];
      src_offset_iter[dst_idx] = hash_table_offset[node_id];
      dst_offset_iter[dst_idx] = dst_idx;
      // output_src_index[dst_idx]._location_id = 0;
      // output_dst_index[dst_idx]._src_offset = 0;
      // output_dst_index[dst_idx]._dst_offset = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename LocIter_T>
__global__ void get_location(
    LocIter_T location_iter,
    const IdType* nodes, const size_t num_nodes,
    const HashTableEntryLocation* hash_table_location) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t dst_idx = block_start + threadIdx.x; dst_idx < block_end;
       dst_idx += BLOCK_SIZE) {
    if (dst_idx < num_nodes) {
      const IdType node_id = nodes[dst_idx];
      location_iter[dst_idx] = hash_table_location[node_id];
    }
  }
}
#endif


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize, 
    typename IdxStore_T>
__global__ void find_boundary(
    IdxStore_T location_iter, const size_t len,
    IdType* boundary_list) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t src_offset = block_start + threadIdx.x; src_offset < block_end;
       src_offset += BLOCK_SIZE) {
    if (src_offset < len) {
      if (src_offset == len-1 || location_iter.src_loc(src_offset) != location_iter.src_loc(src_offset+1)) {
        boundary_list[location_iter.src_loc(src_offset)+1] = src_offset+1;
      } 
      // if (src_offset == 0 || output_src_index[src_offset]._location_id != output_src_index[src_offset-1]._location_id) {
      //   boundary_list[output_src_index[src_offset]._location_id] = src_offset;
      // }
    }
  }
}

#ifdef DEAD_CODE
template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize, bool USE_EMPTY_FEAT=false>
__global__ void init_hash_table_cpu(
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    const size_t num_total_nodes,
    const int cpu_location_id, size_t empty_feat = 0) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t node_id = block_start + threadIdx.x; node_id < block_end; node_id += BLOCK_SIZE) {
    if (node_id < num_total_nodes) {
      hash_table_location[node_id] = cpu_location_id;
      if (USE_EMPTY_FEAT == false) {
        hash_table_offset[node_id] = node_id;
      } else {
        hash_table_offset[node_id] = node_id % (1 << empty_feat);
      }
    }
  }
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void init_hash_table_local(
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    const IdType* local_nodes, const size_t num_node,
    const int local_location_id) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t offset = block_start + threadIdx.x; offset < block_end; offset += BLOCK_SIZE) {
    if (offset < num_node) {
      IdType node_id = local_nodes[offset];
      hash_table_location[node_id] = local_location_id;
      hash_table_offset[node_id] = offset;
    }
  }
}

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void init_hash_table_remote(
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    const HashTableEntryOffset* remote_hash_table_offset,
    const IdType* remote_nodes, const size_t num_node,
    const int remote_location_id) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t offset = block_start + threadIdx.x; offset < block_end; offset += BLOCK_SIZE) {
    if (offset < num_node) {
      IdType node_id = remote_nodes[offset];
      hash_table_location[node_id] = remote_location_id;
      hash_table_offset[node_id] = remote_hash_table_offset[node_id];
    }
  }
}



template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize, bool USE_EMPTY_FEAT=false>
__global__ void decide_source_location(const IdType* nid_to_block_id, const uint8_t* block_is_stored, 
    const int *bit_array_to_src_location,
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    IdType num_total_nodes, IdType num_blocks, 
    IdType local_location_id, IdType cpu_location_id, size_t empty_feat = 0) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (IdType nid = block_start + threadIdx.x; nid < block_end; nid += BLOCK_SIZE) {
    if (nid < num_total_nodes) {
      IdType block_id = nid_to_block_id[nid];
      uint8_t is_stored_bit_array = block_is_stored[block_id];
      hash_table_location[nid] = bit_array_to_src_location[is_stored_bit_array];
      if (USE_EMPTY_FEAT) {
        hash_table_offset[nid] = nid % (1 << empty_feat);
      } else {
        hash_table_offset[nid] = nid;
      }
      // assert((is_stored_bit_array == 0 && bit_array_to_src_location[is_stored_bit_array] == cpu_location_id) ||
      //        ((is_stored_bit_array & (1 << bit_array_to_src_location[is_stored_bit_array])) != 0));
      // if (is_stored_bit_array == 0) {
      //   hash_table_location[nid] = cpu_location_id;
      //   hash_table_offset[nid]   = nid;
      // } else if (is_stored_bit_array & (1 << local_location_id)) {
      //   hash_table_location[nid] = local_location_id;
      //   hash_table_offset[nid]   = nid;
      // } else {
      //   hash_table_location[nid] = log2(is_stored_bit_array & ((~is_stored_bit_array) + 1));
      //   hash_table_offset[nid]   = nid;
      // }
    }
  }
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize, bool USE_EMPTY_FEAT=false>
__global__ void decide_source_location_advised(const IdType* nid_to_block_id, const uint8_t* block_access_advise, 
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    IdType num_total_nodes, IdType num_blocks, 
    IdType local_location_id, IdType cpu_location_id, size_t empty_feat = 0) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (IdType nid = block_start + threadIdx.x; nid < block_end; nid += BLOCK_SIZE) {
    if (nid < num_total_nodes) {
      IdType block_id = nid_to_block_id[nid];
      hash_table_location[nid] = block_access_advise[block_id];
      if (USE_EMPTY_FEAT) {
        hash_table_offset[nid] = nid % (1 << empty_feat);
      } else {
        hash_table_offset[nid] = nid;
      }
    }
  }
}

void PreDecideSrc(int num_bits, int local_id, int cpu_location_id, int * placement_to_src) {
  // auto g = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
  auto g = std::mt19937(RunConfig::seed);
  auto count_bits_fn = [](int a) {
    int count = 0;
    while (a) {
      a &= a-1;
      count ++;
    }
    return count;
  };
  for (int placement = 0; placement < (1 << num_bits); placement++) {
    if (placement == 0) {
      placement_to_src[placement] = cpu_location_id;
    } else if (placement & (1 << local_id)) {
      placement_to_src[placement] = local_id;
    } else {
      // randomly choose a 1...
      int num_nz = count_bits_fn(placement);
      int location = 0;
      for (; location < num_bits; location++) {
        if ((placement & (1 << location)) == 0) continue;
        int choice = std::uniform_int_distribution<int>(1, num_nz)(g);
        if (choice == 1) {
          break;
        } else {
          num_nz --;
        }
      }
      placement_to_src[placement] = location;
    }
  }
}
#endif

}  // namespace

#ifdef DEAD_CODE
void CacheContext::lookup() {}

namespace {

struct SelectByLoc {
  const HashTableEntryLocation* loc_table;
  const HashTableEntryLocation expected_loc;
  const IdType* input_keys;
  CUB_RUNTIME_FUNCTION __forceinline__
  SelectByLoc(const HashTableEntryLocation* loc_table, 
              const HashTableEntryLocation expected_loc, const IdType* input_keys) : loc_table(loc_table), expected_loc(expected_loc), input_keys(input_keys) {}
  template<typename Distance>
  __host__ __device__  __forceinline__
  bool operator()(const Distance & idx) const {
    return loc_table[input_keys[idx]] == expected_loc;
  }
};

struct DstValOutputIter {
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = IdType;
  using pointer           = IdType*;  // or also value_type*
  using reference         = IdType&;  // or also value_type&

  DstVal* output;
  CUB_RUNTIME_FUNCTION __forceinline__
  DstValOutputIter(DstVal* output) : output(output) {}
  template<typename Distance>
  __host__ __device__  __forceinline__
  IdType & operator[](const Distance & idx) {
    return output[idx]._dst_offset;
  }
};


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void get_src_offset(
    const IdType* reordered_idx_list,
    DstVal* output,
    const IdType* keys, const size_t num_node,
    const HashTableEntryOffset* hash_table_offset) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += BLOCK_SIZE) {
    if (idx < num_node) {
      IdType idx_for_key = reordered_idx_list[idx];
      output[idx]._src_offset = hash_table_offset[keys[idx_for_key]];
      output[idx]._dst_offset = idx_for_key;
    }
  }
}

};
#endif

#ifdef COLL_HASH_VALID_LEGACY
void ExtractSession::GetMissCacheIndexByCub(DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes,
    IdType * & group_offset,
    StreamHandle stream) {

  auto cu_stream = static_cast<cudaStream_t>(stream);
  if (num_nodes == 0) return;
  if (output_dst_index_handle == nullptr || output_dst_index_handle->nbytes() < num_nodes * sizeof(SrcKey)) {
    output_dst_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(DstVal));
    output_src_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
  }

  output_dst_index = output_dst_index_handle->ptr<DstVal>();
  group_offset = this->_group_offset;

  auto reordered_idx_list = output_src_index_handle->ptr<IdType>();

  this->_group_offset[0] = 0;

  for (int loc_id = 0; loc_id < _cache_ctx->_num_location; loc_id++) {
    cub::CountingInputIterator<IdType> counter(0);
    SelectByLoc selector(_cache_ctx->_hash_table_location, loc_id, nodes);
    size_t workspace_bytes;
    cub::DeviceSelect::If(nullptr,   workspace_bytes, counter, reordered_idx_list + _group_offset[loc_id], _cache_ctx->d_num_selected_out, num_nodes, selector, cu_stream);
    if (workspace_handle == nullptr || workspace_handle->nbytes() < workspace_bytes) {
      workspace_handle = _cache_ctx->_gpu_mem_allocator(workspace_bytes);
    }
    void *workspace = workspace_handle->ptr();
    cub::DeviceSelect::If(workspace, workspace_bytes, counter, reordered_idx_list + _group_offset[loc_id], _cache_ctx->d_num_selected_out, num_nodes, selector, cu_stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    this->_group_offset[loc_id + 1] = this->_group_offset[loc_id] + *(_cache_ctx->d_num_selected_out);
  }
  CHECK(this->_group_offset[_cache_ctx->_num_location] == num_nodes) << this->_group_offset[_cache_ctx->_num_location] << "!=" << num_nodes;

  {
    SAM_CUDA_PREPARE_1D(num_nodes);
    get_src_offset<><<<grid, block, 0, cu_stream>>>(reordered_idx_list, output_dst_index, nodes, num_nodes, _cache_ctx->_hash_table_offset);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }
}
#endif

template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void check_eq(const uint32_t * a, const uint32_t * b, const size_t n_elem) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t offset = block_start + threadIdx.x; offset < block_end; offset += BLOCK_SIZE) {
    if (offset < n_elem) {
      assert(a[offset] == b[offset]);
    }
  }
}
void CheckCudaEqual(const void * a, const void* b, const size_t nbytes, StreamHandle stream) {
  CHECK(nbytes % 4 == 0);
  const size_t n_elem = nbytes / 4;
  // {
  //   SAM_CUDA_PREPARE_1D(n_elem);
  //   check_eq<><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  // }

  {
    const size_t num_tiles = 1; // RoundUpDiv((n_elem), Constant::kCudaTileSize);
    const dim3 grid(num_tiles);
    const dim3 block(4);

    check_eq<4, 1000000><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  }
  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)stream));
}

template<typename IdxStore_T>
void ExtractSession::GetMissCacheIndex(
    IdxStore_T & idx,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  if (num_nodes == 0) return;
  if (idx_store_handle == nullptr || idx_store_handle->nbytes() < idx.required_mem(num_nodes)) {
    idx_store_handle       = _cache_ctx->_gpu_mem_allocator(idx.required_mem(num_nodes));
    idx_store_alter_handle = _cache_ctx->_gpu_mem_allocator(idx.required_mem(num_nodes));
  }
  idx.prepare_mem(idx_store_handle->ptr<uint8_t>(), num_nodes);

  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - getting miss/hit index...";
  Timer t0;
  {
    Timer t_hash;
    _cache_ctx->_new_hash_table->LookupSrcDst(nodes, num_nodes, idx, _cpu_location_id, stream);
    device->StreamSync(_cache_ctx->_trainer_ctx, stream);
    LOG(DEBUG) << "hashtable get idx " << t_hash.Passed();
  }

  Timer t1;
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - sorting according to group...";
  IdxStore_T idx_alter;
  idx_alter.prepare_mem(idx_store_alter_handle->ptr<uint8_t>(), num_nodes);
  common::cuda::CubSortDispatcher(idx.keys_for_sort(), idx_alter.keys_for_sort(), idx.vals_for_sort(), idx_alter.vals_for_sort(), num_nodes, _cache_ctx->_trainer_ctx, _cache_ctx->_gpu_mem_allocator, false, stream);

  {
    std::memset(_group_offset, 0, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  }
  CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - sorting according to group - done...";
  // std::cout << "coll sort index "<< t1.Passed() << "\n";

  Timer t2;

  // std::cout << "coll free workspace "<< t2.Passed() << "\n";
}

#ifdef DEAD_CODE
void ExtractSession::SortByLocation(
    IdType* & sorted_nodes,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  if (num_nodes == 0) return;
  if (output_src_index_handle == nullptr || output_src_index_handle->nbytes() < num_nodes * sizeof(SrcKey)) {
    output_src_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
    output_src_index_alter_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
    output_sorted_nodes_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(IdType));
  }

  // alias of location indicator
  SrcKey* output_src_index = output_src_index_handle->ptr<SrcKey>();
  SrcKey* output_src_index_alter = output_src_index_alter_handle->ptr<SrcKey>();
  sorted_nodes = output_sorted_nodes_handle->ptr<IdType>();

  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  LOG(DEBUG) << "CollCacheManager: SortByGroup - getting miss/hit index...";
  Timer t0;
  LocationIter location_iter(output_src_index);

  // fixme: add validation
#ifdef COLL_HASH_VALID_LEGACY
  get_location<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
    location_iter, nodes, num_nodes, _cache_ctx->_hash_table_location);
#endif

  auto helper = HashTableLookupHelper::LocOnly(output_src_index_handle->ptr<IdType>(), _cache_ctx->_cpu_location_id);
  _cache_ctx->_new_hash_table->_hash_table->LookupValCustom(nodes, num_nodes, helper, stream);
  // device->StreamSync(_cache_ctx->_trainer_ctx, stream);

  Timer t1;

  size_t workspace_bytes;
  LOG(DEBUG) << "CollCacheManager: SortByGroup - sorting according to group...";

  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, workspace_bytes, 
      (int*)output_src_index, (int*)output_src_index_alter, nodes, sorted_nodes,
      num_nodes, 0, sizeof(SrcKey)*8, cu_stream));

  if (workspace_handle == nullptr || workspace_handle->nbytes() < workspace_bytes) {
    workspace_handle = _cache_ctx->_gpu_mem_allocator(workspace_bytes);
  }
  void *workspace = workspace_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, 
      (int*)output_src_index, (int*)output_src_index_alter, nodes, sorted_nodes,
      num_nodes, 0, sizeof(SrcKey)*8, cu_stream));

  // device->StreamSync(_cache_ctx->_trainer_ctx, stream);
  LOG(DEBUG) << "CollCacheManager: SortByGroup - sorting according to group - done...";
}
#endif

template<typename IdxStore_T>
void ExtractSession::SplitGroup(const IdxStore_T idx_store, const size_t len, IdType * & group_offset, StreamHandle stream){
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  const size_t num_tiles = RoundUpDiv(len, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  Timer t0;
  group_offset = this->_group_offset;
  group_offset[_cache_ctx->_num_location] = len;
  if (len == 0) return;
  LOG(DEBUG) << "CollCache: SplitGroup: legacy finding offset...";
  find_boundary<><<<grid, block, 0, cu_stream>>>(idx_store, len, group_offset);
  device->StreamSync(_cache_ctx->_trainer_ctx, stream);
  LOG(DEBUG) << "CollCache: SplitGroup: legacy fixing offset...";
  for (int i = 1; i < _cache_ctx->_num_location; i++) {
    if (group_offset[i+1] == 0) {
      group_offset[i+1] = group_offset[i];
    }
  }
  // std::cout << "coll split group "<< t0.Passed() << "\n";
  LOG(DEBUG) << "CollCache: SplitGroup: legacy fixing done...";
}

namespace {
template <typename DataIter_T>
void Combine(const DataIter_T data_iter,
    const size_t num_node, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream, IdType limit_block=0, bool async=false);
}

template<typename IdxStore_T>
void ExtractSession::CombineOneGroup(const IdxStore_T idx_store, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block, bool async) {
  DataIterPerLoc<IdxStore_T> data_iter(idx_store, src_data, output, _cache_ctx->_dim);
  Combine(data_iter, num_node, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream, limit_block, async);
}

template<int NUM_LINK, typename IdxStore_T>
void ExtractSession::CombineConcurrent(const IdxStore_T idx_store, const IdType * group_offset, void* output, StreamHandle stream) {
  CHECK(NUM_LINK == RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size());
  ExtractConcurrentParam<NUM_LINK, DataIterPerLoc<IdxStore_T>> param;
  IdType total_required_num_sm = 0;
  TensorPtr link_mapping = Tensor::Empty(kI32, {108}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  TensorPtr sub_block_mappling = Tensor::Empty(kI32, {108}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  IdType total_num_node = 0;
  for (int i = 0; i < NUM_LINK; i++) {
    CHECK(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i].size() == 1);
    int dev_id = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i][0];
    int num_sm = RunConfig::coll_cache_link_desc.link_sm[_cache_ctx->_local_location_id][i];
    param.data_iter_array[i] = DataIterPerLoc<IdxStore_T>(idx_store.sub_array(group_offset[dev_id]), _cache_ctx->_device_cache_data[dev_id], output, _cache_ctx->_dim);
    param.num_node_array[i] = group_offset[dev_id + 1] - group_offset[dev_id];
    param.block_num_prefix_sum[i] = total_required_num_sm;
    total_required_num_sm += num_sm;
    for (int block_id = param.block_num_prefix_sum[i]; block_id < total_required_num_sm; block_id++) {
      link_mapping->Ptr<IdType>()[block_id] = i;
      sub_block_mappling->Ptr<IdType>()[block_id] = block_id - param.block_num_prefix_sum[i];
    }
    total_num_node += param.num_node_array[i];
  }
  if (total_num_node == 0) return;
  link_mapping = Tensor::CopyToExternal(link_mapping, _cache_ctx->_gpu_mem_allocator, _cache_ctx->_trainer_ctx, stream);
  sub_block_mappling = Tensor::CopyToExternal(sub_block_mappling, _cache_ctx->_gpu_mem_allocator, _cache_ctx->_trainer_ctx, stream);
  param.block_num_prefix_sum[NUM_LINK] = total_required_num_sm;
  param.link_mapping = link_mapping->CPtr<IdType>();
  param.sub_block_mappling = sub_block_mappling->CPtr<IdType>();
  param.dim = _cache_ctx->_dim;

  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(1024, 1);
  while (static_cast<size_t>(block.x) >= 2 * _cache_ctx->_dim) {
    block.x /= 2;
    block.y *= 2;
  }
  dim3 grid(total_required_num_sm);


  SWITCH_TYPE(_cache_ctx->_dtype, type, {
    extract_data_concurrent<NUM_LINK, type><<<grid, block, 0, cu_stream>>>(param);
  });

  device->StreamSync(_cache_ctx->_trainer_ctx, stream);
}

template<int NUM_LINK, typename IdxStore_T>
void ExtractSession::CombineFused(const IdxStore_T idx_store, const IdType * group_offset, void* output, StreamHandle stream) {
  dim3 block(1024, 1);
  while (static_cast<size_t>(block.x) >= 2 * _cache_ctx->_dim) {
    block.x /= 2;
    block.y *= 2;
  }

  CHECK(NUM_LINK == RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size());
  ExtractConcurrentParam<NUM_LINK, DataIterPerLoc<IdxStore_T>> param;
  IdType total_required_num_block = 0;
  TensorPtr link_mapping = Tensor::Empty(kI32, {RoundUpDiv(group_offset[this->_cache_ctx->_num_location] - group_offset[0], block.y * 4) * 2}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  TensorPtr sub_block_mappling = Tensor::Empty(kI32, {RoundUpDiv(group_offset[this->_cache_ctx->_num_location] - group_offset[0], block.y * 4) * 2}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  IdType total_num_node = 0;
  for (int i = 0; i < NUM_LINK; i++) {
    CHECK(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i].size() == 1);
    int dev_id = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i][0];
    param.data_iter_array[i] = DataIterPerLoc<IdxStore_T>(idx_store.sub_array(group_offset[dev_id]), _cache_ctx->_device_cache_data[dev_id], output, _cache_ctx->_dim);
    param.num_node_array[i] = group_offset[dev_id + 1] - group_offset[dev_id];
    int num_block = RoundUpDiv(static_cast<size_t>(param.num_node_array[i]), static_cast<size_t>(block.y * 4));
    param.block_num_prefix_sum[i] = total_required_num_block;
    total_required_num_block += num_block;
    for (int block_id = param.block_num_prefix_sum[i]; block_id < total_required_num_block; block_id++) {
      link_mapping->Ptr<IdType>()[block_id] = i;
      sub_block_mappling->Ptr<IdType>()[block_id] = block_id - param.block_num_prefix_sum[i];
    }
    total_num_node += param.num_node_array[i];
  }
  if (total_num_node == 0) return;
  CHECK(link_mapping->Shape()[0] >= total_required_num_block);
  std::vector<size_t> mapping(total_required_num_block);
  cpu::ArrangeArray(mapping.data(), total_required_num_block);
  std::random_shuffle(mapping.begin(), mapping.end());
  TensorPtr new_link_mapping = Tensor::Empty(kI32, {total_required_num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  TensorPtr new_sub_block_mappling = Tensor::Empty(kI32, {total_required_num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  for (size_t i = 0; i < total_required_num_block; i++) {
    new_link_mapping->Ptr<IdType>()[i] = link_mapping->Ptr<IdType>()[mapping[i]];
    new_sub_block_mappling->Ptr<IdType>()[i] = sub_block_mappling->Ptr<IdType>()[mapping[i]];
  }
  link_mapping = new_link_mapping;
  sub_block_mappling = new_sub_block_mappling;
  link_mapping = Tensor::CopyToExternal(link_mapping, _cache_ctx->_gpu_mem_allocator, _cache_ctx->_trainer_ctx, stream);
  sub_block_mappling = Tensor::CopyToExternal(sub_block_mappling, _cache_ctx->_gpu_mem_allocator, _cache_ctx->_trainer_ctx, stream);
  param.block_num_prefix_sum[NUM_LINK] = total_required_num_block;
  param.link_mapping = link_mapping->CPtr<IdType>();
  param.sub_block_mappling = sub_block_mappling->CPtr<IdType>();
  param.dim = _cache_ctx->_dim;

  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 grid(total_required_num_block);

  SWITCH_TYPE(_cache_ctx->_dtype, type, {
    extract_data_concurrent<NUM_LINK, type><<<grid, block, 0, cu_stream>>>(param);
  });

  device->StreamSync(_cache_ctx->_trainer_ctx, stream);
}

namespace {
#ifdef DEAD_CODE
template<typename T>
__global__ void extraction_kernel_random_ref(const T* src, T* dst, const uint32_t* index, size_t num_item_dot_dim, size_t feat_dim) {
  uint32_t linearIdx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t num_item = num_item_dot_dim / feat_dim;
  for (uint32_t i = linearIdx; i < num_item_dot_dim; i += blockDim.x * gridDim.x) {
    uint32_t dstIdx = i / feat_dim ;
    uint32_t offset = i % feat_dim ;
    uint32_t dstStart = (index[dstIdx] % num_item) * feat_dim;
    uint32_t srcStart = (index[dstIdx]) * feat_dim ;
    dst[dstStart + offset] = src[srcStart + offset];
  }
}

template<typename DType>
void gpu_extraction_random(const DType* src, DType* dst, uint32_t* index, size_t num_item, size_t feat_dim, cudaStream_t stream, size_t block_limit) {
  dim3 block(1024, 1);
  dim3 grid(RoundUpDiv(num_item * feat_dim, static_cast<size_t>(block.x)));
  if (block_limit != 0) {
    grid.x = block_limit;
  }
  extraction_kernel_random_ref<<<grid, block, 0, stream>>>(src, dst, index, num_item * feat_dim, feat_dim);
}
#endif

template <typename T, typename DataIter_T>
__global__ void extract_data_revised(
      DataIter_T data_iter,
      const size_t num_item, const size_t feat_dim) {
  size_t linearIdx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t num_item_dot_dim = num_item * feat_dim;

  for (size_t i = linearIdx; i < num_item_dot_dim; i += blockDim.x * gridDim.x) {
    size_t item_id = i / feat_dim ;
    size_t col = i % feat_dim ;
    assert(item_id < num_item);
    assert(col < feat_dim);
    T* dst = data_iter.template dst<T>(item_id);
    const T* src = data_iter.template src<T>(item_id);

    dst[col] = src[col];
  }
}

template <typename DataIter_T>
void CombineRevised(DataIter_T data_iter,
    const size_t num_node, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream, IdType limit_block, bool async) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const dim3 block(1024, 1);
  dim3 grid(RoundUpDiv(num_node * _dim, static_cast<size_t>(block.x)));
  if (limit_block != 0) {
    grid.x = limit_block;
  }

  SWITCH_TYPE(_dtype, type, {
      extract_data_revised<type><<<grid, block, 0, cu_stream>>>(
          data_iter, num_node, _dim);
  });

  if (async == false) {
    device->StreamSync(_trainer_ctx, stream);
  }
}

template <typename T, typename SrcDataIter_T, typename DstDataIter_T>
__global__ void extract_data_wg(
      const SrcDataIter_T full_src, DstDataIter_T dst_index,
      const size_t num_item, const size_t feat_dim) {
  size_t item_id = blockIdx.x;
  size_t col = threadIdx.x;
  T* dst = dst_index.template operator[]<T>(item_id);
  const T* src = full_src.template operator[]<T>(item_id);
  dst[col] = src[col];
}
template <typename T, typename DataIterRRSrc_T>
__global__ void extract_data_cliq(
      DataIterRRSrc_T src, DataIterCliqDst dst, const IdType* key_list,
      const size_t num_item, const size_t feat_dim) {
  T* dst_entry =  dst.call<T>(blockIdx.x, feat_dim);
  const T* src_entry = src.template call<T>(key_list[blockIdx.x], feat_dim);
  dst_entry[threadIdx.x] = src_entry[threadIdx.x];
}

template <typename SrcDataIter_T, typename DstDataIter_T>
void CombineWG(const SrcDataIter_T src_data_iter, DstDataIter_T dst_data_iter,
    const size_t num_node, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream, bool async) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const dim3 block(_dim);
  dim3 grid(num_node);

  SWITCH_TYPE(_dtype, type, {
      extract_data_wg<type><<<grid, block, 0, cu_stream>>>(
          src_data_iter, dst_data_iter, num_node, _dim);
  });

  if (async == false) {
    device->StreamSync(_trainer_ctx, stream);
  }
}

template <typename DataIter_T>
void Combine(const DataIter_T data_iter,
    const size_t num_node, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream, IdType limit_block, bool async) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  dim3 block(1024, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  dim3 grid(RoundUpDiv(num_node, static_cast<size_t>(block.y * 4)));
  if (limit_block != 0) {
    grid.x = limit_block;
  }

  SWITCH_TYPE(_dtype, type, {
      extract_data<type><<<grid, block, 0, cu_stream>>>(data_iter, num_node, _dim);
  });

  if (async == false) {
    device->StreamSync(_trainer_ctx, stream);
  }
}
}

template<typename IdxStore_T>
void ExtractSession::CombineOneGroupRevised(const IdxStore_T idx_store, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block, bool async) {
  DataIterPerLoc<IdxStore_T> data_iter(idx_store, src_data, output, _cache_ctx->_dim);
  CombineRevised<>(data_iter, num_node, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream, limit_block, async);
}
#ifdef COLL_HASH_VALID_LEGACY
void ExtractSession::CombineNoGroup(const IdType * nodes, const size_t num_node, void* output, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  // dim3 block(256, 1);
  dim3 block(1024, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_node, static_cast<size_t>(block.y * 4)));

  const DataIterMultiLocation<const IdType*> src_iter(nodes, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, _cache_ctx->_device_cache_data, _dim);
  DataIter<DirectOffIter> dst_iter(DirectOffIter(), output, _dim);

  SWITCH_TYPE(_dtype, type, {
      extract_data<type><<<grid, block, 0, cu_stream>>>(src_iter, dst_iter, num_node, _dim);
  });

  device->StreamSync(_trainer_ctx, stream);
}
#endif
#ifdef DEAD_CODE
void ExtractSession::CombineMixGroup(const SrcKey* src_key, const DstVal* dst_val, const size_t num_node, void* output, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  // dim3 block(256, 1);
  dim3 block(1024, 1);
  while (static_cast<size_t>(block.x) >= 2 * _dim) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid(RoundUpDiv(num_node, static_cast<size_t>(block.y * 4)));

  // const dim3 block(_dim);
  // const dim3 grid(num_node);

  const DataIterMixLocation<LocationIter, SrcOffIter> src_iter(LocationIter(src_key), SrcOffIter(dst_val), _cache_ctx->_device_cache_data, _dim);
  DataIter<DirectOffIter> dst_iter(DirectOffIter(), output, _dim);

  SWITCH_TYPE(_dtype, type, {
      extract_data<type><<<grid, block, 0, cu_stream>>>(src_iter, dst_iter, num_node, _dim);
      // extract_data_wg<type><<<grid, block, 0, cu_stream>>>(src_iter, dst_iter, num_node, _dim);
  });

  device->StreamSync(_trainer_ctx, stream);
}
#endif
void ExtractSession::CombineCliq(const IdType* key_list, const size_t num_node, void* output, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream) {
  if (num_node == 0) return;
  auto device = Device::Get(_trainer_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  const dim3 block(_dim);
  const dim3 grid(num_node);

  auto clique_size = RunConfig::coll_cache_link_desc.CliqueSize();
  IdType chunk_size = RoundUpDiv<size_t>(RunConfig::num_total_item, clique_size);

  #define SWITCH_CLIQ_SIZE(val, alias, ...) \
    switch(val) {                     \
      case 4: { constexpr int alias = 4; { __VA_ARGS__ }; break; } \
      case 8: { constexpr int alias = 8; { __VA_ARGS__ }; break; } \
      default: CHECK(false);           \
    }

  #define SWITCH_CLIQ_HASH(val, alias, cliq_size, ...) \
    switch(val) {                     \
      case kRR:    { using alias= DataIterCliqSrcRR<cliq_size>       ; { __VA_ARGS__ }; break; } \
      case kChunk: { using alias= DataIterCliqSrcChunk<cliq_size>    ; { __VA_ARGS__ }; break; } \
      case kDefault: \
      default: CHECK(false);           \
    }

  SWITCH_CLIQ_SIZE(clique_size, cliq_size, {
    SWITCH_CLIQ_HASH(RunConfig::coll_hash_impl, DATA_ITER_T, cliq_size, {
      DATA_ITER_T src(clique_size, _cache_ctx->_device_cache_data_clique, _dim, chunk_size);
      DataIterCliqDst dst(output, _dim);
      SWITCH_TYPE(_dtype, type, {
          extract_data_cliq<type><<<grid, block, 0, cu_stream>>>(src, dst, key_list, num_node, _dim);
      });
    });
  });

  device->StreamSync(_trainer_ctx, stream);
}

void ExtractSession::ExtractFeat(const IdType* nodes, const size_t num_nodes,
                  void* output, StreamHandle stream, uint64_t task_key) {
  if (_cache_ctx->IsDirectMapping()) {
    CHECK(_cache_ctx->_num_location == 1);
    // fast path
    // direct mapping from node id to freature, no need to go through hashtable
    LOG(DEBUG) << "CollCache: ExtractFeat: Direct mapping, going fast path... ";
    Timer t0;
    CUDA_CALL(cudaGetLastError());
    /**
     * need to distinguish 0% or 100%. for host, we expect empty feat to work. for local, we expect to ignore empty feat.
     */
    SWITCH_BOOL(RunConfig::option_empty_feat != 0 && _cache_ctx->_dim != 1 && _cache_ctx->_cache_space_capacity == 0, use_empty_feat, {
      IdxStoreDirect<use_empty_feat> idx_store(nodes);
      DataIterPerLoc<decltype(idx_store)> data_iter(idx_store, _cache_ctx->_device_cache_data[0], output, _cache_ctx->_dim);
      Combine(data_iter, num_nodes, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);
    })
    double combine_time = t0.Passed();
    if (task_key != 0xffffffffffffffff) {
      size_t nbytes = GetTensorBytes(_cache_ctx->_dtype, {num_nodes, _cache_ctx->_dim});
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1FeatureBytes, nbytes);
      if (_cache_ctx->_cpu_location_id == -1) {
        // full cache
        _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineCacheTime,combine_time);
      } else {
        // no cache
        _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1MissBytes, nbytes);
        _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineMissTime,combine_time);
      }
      _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochFeatureBytes,nbytes);
      _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochMissBytes, nbytes);
    }
    CUDA_CALL(cudaGetLastError());
  } else if (IsLegacy()) {
#ifdef DEAD_CODE
    auto trainer_gpu_device = Device::Get(_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    SrcKey * src_index = nullptr;
    DstVal * dst_index = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: legacy, get miss cache index... ";
    Timer t0;
    GetMissCacheIndex(src_index, dst_index, nodes, num_nodes, stream);
    // std::cout << "Get Idx " << t0.Passed() << "\n";
    Timer t1;
    IdType * group_offset = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: legacy, splitting group... ";
    SplitGroup(src_index, num_nodes, group_offset, stream);
    double get_index_time = t0.Passed();
    // std::cout << "Split GrOup " <<t1.Passed() << "\n";
    double combine_times[2];
    for (int src_device_id = 1; src_device_id >= 0; src_device_id --) {
      LOG(DEBUG) << "CollCache: ExtractFeat: legacy, combining group " << src_device_id << " [" << group_offset[src_device_id] << "," << group_offset[src_device_id+1] << ")...";
      Timer t1;
      CombineOneGroup(
          src_index + group_offset[src_device_id], dst_index + group_offset[src_device_id], 
          nodes + group_offset[src_device_id], 
          group_offset[src_device_id+1] - group_offset[src_device_id], 
          _device_cache_data[src_device_id], output, stream);
      combine_times[src_device_id] = t1.Passed();
    }
    trainer_gpu_device->FreeWorkspace(_trainer_ctx, src_index);
    trainer_gpu_device->FreeWorkspace(_trainer_ctx, dst_index);
    if (task_key != 0xffffffffffffffff) {
      size_t num_miss = group_offset[2]- group_offset[1];
      // size_t num_hit = group_offset[1];
      Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[1]);
      Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[0]);
      Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
    cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
#endif
  } else if (RunConfig::coll_skip_hash) {
    auto trainer_gpu_device = Device::Get(_cache_ctx->_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    // Timer t0;
    // double get_index_time = t0.Passed();
    Timer t1;
    CombineCliq(nodes, num_nodes, output, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);
    double combine_time = t1.Passed();
    if (task_key != 0xffffffffffffffff) {
      // size_t num_hit = group_offset[1];
      auto _dtype = _cache_ctx->_dtype;
      auto _dim = _cache_ctx->_dim;
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineCacheTime,combine_time);
      _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
  } else if (RunConfig::concurrent_link_impl == common::kDirectNoGroup || RunConfig::concurrent_link_impl == common::kOrderedNoGroup) {
    // CHECK(false) << "Multi source extraction is not supported now";
    auto trainer_gpu_device = Device::Get(_cache_ctx->_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    // Timer t0;
    // double get_index_time = t0.Passed();
    Timer t0;
    // if (RunConfig::coll_cache_no_group == common::kOrderedNoGroup) {
    //   IdType* sorted_nodes;
    //   SortByLocation(sorted_nodes, nodes, num_nodes, stream);
    //   nodes = sorted_nodes;
    // }
    IdxStore idx;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, get miss cache index... ";
    GetMissCacheIndex(idx, nodes, num_nodes, stream);
    double get_index_time = t0.Passed();
    Timer t1;
    DataIterMixLoc<decltype(idx)> data_iter(idx, _cache_ctx->_device_cache_data, output, _cache_ctx->_dim);
    Combine(data_iter, num_nodes, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);
    double combine_time = t1.Passed();
    if (task_key != 0xffffffffffffffff) {
      // size_t num_hit = group_offset[1];
      auto _dtype = _cache_ctx->_dtype;
      auto _dim = _cache_ctx->_dim;
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
      // _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineCacheTime,combine_time);
      _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
  } else if (RunConfig::concurrent_link_impl != kNoConcurrentLink) {
    // CHECK(false) << "Multi source extraction is not supported now";
    auto trainer_gpu_device = Device::Get(_cache_ctx->_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, get miss cache index... ";
    Timer t0;
    // IdxStore idx_store;
    IdxStoreCompact idx_store;
    GetMissCacheIndex(idx_store, nodes, num_nodes, stream);
    // std::cout << "Get Idx " << t0.Passed() << "\n";
    IdType * group_offset = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, splitting group... ";
    SplitGroup(idx_store, num_nodes, group_offset, stream);
    // GetMissCacheIndexByCub(dst_index, nodes, num_nodes, group_offset, stream);
    double get_index_time = t0.Passed();

    // std::cout << "Split GrOup " <<t1.Passed() << "\n";
    double combine_times[3] = {0, 0, 0};
    if (RunConfig::concurrent_link_impl == common::kMPS) {
      // ┌─────┬──────────┬...┬────────────┐
      // │     │local  0/n│...│local  n-1/n│
      // │ cpu ├──────────┼...┼────────────┤
      // │     │remote 0/n│...│remote n-1/n│
      // └─────┴──────────┴...┴────────────┘
      // auto local_combine = [src_index, group_offset, dst_index, nodes, this, output, num_nodes](int link_id, StreamHandle stream){
      //   if (num_nodes == 0) return;
      //   auto & link_src = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id];
      //   size_t loc_id = this->_cache_ctx->_local_location_id;
      //   size_t local_total_num = group_offset[loc_id+1] - group_offset[loc_id];
      //   size_t local_part_num = RoundUpDiv(local_total_num, link_src.size());
      //   size_t part_begin = local_part_num * link_id;
      //   size_t part_end = std::min((link_id + 1) * local_part_num, local_total_num);
      //   CombineOneGroup(src_index + group_offset[loc_id] + part_begin, 
      //                   dst_index + group_offset[loc_id] + part_begin, 
      //                   nodes + group_offset[loc_id] + part_begin, 
      //                   part_end - part_begin, 
      //                   _cache_ctx->_device_cache_data[loc_id], output, stream, 0, true);
      //   CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
      // };

      auto call_combine = [idx_store, group_offset, this, output, num_nodes](int location_id, StreamHandle stream){
        if (group_offset[location_id+1] - group_offset[location_id] == 0) return;
        CombineOneGroup(idx_store.sub_array(group_offset[location_id]), 
                        group_offset[location_id+1] - group_offset[location_id], 
                        _cache_ctx->_device_cache_data[location_id], output, stream, 0, true);
        CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
      };
      // launch cpu extraction
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->v2_forward_one_step([&combine_times, call_combine, loc_id = _cache_ctx->_cpu_location_id](cudaStream_t cu_s){
        Timer t_cpu;
        call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
        combine_times[0] = t_cpu.Passed();
      });
      _cpu_syncer->on_send_job();
      // launch local extraction
      auto & link_src = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id];
      auto num_link = link_src.size();
      Timer t_local;
      this->_extract_ctx[_cache_ctx->_local_location_id]->v2_forward_one_step([call_combine, loc_id = _cache_ctx->_local_location_id](cudaStream_t cu_s){
        call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
      });
      _local_syncer->on_send_job()->on_wait_job_done();
      combine_times[2] = t_local.Passed();
      // launch remote extraction
      Timer t_remote;
      for (int i = 0; i < num_link; i++) {
        CHECK(link_src[i].size() == 1);
        int loc_id = link_src[i][0];
        this->_extract_ctx[loc_id]->v2_forward_one_step([call_combine, loc_id](cudaStream_t cu_s){
          call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
        });
      }
      _remote_syncer->on_send_job()->on_wait_job_done();
      combine_times[1] = t_remote.Passed();
      _cpu_syncer->on_wait_job_done();
    } else if (RunConfig::concurrent_link_impl == kMPSPhase ||
               RunConfig::concurrent_link_impl == kSMMaskPhase) {
      auto call_combine = [idx_store, group_offset, this, output, num_nodes](int location_id, StreamHandle stream, bool sync=true){
        if (group_offset[location_id+1] - group_offset[location_id] == 0) return;
        Timer t;
        // CombineOneGroupRevised
        CombineOneGroup(idx_store.sub_array(group_offset[location_id]),
                        group_offset[location_id+1] - group_offset[location_id],
                        _cache_ctx->_device_cache_data[location_id], output, stream, 0, true);
        if (sync) {
          CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        }
        phase_time_record[location_id] = t.Passed();
        accu_each_src_time[location_id] += t.Passed();
        accu_each_src_nkey[location_id] += group_offset[location_id+1] - group_offset[location_id];
      };
      // launch cpu extraction
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->v2_forward_one_step([&combine_times, call_combine, loc_id = _cache_ctx->_cpu_location_id](cudaStream_t cu_s){
        Timer t_cpu;
        call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
        combine_times[0] = t_cpu.Passed();
      });
      _cpu_syncer->on_send_job();
      auto & link_src = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id];
      auto num_link = link_src.size();

      Timer t_remote;
      Timer t_local;
      // set remote extraction lambda
      for (int i = 0; i < num_link; i++) {
        CHECK(link_src[i].size() == 1);
        int loc_id = link_src[i][0];
        this->_extract_ctx[loc_id]->v2_forward_one_step([call_combine, loc_id](cudaStream_t cu_s){
          call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
        });
      }
      // launch remote extraction
      _remote_syncer->on_send_job();
      usleep(60);
      // launch local extraction using this thread
      call_combine(_cache_ctx->_local_location_id, _local_ext_stream, false);

      _remote_syncer->on_wait_job_done();
      // combine_times[1] = t_remote.Passed();

      CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(_local_ext_stream)));
      combine_times[2] = t_local.Passed();
      for (int i = 0; i < num_link; i++) {
        int loc_id = link_src[i][0];
        combine_times[1] += phase_time_record[loc_id];
      }
      combine_times[1] /= num_link;
      combine_times[2] -= combine_times[1];

      _cpu_syncer->on_wait_job_done();
    } else if (RunConfig::concurrent_link_impl == kMPSForLandC) {
      auto call_combine = [idx_store, group_offset, this, output, num_nodes](int location_id, IdType num_sm, StreamHandle stream){
        // LOG(ERROR) << "combine from " << location_id << " with sm=" << num_sm;
        if (group_offset[location_id+1] - group_offset[location_id] == 0) return;
        CombineOneGroupRevised(idx_store.sub_array(group_offset[location_id]), 
                        group_offset[location_id+1] - group_offset[location_id], 
                        _cache_ctx->_device_cache_data[location_id], output, stream, num_sm, true);
        CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
      };

      auto & link_desc = RunConfig::coll_cache_link_desc;
      auto & link_src = link_desc.link_src[_cache_ctx->_local_location_id];
      int num_link = link_src.size();

      // launch cpu extraction
      CHECK(this->_extract_ctx[_cache_ctx->_cpu_location_id]->cu_ctx_ != nullptr);
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->v2_forward_one_step([&combine_times, &link_desc, this, call_combine, loc_id = _cache_ctx->_cpu_location_id](cudaStream_t cu_s){
        Timer t_cpu;
        call_combine(loc_id, 0, reinterpret_cast<StreamHandle>(cu_s));
        combine_times[0] = t_cpu.Passed();
      });
      _cpu_syncer->on_send_job();

      // launch local extraction
      CHECK(this->_extract_ctx[_cache_ctx->_local_location_id]->cu_ctx_ != nullptr);
      Timer t_local;
      this->_extract_ctx[_cache_ctx->_local_location_id]->v2_forward_one_step([&link_desc, this, call_combine, loc_id = _cache_ctx->_local_location_id](cudaStream_t cu_s){
        call_combine(loc_id, 0, reinterpret_cast<StreamHandle>(cu_s));
      });
      _local_syncer->on_send_job()->on_wait_job_done();
      combine_times[2] = t_local.Passed();

      // launch remote extraction
      Timer t_remote;
      for (int i = 0; i < num_link; i++) {
        CHECK(link_src[i].size() == 1);
        int loc_id = link_src[i][0];
        this->_extract_ctx[loc_id]->v2_forward_one_step([&link_desc, this, call_combine, loc_id, i](cudaStream_t cu_s){
          call_combine(loc_id, link_desc.link_sm[_cache_ctx->_local_location_id][i], reinterpret_cast<StreamHandle>(cu_s));
        });
      }
      _remote_syncer->on_send_job()->on_wait_job_done();
      combine_times[1] = t_remote.Passed();
      _cpu_syncer->on_wait_job_done();
    } else if (RunConfig::concurrent_link_impl == kMultiKernelNumBlock) {
      auto call_combine = [idx_store, group_offset, this, output, num_nodes](int location_id, IdType num_sm, StreamHandle stream){
        // LOG(ERROR) << "combine from " << location_id << " with sm=" << num_sm;
        if (group_offset[location_id+1] - group_offset[location_id] == 0) return;
        Timer t;
        CombineOneGroupRevised(idx_store.sub_array(group_offset[location_id]), 
                        group_offset[location_id+1] - group_offset[location_id], 
                        _cache_ctx->_device_cache_data[location_id], output, stream, num_sm, true);
        CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        accu_each_src_time[location_id] += t.Passed();
      };

      auto & link_desc = RunConfig::coll_cache_link_desc;
      auto & link_src = link_desc.link_src[_cache_ctx->_local_location_id];
      int num_link = link_src.size();

      // launch cpu extraction
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->v2_forward_one_step([&combine_times, &link_desc, this, call_combine, loc_id = _cache_ctx->_cpu_location_id](cudaStream_t cu_s){
        Timer t_cpu;
        call_combine(loc_id, link_desc.cpu_sm[_cache_ctx->_local_location_id], reinterpret_cast<StreamHandle>(cu_s));
        combine_times[0] = t_cpu.Passed();
      });
      _cpu_syncer->on_send_job();

      // launch local extraction
      Timer t_local;
      this->_extract_ctx[_cache_ctx->_local_location_id]->v2_forward_one_step([&link_desc, this, call_combine, loc_id = _cache_ctx->_local_location_id](cudaStream_t cu_s){
        call_combine(loc_id, link_desc.local_sm[_cache_ctx->_local_location_id], reinterpret_cast<StreamHandle>(cu_s));
      });
      _local_syncer->on_send_job()->on_wait_job_done();
      combine_times[2] = t_local.Passed();

      // launch remote extraction
      Timer t_remote;
      for (int i = 0; i < num_link; i++) {
        CHECK(link_src[i].size() == 1);
        int loc_id = link_src[i][0];
        this->_extract_ctx[loc_id]->v2_forward_one_step([&link_desc, this, call_combine, loc_id, i](cudaStream_t cu_s){
          call_combine(loc_id, link_desc.link_sm[_cache_ctx->_local_location_id][i], reinterpret_cast<StreamHandle>(cu_s));
        });
      }
      _remote_syncer->on_send_job()->on_wait_job_done();
      combine_times[1] = t_remote.Passed();
      _cpu_syncer->on_wait_job_done();
    } else if (RunConfig::concurrent_link_impl == kMultiKernelNumBlockOld) {
      auto call_combine = [idx_store, group_offset, this, output, num_nodes](int location_id, IdType num_sm, StreamHandle stream){
        // LOG(ERROR) << "combine from " << location_id << " with sm=" << num_sm;
        if (group_offset[location_id+1] - group_offset[location_id] == 0) return;
        CombineOneGroup(idx_store.sub_array(group_offset[location_id]), 
                        group_offset[location_id+1] - group_offset[location_id], 
                        _cache_ctx->_device_cache_data[location_id], output, stream, num_sm, true);
        CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
      };

      auto & link_desc = RunConfig::coll_cache_link_desc;
      auto & link_src = link_desc.link_src[_cache_ctx->_local_location_id];
      int num_link = link_src.size();

      // launch cpu extraction
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->v2_forward_one_step([&combine_times, &link_desc, this, call_combine, loc_id = _cache_ctx->_cpu_location_id](cudaStream_t cu_s){
        Timer t_cpu;
        call_combine(loc_id, link_desc.cpu_sm[_cache_ctx->_local_location_id], reinterpret_cast<StreamHandle>(cu_s));
        combine_times[0] = t_cpu.Passed();
      });
      _cpu_syncer->on_send_job();

      // launch local extraction
      Timer t_local;
      this->_extract_ctx[_cache_ctx->_local_location_id]->v2_forward_one_step([&link_desc, this, call_combine, loc_id = _cache_ctx->_local_location_id](cudaStream_t cu_s){
        call_combine(loc_id, link_desc.local_sm[_cache_ctx->_local_location_id], reinterpret_cast<StreamHandle>(cu_s));
      });
      _local_syncer->on_send_job()->on_wait_job_done();
      combine_times[2] = t_local.Passed();

      // launch remote extraction
      Timer t_remote;
      for (int i = 0; i < num_link; i++) {
        CHECK(link_src[i].size() == 1);
        int loc_id = link_src[i][0];
        this->_extract_ctx[loc_id]->v2_forward_one_step([&link_desc, this, call_combine, loc_id, i](cudaStream_t cu_s){
          call_combine(loc_id, link_desc.link_sm[_cache_ctx->_local_location_id][i], reinterpret_cast<StreamHandle>(cu_s));
        });
      }
      _remote_syncer->on_send_job()->on_wait_job_done();
      combine_times[1] = t_remote.Passed();
      _cpu_syncer->on_wait_job_done();
    } else {
      // cpu first, then concurrent remote, then local
      auto call_combine = [idx_store, group_offset, this, output](int location_id, StreamHandle stream){
        CombineOneGroup(idx_store.sub_array(group_offset[location_id]), 
                        group_offset[location_id+1] - group_offset[location_id], 
                        _cache_ctx->_device_cache_data[location_id], output, stream);
      };
      Timer t1;
      call_combine(_cache_ctx->_cpu_location_id, stream);
      combine_times[0] = t1.Passed();

      // DistEngine::Get()->GetTrainerBarrier()->Wait();

      {
        // impl1: single kernel, limited num block
        t1.Reset();
        if (RunConfig::concurrent_link_impl == kFused) {
          switch(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size()) {
            case 1: CombineFused<1>(idx_store, group_offset, output, stream); break;
            case 2: CombineFused<2>(idx_store, group_offset, output, stream); break;
            case 3: CombineFused<3>(idx_store, group_offset, output, stream); break;
            case 4: CombineFused<4>(idx_store, group_offset, output, stream); break;
            case 5: CombineFused<5>(idx_store, group_offset, output, stream); break;
            case 6: CombineFused<6>(idx_store, group_offset, output, stream); break;
            case 7: CombineFused<7>(idx_store, group_offset, output, stream); break;
            default: CHECK(false);
          }
        } else if (RunConfig::concurrent_link_impl == kFusedLimitNumBlock) {
          switch(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size()) {
            case 1: CombineConcurrent<1>(idx_store, group_offset, output, stream); break;
            case 2: CombineConcurrent<2>(idx_store, group_offset, output, stream); break;
            case 3: CombineConcurrent<3>(idx_store, group_offset, output, stream); break;
            case 4: CombineConcurrent<4>(idx_store, group_offset, output, stream); break;
            case 5: CombineConcurrent<5>(idx_store, group_offset, output, stream); break;
            case 6: CombineConcurrent<6>(idx_store, group_offset, output, stream); break;
            case 7: CombineConcurrent<7>(idx_store, group_offset, output, stream); break;
            default: CHECK(false);
          }
        }
        combine_times[1] = t1.Passed();
      }{
        // impl2: launch multiple kernel concurrently. each kernel only use part of sm by limiting num block
        // t1.Reset();
        // for (int src_link_order = 0; src_link_order < _remote_device_list.size(); src_link_order ++) {
        //   auto [dev_id, num_sm] = _remote_device_list[src_link_order];
        //   IdType offset = group_offset[dev_id];
        //   IdType link_num_node = group_offset[dev_id+1] - offset;
        //   // std::cout << "Dev[" << _local_location_id << "],Link[" << src_link_order << "], dev_id=" << dev_id << ", sm=" << num_sm << "," << "link_num_node=" << link_num_node << "\n"; 
        //   CombineOneGroup(src_index + offset, dst_index + offset, nodes + offset, link_num_node, _device_cache_data[dev_id], output, _concurrent_stream_array[src_link_order], num_sm, true);
        // }
        // for (int src_link_order = 0; src_link_order < _remote_device_list.size(); src_link_order ++) {
        //   trainer_gpu_device->StreamSync(_trainer_ctx, _concurrent_stream_array[src_link_order]);
        // }
        // combine_times[1] = t1.Passed();
      }{
        // impl3: use mps to launch multiple kernel concurrently. each kernel does not limit num block
        //   it's compute power is limited at mps ctx building
      }

      t1.Reset();
      call_combine(_cache_ctx->_local_location_id, stream);
      combine_times[2] = t1.Passed();
    }

    // output_src_index_handle = nullptr;
    // output_dst_index_handle = nullptr;
    if (task_key != 0xffffffffffffffff) {
      size_t num_miss = group_offset[_cache_ctx->_cpu_location_id+1]- group_offset[_cache_ctx->_cpu_location_id];
      size_t num_local = group_offset[_cache_ctx->_local_location_id+1] - group_offset[_cache_ctx->_local_location_id];
      size_t num_remote = num_nodes - num_miss - num_local;
      // size_t num_hit = group_offset[1];
      auto _dtype = _cache_ctx->_dtype;
      auto _dim = _cache_ctx->_dim;
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[2]);
      accu_cpu_time += combine_times[0];
      accu_remote_time += combine_times[1];
      accu_local_time += combine_times[2];
      accu_step ++;
      if (accu_step % 100 == 0) {
        // std::stringstream ss;
        // ss << _local_location_id << ":" << std::fixed << std::setw(10) << std::setprecision(6) 
        //    << std::setw(10) << accu_cpu_time / 100 
        //    << std::setw(10) << accu_remote_time / 100 
        //    << std::setw(10) << accu_local_time / 100  
        //    << " | "
        //    << std::setw(10) << accu_each_src_time[0] / 100
        //    << std::setw(10) << accu_each_src_time[1] / 100
        //    << std::setw(10) << accu_each_src_time[2] / 100
        //    << std::setw(10) << accu_each_src_time[3] / 100
        //    << std::setw(10) << accu_each_src_time[4] / 100
        //    << std::setw(10) << accu_each_src_time[5] / 100
        //    << std::setw(10) << accu_each_src_time[6] / 100
        //    << std::setw(10) << accu_each_src_time[7] / 100
        //    << std::setw(10) << accu_each_src_time[8] / 100
        //    << " | "
        //    << std::setw(10) << (int)(accu_each_src_nkey[0] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[1] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[2] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[3] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[4] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[5] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[6] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[7] / 100)
        //    << std::setw(10) << (int)(accu_each_src_nkey[8] / 100)
        //    << "\n";
        // ;
        // std::cerr << ss.str();
        // accu_cpu_time = 0;
        // accu_remote_time = 0;
        // accu_local_time = 0;
        // memset(accu_each_src_time, 0, sizeof(accu_each_src_time));
        // memset(accu_each_src_nkey, 0, sizeof(accu_each_src_nkey));
      }
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
    // cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
  } else {
    // CHECK(false) << "Multi source extraction is not supported now";
    auto trainer_gpu_device = Device::Get(_cache_ctx->_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    IdxStore idx_store;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, get miss cache index... ";
    Timer t0;
    GetMissCacheIndex(idx_store, nodes, num_nodes, stream);
    // std::cout << "Get Idx " << t0.Passed() << "\n";
    Timer t1;
    IdType * group_offset = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, splitting group... ";
    SplitGroup(idx_store, num_nodes, group_offset, stream);
    double get_index_time = t0.Passed();
    
    // std::cout << "Split GrOup " <<t1.Passed() << "\n";
    double combine_times[3] = {0, 0, 0};
    // cpu first, then remote, then local 

    auto call_combine = [idx_store, group_offset, this, output, stream, num_nodes](int location_id){
      if (num_nodes == 0) return;
      CombineOneGroup(idx_store.sub_array(group_offset[location_id]), 
                      group_offset[location_id+1] - group_offset[location_id], 
                      _cache_ctx->_device_cache_data[location_id], output, stream);
    };
    _cache_ctx->_barrier->Wait();
    Timer t_cpu;
    call_combine(_cache_ctx->_cpu_location_id);
    if (group_offset[_cache_ctx->_cpu_location_id+1] - group_offset[_cache_ctx->_cpu_location_id] != 0) {
    }
    _cache_ctx->_barrier->Wait();
    combine_times[0] = t_cpu.Passed();

    // DistEngine::Get()->GetTrainerBarrier()->Wait();
    {
      t1.Reset();
      // _cache_ctx->_barrier->Wait();
      for (auto & link : RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id]) {
        for (auto dev_id : link) {
          call_combine(dev_id);
          // IdType offset = group_offset[dev_id];
          // IdType link_num_node = group_offset[dev_id+1] - offset;
          // CombineOneGroup(src_index + offset, dst_index + offset, nodes + offset, link_num_node, cache_ctx->_device_cache_data[dev_id], output, stream);
        }
      }
      _cache_ctx->_barrier->Wait();
      combine_times[1] = t1.Passed();
    }

    t1.Reset();
    call_combine(_cache_ctx->_local_location_id);
    combine_times[2] = t1.Passed();

    // output_src_index_handle = nullptr;
    // output_dst_index_handle = nullptr;
    if (task_key != 0xffffffffffffffff) {
      size_t num_miss = group_offset[_cache_ctx->_cpu_location_id+1]- group_offset[_cache_ctx->_cpu_location_id];
      size_t num_local = group_offset[_cache_ctx->_local_location_id+1] - group_offset[_cache_ctx->_local_location_id];
      size_t num_remote = num_nodes - num_miss - num_local;
      // size_t num_hit = group_offset[1];
      auto _dtype = _cache_ctx->_dtype;
      auto _dim = _cache_ctx->_dim;
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
      _cache_ctx->_coll_cache->_profiler->LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[2]);
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
    // cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
  }
  _cache_ctx->progress.fetch_add(1);
}

#ifdef DEAD_CODE
CollCacheManager CollCacheManager::BuildLegacy(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  TensorPtr cache_node_ptr, size_t num_total_nodes,
                  double cache_percentage, StreamHandle stream) {
  LOG(ERROR) << "Building Legacy Cache...";
  if (cache_percentage == 0 || cache_percentage == 1) {
    return BuildLegacy(trainer_ctx, cpu_src_data, dtype, dim, nullptr, num_total_nodes, cache_percentage, stream);
  }
  IdType* cache_node_list = (IdType*)cache_node_ptr->Data();
  CHECK_NE(cache_node_list, nullptr);
  return BuildLegacy(trainer_ctx, cpu_src_data, dtype, dim, cache_node_list, num_total_nodes, cache_percentage, stream);
}

CollCacheManager CollCacheManager::BuildLegacy(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  const IdType* cache_node_list, size_t num_total_nodes,
                  double cache_percentage, StreamHandle stream) {
  std::cout << "test_result:init:feat_nbytes=" << GetTensorBytes(dtype, {num_total_nodes, dim}) << "\n";
  if (cache_percentage == 0) {
    return BuildNoCache(trainer_ctx, cpu_src_data, dtype, dim, stream);
  } else if (cache_percentage == 1) {
    return BuildFullCache(trainer_ctx, cpu_src_data, dtype, dim, num_total_nodes, stream);
  }
  CHECK_NE(cache_node_list, nullptr);
  CollCacheManager cm(trainer_ctx, dtype, dim, 1);

  Timer t;

  size_t num_cached_nodes = num_total_nodes * cache_percentage;

  cm._cache_nbytes = GetTensorBytes(cm._dtype, {num_cached_nodes, cm._dim});

  // auto cpu_device = Device::Get(_extractor_ctx);
  auto trainer_gpu_device = Device::Get(trainer_ctx);

  // _cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(_extractor_ctx, sizeof(IdType) * _num_nodes));
  cm._device_cache_data.resize(2);
  cm._device_cache_data[0] = trainer_gpu_device->AllocDataSpace(trainer_ctx, cm._cache_nbytes);
  cm._device_cache_data[1] = cpu_src_data;
  cm._hash_table_location = (HashTableEntryLocation*)trainer_gpu_device->AllocDataSpace(trainer_ctx, sizeof(HashTableEntryLocation) * num_total_nodes);
  cm._hash_table_offset = (HashTableEntryOffset*)trainer_gpu_device->AllocDataSpace(trainer_ctx, sizeof(HashTableEntryOffset) * num_total_nodes);

  LOG(INFO) << "CollCacheManager: Initializing hashtable...";

  // 1. Initialize the hashtable with all miss
  auto cu_stream = static_cast<cudaStream_t>(stream);
  // auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  {
    SAM_CUDA_PREPARE_1D(num_total_nodes);
    init_hash_table_cpu<><<<grid, block, 0, cu_stream>>>(cm._hash_table_location, cm._hash_table_offset, num_total_nodes, cm._cpu_location_id);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
  }

  LOG(INFO) << "CollCacheManager: Initializing cache data...";
  // 2. use the hash table to extract cache from cpu
  SrcKey * src_index = nullptr;
  DstVal * dst_index = nullptr;
  {
    LOG(DEBUG) << "CollCacheManager: Initializing cache data - getting miss/hit index...";
    cm.GetMissCacheIndex(src_index, dst_index, cache_node_list, num_cached_nodes, stream);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
    // all location must be cpu now.
    LOG(DEBUG) << "CollCacheManager: Initializing cache data - getting cache data...";
    cm.CombineOneGroup(src_index, dst_index, cache_node_list, num_cached_nodes, cpu_src_data, cm._device_cache_data[0], stream);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
  }

  LOG(INFO) << "CollCacheManager: Add cache entry to hashtable...";
  // 3. modify hash table with cached nodes
  if (num_cached_nodes > 0){
    SAM_CUDA_PREPARE_1D(num_cached_nodes);
    init_hash_table_local<><<<grid, block, 0, cu_stream>>>(cm._hash_table_location, cm._hash_table_offset, cache_node_list, num_cached_nodes, 0);
    trainer_gpu_device->StreamSync(trainer_ctx, stream);
  }

  // 4. Free index
  trainer_gpu_device->FreeWorkspace(trainer_ctx, src_index);
  trainer_gpu_device->FreeWorkspace(trainer_ctx, dst_index);

  LOG(INFO) << "Collaborative GPU cache (policy: " << RunConfig::cache_policy
            << ") " << num_cached_nodes << " / " << num_total_nodes << " nodes ( "
            << ToPercentage(cache_percentage) << " | "
            << ToReadableSize(cm._cache_nbytes) << " | " << t.Passed()
            << " secs )";
  std::cout << "test_result:init:cache_nbytes=" << cm._cache_nbytes << "\n";
  return cm;
}
#endif

void CacheContext::build_no_cache(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_src_data, DataType dtype, size_t dim, Context gpu_ctx, StreamHandle stream) {
  _cpu_location_id = 0;
  _num_location = 1;
  _dtype = dtype;
  _dim = dim;
  _trainer_ctx = gpu_ctx;

  _cache_nbytes = 0;

  _device_cache_data.resize(1);
  _device_cache_data[0] = cpu_src_data;

  _cache_nodes = 0;
  _cache_space_capacity = 0;

  LOG(INFO) << "Collaborative GPU cache (policy: " << "no cache" << ")";
  std::cout << "test_result:init:cache_nbytes=" << _cache_nbytes << "\n";
}

void CacheContext::build_full_cache(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_src_data, DataType dtype, size_t dim, Context gpu_ctx, size_t num_total_nodes, StreamHandle stream) {
  _cpu_location_id = -1;
  _num_location = 1;
  _dtype = dtype;
  _dim = dim;
  _trainer_ctx = gpu_ctx;

  Timer t;

  _cache_nbytes = GetTensorBytes(_dtype, {num_total_nodes, _dim});
  _cache_space_capacity = _cache_nbytes;

  auto trainer_gpu_device = Device::Get(gpu_ctx);

  _device_cache_data_local_handle = _eager_gpu_mem_allocator(_cache_nbytes);
  void* local_cache = _device_cache_data_local_handle->ptr();

  size_t cache_init_nbytes = _cache_nbytes;
  if (RunConfig::option_empty_feat != 0) {
    cache_init_nbytes = GetTensorBytes(_dtype, {(size_t)1 << RunConfig::option_empty_feat, _dim});
  }
  trainer_gpu_device->CopyDataFromTo(cpu_src_data, 0, local_cache, 0, cache_init_nbytes, CPU(), gpu_ctx, stream);
  trainer_gpu_device->StreamSync(gpu_ctx, stream);

  _device_cache_data.resize(1);
  _device_cache_data[0] = local_cache;

  LOG(INFO) << "Collaborative GPU cache (policy: " << "full cache"
            << ") " << num_total_nodes << " nodes ( "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs )";
  std::cout << "test_result:init:cache_nbytes=" << _cache_nbytes << "\n";
}

#ifdef DEAD_CODE
void CollCacheManager::CheckCudaEqual(const void * a, const void* b, const size_t nbytes, StreamHandle stream) {
  CHECK(nbytes % 4 == 0);
  const size_t n_elem = nbytes / 4;
  {
    SAM_CUDA_PREPARE_1D(n_elem);
    check_eq<><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  }

  // {
  //   const size_t num_tiles = 1; // RoundUpDiv((n_elem), Constant::kCudaTileSize);
  //   const dim3 grid(num_tiles);
  //   const dim3 block(4);

  //   check_eq<4, 1000000><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  // }
  CUDA_CALL(cudaStreamSynchronize((cudaStream_t)stream));
}
#endif

#ifdef COLL_HASH_VALID_LEGACY
void CacheContext::build_without_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream) {
  auto hash_table_offset_list = DevicePointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName);
  auto device_cache_data_list = DevicePointerExchanger(_barrier, Constant::kCollCacheDeviceCacheDataPtrShmName);
  _trainer_ctx = gpu_ctx;
  _dtype = dtype;
  _dim = dim;
  // cpu counts as a location
  _num_location = RunConfig::num_device + 1,
  _cpu_location_id = RunConfig::num_device;

  LOG(INFO) << "Building Coll Cache... num gpu device is " << RunConfig::num_device;
  CHECK(location_id == gpu_ctx.device_id);
  _local_location_id = location_id;
  size_t num_total_nodes = coll_cache_ptr->_nid_to_block->Shape()[0];
  size_t num_blocks = coll_cache_ptr->_block_placement->Shape()[0];

  // CHECK(RunConfig::concurrent_link_impl == common::kNoConcurrentLink) << "Not sure old init method support concurrent link";

  Timer t;

  auto gpu_device = Device::Get(gpu_ctx);
  auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));

  _device_cache_data.resize(_num_location, nullptr);
  _device_cache_data[_cpu_location_id] = cpu_data;
  _hash_table_location_handle = _eager_gpu_mem_allocator(sizeof(HashTableEntryLocation) * num_total_nodes);
  _hash_table_offset_handle   = _eager_gpu_mem_allocator(sizeof(HashTableEntryOffset)   * num_total_nodes);

  _hash_table_location = _hash_table_location_handle->ptr<HashTableEntryLocation>();
  _hash_table_offset   = _hash_table_offset_handle->ptr<HashTableEntryOffset>();


  LOG(INFO) << "CollCacheManager: Initializing hashtable location...";

  auto cu_stream = static_cast<cudaStream_t>(stream);
  // 1. Build a mapping from node id to target device
  {
    TensorPtr node_to_block_gpu = Tensor::CopyToExternal(coll_cache_ptr->_nid_to_block, _eager_gpu_mem_allocator, gpu_ctx, stream);   // large
    TensorPtr block_placement_gpu = Tensor::CopyToExternal(coll_cache_ptr->_block_placement, _eager_gpu_mem_allocator, gpu_ctx, stream); // small
    // build a map from placement combinations to source decision
    size_t placement_to_src_nbytes = sizeof(int) * (1 << RunConfig::num_device);
    int * placement_to_src_cpu = (int*) cpu_device->AllocWorkspace(CPU(), placement_to_src_nbytes);
    PreDecideSrc(RunConfig::num_device, _local_location_id, _cpu_location_id, placement_to_src_cpu);
    MemHandle placement_to_src_gpu_handle = _eager_gpu_mem_allocator(placement_to_src_nbytes);
    int * placement_to_src_gpu = placement_to_src_gpu_handle->ptr<int>();
    gpu_device->CopyDataFromTo(placement_to_src_cpu, 0, placement_to_src_gpu, 0, placement_to_src_nbytes, CPU(), gpu_ctx, stream);

    SAM_CUDA_PREPARE_1D(num_total_nodes);
    if (RunConfig::option_empty_feat == 0) {
      decide_source_location<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
        (const IdType*)node_to_block_gpu->Data(), (const uint8_t*)block_placement_gpu->Data(),
        placement_to_src_gpu,
        _hash_table_location, _hash_table_offset, num_total_nodes,
        num_blocks, _local_location_id, _cpu_location_id);
    } else {
      LOG(WARNING) << "using empty feat=" << RunConfig::option_empty_feat;
      decide_source_location<Constant::kCudaBlockSize, Constant::kCudaTileSize, true><<<grid, block, 0, cu_stream>>>(
        (const IdType*)node_to_block_gpu->Data(), (const uint8_t*)block_placement_gpu->Data(),
        placement_to_src_gpu,
        _hash_table_location, _hash_table_offset, num_total_nodes,
        num_blocks, _local_location_id, _cpu_location_id, RunConfig::option_empty_feat);
    }
    gpu_device->StreamSync(_trainer_ctx, stream);
    cpu_device->FreeWorkspace(CPU(), placement_to_src_cpu);
  }
  LOG(INFO) << "CollCacheManager: Initializing hashtable location done";

  /**
   * 2. build list of cached node
   *    prepare local cache
   *    fill hash table for local nodes.
   */
  LOG(INFO) << "CollCacheManager: grouping node of same location";
  // auto loc_list  = (HashTableEntryLocation*)trainer_gpu_device->AllocDataSpace(trainer_ctx, sizeof(HashTableEntryLocation) * num_total_nodes);

  // this buffer only stores node for local and each remote, so no need to be that large
  auto node_list_buffer_handle = _eager_gpu_mem_allocator((size_t)(num_total_nodes * (cache_percentage + 0.001)) * sizeof(IdType));
  IdType* node_list_buffer = node_list_buffer_handle->ptr<IdType>();
  // IdType * group_offset;
  size_t num_cached_nodes;
  size_t num_cpu_nodes;
  {
    IdType* cache_node_list = node_list_buffer;
    // now we want to select nodes with hash_table_location==local id
    cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, cache_node_list, num_cached_nodes, _local_location_id, _eager_gpu_mem_allocator, stream);
    cuda::CubCountByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, num_cpu_nodes, _cpu_location_id, _eager_gpu_mem_allocator, stream);
    // cuda::CubSortPair<int, IdType>(
    //     (const int*)cm._hash_table_location, loc_list, 
    //     (const IdType*)cm._hash_table_offset, node_list,
    //     num_total_nodes, trainer_ctx, 0, sizeof(int)*8, stream);
    // LOG(INFO) << "CollCacheManager: split group";
    // _SplitGroup<>(loc_list, num_total_nodes, group_offset, trainer_ctx, num_device+1, stream);

    // now node_list[ group_offset[local_location_id]...group_offset[local_location_id+1] ] holds cached nodes
    // IdType * cache_node_list = node_list + group_offset[local_location_id];
    // num_cached_nodes = group_offset[local_location_id+1]-group_offset[local_location_id];
    CHECK_NE(num_cached_nodes, 0);
    // num_cpu_nodes = group_offset[cm._cpu_location_id + 1] - group_offset[cm._cpu_location_id];
    _cache_nbytes = GetTensorBytes(_dtype, {num_cached_nodes, _dim});
    _device_cache_data_local_handle = _eager_gpu_mem_allocator(_cache_nbytes);
    _device_cache_data[_local_location_id] = _device_cache_data_local_handle->ptr();
    // LOG(INFO) << "CollCacheManager: combine local data : [" << group_offset[local_location_id] << "," << group_offset[local_location_id+1] << ")";
    if (RunConfig::option_empty_feat == 0) {
      const DataIter<const IdType*> src_data_iter(cache_node_list, cpu_data, dim);
      DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), _device_cache_data[_local_location_id], dim);
      Combine(src_data_iter, dst_data_iter, num_cached_nodes, gpu_ctx, dtype, dim, stream);
    } else {
      const DataIter<MockSrcOffIter> src_data_iter(MockSrcOffIter(cache_node_list), cpu_data, dim);
      // const DataIter<const IdType*> src_data_iter(cache_node_list, cpu_src_data, dim);
      DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), _device_cache_data[_local_location_id], dim);
      Combine(src_data_iter, dst_data_iter, num_cached_nodes, gpu_ctx, dtype, dim, stream);

    }

    LOG(INFO) << "CollCacheManager: fix offset of local nodes in hash table";
    SAM_CUDA_PREPARE_1D(num_cached_nodes);
    init_hash_table_local<><<<grid, block, 0, cu_stream>>>(
        _hash_table_location, _hash_table_offset, 
        cache_node_list, num_cached_nodes, _local_location_id);
    gpu_device->StreamSync(gpu_ctx, stream);
  }

  LOG(INFO) << "CollCacheManager: waiting for remotes, here is " << _local_location_id;

  // wait until all device's hashtable is ready
  hash_table_offset_list.signin(_local_location_id, _hash_table_offset);
  if (num_cached_nodes > 0) {
    device_cache_data_list.signin(_local_location_id, _device_cache_data[_local_location_id]);
  }

  /**
   * 3. get hashtable entry, cache data from remote devices
   */
  for (int i = 0; i < RunConfig::num_device; i++) {
    if (i == _local_location_id) continue;
    // reuse cache node list as buffer
    IdType * remote_node_list = node_list_buffer;
    size_t num_remote_nodes;
    cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, remote_node_list, num_remote_nodes, i, _eager_gpu_mem_allocator, stream);
    if (num_remote_nodes == 0) continue;
    if (!RunConfig::cross_process) {
      auto cuda_err = cudaDeviceEnablePeerAccess(i, 0);
      if (cuda_err != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CALL(cuda_err);
      }
    }
    _device_cache_data[i] = device_cache_data_list.extract(i);
    auto remote_hash_table_offset = (const HashTableEntryOffset * )hash_table_offset_list.extract(i);

    SAM_CUDA_PREPARE_1D(num_remote_nodes);
    init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
        _hash_table_location, _hash_table_offset, 
        remote_hash_table_offset, remote_node_list, num_remote_nodes, i);
    gpu_device->StreamSync(gpu_ctx, stream);
    hash_table_offset_list.close((void*)remote_hash_table_offset);
    // CUDA_CALL(cudaIpcCloseMemHandle((void*)remote_hash_table_offset))
  }

  // 4. Free index
  node_list_buffer_handle = nullptr;
  // trainer_gpu_device->FreeDataSpace(trainer_ctx, loc_list);
  // cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);

  size_t num_remote_nodes = num_total_nodes - num_cached_nodes - num_cpu_nodes;

  LOG(ERROR) << "Collaborative GPU cache (policy: " << RunConfig::cache_policy << ") | "
            << "local "  << num_cached_nodes << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_cached_nodes / (double)num_total_nodes) << "~" << ToPercentage(cache_percentage) << ") | "
            << "remote " << num_remote_nodes << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_remote_nodes / (double)num_total_nodes) << ") | "
            << "cpu "    << num_cpu_nodes    << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_cpu_nodes    / (double)num_total_nodes) << ") | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs ";
  std::cout << "test_result:init:feat_nbytes=" << GetTensorBytes(dtype, {num_total_nodes, dim}) << "\n";
  std::cout << "test_result:init:cache_nbytes=" << _cache_nbytes << "\n";
}
#endif

#ifdef COLL_HASH_VALID_LEGACY
void CacheContext::build_with_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream) {
  // auto hash_table_list = DevicePointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName + "_hashtable");
  auto hash_table_offset_list = DevicePointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName);
  // auto device_cache_data_list = DevicePointerExchanger(_barrier, Constant::kCollCacheDeviceCacheDataPtrShmName);
  auto hash_table_location_list = DevicePointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName + "_location");
  _trainer_ctx = gpu_ctx;
  _dtype = dtype;
  _dim = dim;
  // cpu counts as a location
  _num_location = RunConfig::num_device + 1,
  _cpu_location_id = RunConfig::num_device;
  LOG(INFO) << "Coll cache init with " << RunConfig::num_device << " gpus, " << _num_location << " locations";

  LOG(ERROR) << "Building Coll Cache with ... num gpu device is " << RunConfig::num_device;

  CHECK(location_id == gpu_ctx.device_id);
  _local_location_id = location_id;
  size_t num_total_nodes = coll_cache_ptr-> _nid_to_block->Shape()[0];
  size_t num_blocks = coll_cache_ptr->_block_placement->Shape()[0];

  // RunConfig::coll_cache_link_desc = coll_cache::AsymmLinkDesc::AutoBuild(trainer_ctx);

  Timer t;

  auto gpu_device = Device::Get(gpu_ctx);
  auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));

  // _remote_hash_table.resize(_num_location, nullptr);
  _remote_hash_table_offset.resize(_num_location, nullptr);
  _remote_hash_table_location.resize(_num_location, nullptr);
  // _device_cache_data.resize(_num_location, nullptr);
  // _device_cache_data[_cpu_location_id] = cpu_data;

  auto cpu_hashtable_location_list = HostPointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName + "_cpu");
  auto cpu_hashtable_offset_list = HostPointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName + "_location_cpu");

  if (GetEnv("COLL_CPU_HASHTABLE") != "") {
    _hash_table_location = (HashTableEntryLocation*)cpu_hashtable_location_list.signin(_local_location_id, sizeof(HashTableEntryLocation) * num_total_nodes);
    _hash_table_offset = (HashTableEntryOffset*)cpu_hashtable_offset_list.signin(_local_location_id, sizeof(HashTableEntryOffset) * num_total_nodes);
  } else {
    _hash_table_location_handle = _eager_gpu_mem_allocator(sizeof(HashTableEntryLocation) * num_total_nodes);
    _hash_table_offset_handle   = _eager_gpu_mem_allocator(sizeof(HashTableEntryOffset)   * num_total_nodes);

    _hash_table_location = _hash_table_location_handle->ptr<HashTableEntryLocation>();
    _hash_table_offset   = _hash_table_offset_handle->ptr<HashTableEntryOffset>();
  }


  LOG(INFO) << "CollCacheManager: Initializing hashtable location...";

  auto cu_stream = static_cast<cudaStream_t>(stream);
  // 1. Build a mapping from node id to target device
  {
    TensorPtr node_to_block_gpu = Tensor::CopyToExternal(coll_cache_ptr->_nid_to_block, _eager_gpu_mem_allocator, gpu_ctx, stream);   // large
    TensorPtr block_access_advise_gpu = Tensor::CopyLineToExternel(coll_cache_ptr->_block_access_advise, location_id, _eager_gpu_mem_allocator, gpu_ctx, stream); // small

    SAM_CUDA_PREPARE_1D(num_total_nodes);
    if (RunConfig::option_empty_feat == 0) {
      decide_source_location_advised<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
        node_to_block_gpu->CPtr<IdType>(), block_access_advise_gpu->CPtr<uint8_t>(),
        _hash_table_location, _hash_table_offset, num_total_nodes,
        num_blocks, _local_location_id, _cpu_location_id);
    } else {
      LOG(WARNING) << "using empty feat=" << RunConfig::option_empty_feat;
      decide_source_location_advised<Constant::kCudaBlockSize, Constant::kCudaTileSize, true><<<grid, block, 0, cu_stream>>>(
        node_to_block_gpu->CPtr<IdType>(), block_access_advise_gpu->CPtr<uint8_t>(),
        _hash_table_location, _hash_table_offset, num_total_nodes,
        num_blocks, _local_location_id, _cpu_location_id, RunConfig::option_empty_feat);
    }
    gpu_device->StreamSync(gpu_ctx, stream);
  }
  LOG(INFO) << "CollCacheManager: Initializing hashtable location done";

  /**
   * 2. build list of cached node
   *    prepare local cache
   *    fill hash table for local nodes.
   */
  LOG(INFO) << "CollCacheManager: grouping node of same location";
  // auto loc_list  = (HashTableEntryLocation*)trainer_gpu_device->AllocDataSpace(trainer_ctx, sizeof(HashTableEntryLocation) * num_total_nodes);

  // this buffer only stores node for local and each remote, so no need to be that large
  auto node_list_buffer_handle = _eager_gpu_mem_allocator((size_t)(num_total_nodes * (cache_percentage + 0.001)) * sizeof(IdType));
  IdType* node_list_buffer = (IdType*)node_list_buffer_handle->ptr();
  // IdType * group_offset;
  size_t num_cached_nodes;
  size_t num_cpu_nodes;
  {
    IdType* cache_node_list = node_list_buffer;
    // now we want to select nodes with hash_table_location==local id
    cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, cache_node_list, num_cached_nodes, _local_location_id, _eager_gpu_mem_allocator, stream);
    _local_node_list_tensor = Tensor::Empty(kI32, {num_cached_nodes}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    gpu_device->CopyDataFromTo(cache_node_list, 0, _local_node_list_tensor->Ptr<IdType>(), 0, _local_node_list_tensor->NumBytes(), gpu_ctx, CPU(CPU_CLIB_MALLOC_DEVICE), stream);
    cuda::CubCountByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, num_cpu_nodes, _cpu_location_id, _eager_gpu_mem_allocator, stream);
    // CHECK_NE(num_cached_nodes, 0);
    _cache_space_capacity = (size_t)(num_total_nodes * (cache_percentage + 0.001));
    _cache_nbytes = GetTensorBytes(_dtype, {_cache_space_capacity, _dim});
    _cache_nodes = num_cached_nodes;
    // _device_cache_data_local_handle = _eager_gpu_mem_allocator(_cache_nbytes);
    // _device_cache_data[_local_location_id] = _device_cache_data_local_handle->ptr();
    if (num_cached_nodes > 0) {
      // if (RunConfig::option_empty_feat == 0) {
      //   const DataIter<const IdType*> src_data_iter(cache_node_list, cpu_data, dim);
      //   DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), _device_cache_data[_local_location_id], dim);
      //   Combine(src_data_iter, dst_data_iter, num_cached_nodes, gpu_ctx, dtype, dim, stream);
      // } else {
      //   const DataIter<MockOffIter> src_data_iter(MockOffIter(), cpu_data, dim);
      //   DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), _device_cache_data[_local_location_id], dim);
      //   Combine(src_data_iter, dst_data_iter, num_cached_nodes, gpu_ctx, dtype, dim, stream);
      // }

      LOG(INFO) << "CollCacheManager: fix offset of local nodes in hash table";
      SAM_CUDA_PREPARE_1D(num_cached_nodes);
      init_hash_table_local<><<<grid, block, 0, cu_stream>>>(
          _hash_table_location, _hash_table_offset, 
          cache_node_list, num_cached_nodes, _local_location_id);
      gpu_device->StreamSync(gpu_ctx, stream);
    }
  }
  // TensorPtr keys_for_each_source[9] = {nullptr};
  // {
  //   this->_new_hash_table = new CacheEntryManager;
  //   _new_hash_table->DetectKeysForAllSource(
  //       coll_cache_ptr->_nid_to_block, Tensor::CopyLine(coll_cache_ptr->_block_access_advise, location_id, CPU(), stream), location_id, coll_cache_ptr->_block_density, num_total_nodes, keys_for_each_source);
  //   gpu_device->StreamSync(gpu_ctx, stream);
  //   auto cache_node_list = Tensor::CopyToExternal(keys_for_each_source[location_id], _eager_gpu_mem_allocator, _trainer_ctx, stream);
  //   gpu_device->StreamSync(gpu_ctx, stream);
  //   CHECK_EQ(num_cached_nodes, keys_for_each_source[location_id]->NumItem());
  //   if (keys_for_each_source[_cpu_location_id] == nullptr) {
  //     CHECK_EQ(num_cpu_nodes, 0);
  //   } else {
  //     CHECK_EQ(num_cpu_nodes, keys_for_each_source[_cpu_location_id]->NumItem());
  //   }
  //   _new_hash_table->_hash_table = std::make_shared<SimpleHashTable>(_eager_gpu_mem_allocator, (size_t)(num_total_nodes - num_cpu_nodes), _trainer_ctx, stream);
  //   CheckCudaEqual(cache_node_list->Data(), node_list_buffer, num_cached_nodes * sizeof(IdType), stream);
  //   gpu_device->StreamSync(gpu_ctx, stream);
  //   _new_hash_table->InsertSeqOffWithLoc(cache_node_list, location_id, stream);
  //   gpu_device->StreamSync(gpu_ctx, stream);
  //   _new_hash_table->_hash_table->CountEntries(stream);
  //   gpu_device->StreamSync(gpu_ctx, stream);
  // }

  LOG(INFO) << "CollCacheManager: waiting for remotes, here is " << _local_location_id;

  // wait until all device's hashtable is ready
  if (GetEnv("COLL_CPU_HASHTABLE") != "") {
    // _barrier->Wait();
  } else {
    hash_table_location_list.signin(_local_location_id, _hash_table_location);
    hash_table_offset_list.signin(_local_location_id, _hash_table_offset);
  }
  if (num_cached_nodes > 0) {
    // device_cache_data_list.signin(_local_location_id, _device_cache_data[_local_location_id]);
  }
  // hash_table_list.signin(_local_location_id, _new_hash_table->_hash_table->_o2n_table);

  /**
   * 3. get hashtable entry, cache data from remote devices
   */
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      LOG(ERROR) << "Device " << _local_location_id << " init p2p of link " << dev_id;
      IdType * remote_node_list = node_list_buffer;
      size_t num_remote_nodes;
      cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, remote_node_list, num_remote_nodes, dev_id, _eager_gpu_mem_allocator, stream);
      // if (!RunConfig::cross_process) {
      //   auto cuda_err = cudaDeviceEnablePeerAccess(dev_id, 0);
      //   if (cuda_err != cudaErrorPeerAccessAlreadyEnabled) {
      //     CUDA_CALL(cuda_err);
      //   }
      // }
      // _device_cache_data[dev_id] = device_cache_data_list.extract(dev_id);
      if (GetEnv("COLL_CPU_HASHTABLE") != "") {
        _remote_hash_table_location[dev_id] = (HashTableEntryLocation *)cpu_hashtable_location_list.extract(dev_id);
        _remote_hash_table_offset[dev_id] = (HashTableEntryOffset * )cpu_hashtable_offset_list.extract(dev_id);
      } else {
        _remote_hash_table_location[dev_id] = (HashTableEntryLocation *)hash_table_location_list.extract(dev_id);
        _remote_hash_table_offset[dev_id] = (HashTableEntryOffset * )hash_table_offset_list.extract(dev_id);
      }

      if (num_remote_nodes == 0) continue;
      SAM_CUDA_PREPARE_1D(num_remote_nodes);
      init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
          _hash_table_location, _hash_table_offset, 
          _remote_hash_table_offset[dev_id], remote_node_list, num_remote_nodes, dev_id);
      gpu_device->StreamSync(gpu_ctx, stream);
      // hash_table_offset_list.close((void*)_remote_hash_table_offset[dev_id]);
      // CUDA_CALL(cudaIpcCloseMemHandle((void*)_remote_hash_table_offset[dev_id]))
      // {
      //   CHECK_EQ(num_remote_nodes, keys_for_each_source[dev_id]->NumItem());
      //   _remote_hash_table[dev_id] = (BucketO2N * )hash_table_list.extract(dev_id);
      //   CacheEntryManager remote_cache_manager;
      //   remote_cache_manager._hash_table = std::make_shared<SimpleHashTable>(_remote_hash_table[dev_id], (size_t)(num_total_nodes - num_cpu_nodes), _trainer_ctx, stream);
      //   auto keys = Tensor::CopyToExternal(keys_for_each_source[dev_id], _eager_gpu_mem_allocator, _trainer_ctx, stream);
      //   auto off = Tensor::EmptyExternal(kI32, keys->Shape(), _eager_gpu_mem_allocator, _trainer_ctx, "");
      //   remote_cache_manager.LookupOffset(keys, off, stream);
      //   gpu_device->StreamSync(gpu_ctx, stream);
      //   _new_hash_table->InsertWithLoc(keys, off, dev_id, stream);
      //   gpu_device->StreamSync(gpu_ctx, stream);
      //   CheckCudaEqual(keys->Data(), remote_node_list, num_remote_nodes*sizeof(IdType), stream);
      //   gpu_device->StreamSync(gpu_ctx, stream);

      //   auto tmp = Tensor::EmptyExternal(kI32, keys->Shape(), _eager_gpu_mem_allocator, _trainer_ctx, "");

      //   auto new_loc = tmp;
      //   _new_hash_table->LookupLoc(keys, new_loc, stream);
      //   check_cuda_array(new_loc->CPtr<IdType>(), dev_id, num_remote_nodes, true, stream);
      //   gpu_device->StreamSync(gpu_ctx, stream);

      //   auto orig_off = tmp;
      //   DataIter<const IdType*> src_data_iter(remote_node_list, _hash_table_offset, 1);
      //   DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), orig_off->Ptr<IdType>(), 1);
      //   Combine(src_data_iter, dst_data_iter, num_remote_nodes, gpu_ctx, kI32, 1, stream);
      //   gpu_device->StreamSync(gpu_ctx, stream);
      //   CheckCudaEqual(orig_off->Data(), off->Data(), num_remote_nodes*sizeof(IdType), stream);
      //   gpu_device->StreamSync(gpu_ctx, stream);
      // }
    }
  }
  // _new_hash_table->_hash_table->CountEntries(stream);
  // gpu_device->StreamSync(gpu_ctx, stream);
  // _new_hash_table->_cached_keys = keys_for_each_source[location_id];
  // _new_hash_table->_cache_space_capacity = _cache_space_capacity;
  // _new_hash_table->cpu_location_id = _cpu_location_id;
  // _new_hash_table->num_total_key = num_total_nodes;
  // _new_hash_table->_free_offsets = Tensor::Empty(kI32, {_cache_space_capacity - num_cached_nodes}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  // cpu::ArrangeArray<IdType>(_new_hash_table->_free_offsets->Ptr<IdType>(), _cache_space_capacity - num_cached_nodes, num_cached_nodes);
  // 4. Free index
  node_list_buffer_handle = nullptr;

  size_t num_remote_nodes = num_total_nodes - num_cached_nodes - num_cpu_nodes;

  LOG(ERROR) << "Asymm Coll cache (policy: " << RunConfig::cache_policy << ") | "
            << "local "  << num_cached_nodes << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_cached_nodes / (double)num_total_nodes) << "~" << ToPercentage(cache_percentage) << ") | "
            << "remote " << num_remote_nodes << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_remote_nodes / (double)num_total_nodes) << ") | "
            << "cpu "    << num_cpu_nodes    << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_cpu_nodes    / (double)num_total_nodes) << ") | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs ";
  std::cout << "test_result:init:feat_nbytes=" << GetTensorBytes(dtype, {num_total_nodes, dim}) << "\n";
  std::cout << "test_result:init:cache_nbytes=" << _cache_nbytes << "\n";
}
#endif

class HostWorkspaceHandle : public ExternelGPUMemoryHandler {
 public:
  Context _ctx;
  void* _data;
  size_t _nbytes = 0;
  void* ptr() override { return _data; }
  size_t nbytes() override { return _nbytes; }
  HostWorkspaceHandle(Context ctx, size_t nb) : _ctx(ctx), _nbytes(nb) {
    _data = Device::Get(_ctx)->AllocWorkspace(_ctx, _nbytes);
  } 
  ~HostWorkspaceHandle() { 
    Device::Get(_ctx)->FreeWorkspace(_ctx, _data, _nbytes);
  }
};

TensorPtr ConcatAllRemote(TensorPtr* key_list, int num_gpu, int local_id) {
  size_t total_len = 0;
  for (int i = 0; i < num_gpu; i++) {
    if (i == local_id) continue;
    if (key_list[i] == nullptr) continue;
    total_len += key_list[i]->NumItem();
  }
  TensorPtr rst = Tensor::Empty(kI32, {total_len}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  total_len = 0;
  for (int i = 0; i < num_gpu; i++) {
    if (i == local_id) continue;
    if (key_list[i] == nullptr) continue;
    memcpy(rst->Ptr<IdType>() + total_len, key_list[i]->CPtr<IdType>(), key_list[i]->NumBytes());
    total_len += key_list[i]->NumItem();
  }
  return rst;
}

void CacheContext::build_with_advise_new_hash(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream) {
  auto hash_table_list = DevicePointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName);
  auto device_cache_data_list = DevicePointerExchanger(_barrier, Constant::kCollCacheDeviceCacheDataPtrShmName);
  _trainer_ctx = gpu_ctx;
  _dtype = dtype;
  _dim = dim;
  // cpu counts as a location
  _num_location = RunConfig::num_device + 1,
  _cpu_location_id = RunConfig::num_device;
  LOG(INFO) << "Coll cache init with " << RunConfig::num_device << " gpus, " << _num_location << " locations";

  LOG(ERROR) << "Building Coll Cache with ... num gpu device is " << RunConfig::num_device;

  CHECK(location_id == gpu_ctx.device_id);
  _local_location_id = location_id;
  size_t num_total_nodes = coll_cache_ptr-> _nid_to_block->Shape()[0];
  size_t num_blocks = coll_cache_ptr->_block_placement->Shape()[0];

  Timer t;

  auto gpu_device = Device::Get(gpu_ctx);
  auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));

  _remote_new_hash_table.resize(_num_location, nullptr);
  _remote_hash_table_flat.resize(_num_location, nullptr);
  _remote_hash_table_simple.resize(_num_location, nullptr);
  _device_cache_data.resize(_num_location, nullptr);
  _device_cache_data[_cpu_location_id] = cpu_data;

  LOG(INFO) << "CollCacheManager: Initializing hashtable location...";

  auto cu_stream = static_cast<cudaStream_t>(stream);
  // 1. Build a mapping from node id to target device
  LOG(INFO) << "CollCacheManager: Initializing hashtable location done";

  /**
   * 2. build list of cached node
   *    prepare local cache
   *    fill hash table for local nodes.
   */
  LOG(INFO) << "CollCacheManager: grouping node of same location";

  _new_hash_table = new CacheEntryManager;
  TensorPtr* keys_for_each_source = _new_hash_table->_keys_for_each_src;
  CHECK(coll_cache_ptr->_nid_to_block != nullptr);
  CHECK(coll_cache_ptr->_block_access_advise != nullptr);
  CHECK(coll_cache_ptr->_block_density != nullptr);
  CacheEntryManager::DetectKeysForAllSource(
      coll_cache_ptr->_nid_to_block, Tensor::CopyLine(coll_cache_ptr->_block_access_advise, location_id, CPU(), stream), location_id, coll_cache_ptr->_block_density, num_total_nodes, keys_for_each_source, RunConfig::num_device);
  gpu_device->StreamSync(gpu_ctx, stream);
  LOG(INFO) << "CollCacheManager: grouping node of same location done";
  size_t num_cached_nodes = keys_for_each_source[location_id]->NumItem();
  _new_hash_table->_remote_keys = ConcatAllRemote(keys_for_each_source, RunConfig::num_device, location_id);
  size_t num_cpu_nodes = num_total_nodes - _new_hash_table->_remote_keys->NumItem() - num_cached_nodes;
  LOG(ERROR) << "num cpu nodes is " << num_cpu_nodes;
  CUDA_CALL(cudaGetLastError());

  {
    // CHECK_NE(num_cached_nodes, 0);
    _cache_space_capacity = (size_t)(num_total_nodes * (cache_percentage + 0.001));
    _cache_nbytes = GetTensorBytes(_dtype, {_cache_space_capacity, _dim});
    _cache_nodes = num_cached_nodes;
    _device_cache_data_local_handle = _eager_gpu_mem_allocator(_cache_nbytes);
    _device_cache_data[_local_location_id] = _device_cache_data_local_handle->ptr();
    auto cache_node_list = Tensor::CopyToExternal(keys_for_each_source[location_id], _eager_gpu_mem_allocator, _trainer_ctx, stream);
    if (num_cached_nodes > 0) {
      SWITCH_BOOL(RunConfig::option_empty_feat != 0, use_empty_feat, {
        IdxStoreDirect<use_empty_feat> idx_store(cache_node_list->CPtr<IdType>());
        DataIterPerLoc<decltype(idx_store)> data_iter(idx_store, cpu_data, _device_cache_data[_local_location_id], dim);
        Combine(data_iter, num_cached_nodes, gpu_ctx, dtype, dim, stream);
      });
      CUDA_CALL(cudaGetLastError());

      LOG(INFO) << "CollCacheManager: fix offset of local nodes in hash table";
      if (RunConfig::coll_skip_hash == false) {
        if (TableSize(num_total_nodes - num_cpu_nodes) > num_total_nodes / 2) {
          RunConfig::use_flat_hashtable = true;
          LOG(ERROR) << "simple hashtable too large, use flat hashtable";
          _new_hash_table->_flat_hash_table = std::make_shared<FlatHashTable>(_eager_gpu_mem_allocator, num_total_nodes, _trainer_ctx, stream);
        } else {
          RunConfig::use_flat_hashtable = false;
          LOG(ERROR) << "simple hashtable small enough, use simple hashtable";
          _new_hash_table->_simple_hash_table = std::make_shared<SimpleHashTable>(_eager_gpu_mem_allocator, (size_t)(num_total_nodes - num_cpu_nodes), _trainer_ctx, stream);
        }
        // if (TableSize(num_total_nodes - num_cpu_nodes) > num_total_nodes / 2) {
        //   LOG(ERROR) << "simple hashtable too large, use flat hashtable";
        //   _new_hash_table->_flat_hash_table = std::make_shared<FlatHashTable>(_eager_gpu_mem_allocator, num_total_nodes, _trainer_ctx, stream);
        //   use_flat_hashtable = true;
        // } else {
        //   _new_hash_table->_simple_hash_table = std::make_shared<SimpleHashTable>(_eager_gpu_mem_allocator, (size_t)(num_total_nodes - num_cpu_nodes), _trainer_ctx, stream);
        //   use_flat_hashtable = false;
        // }
        _new_hash_table->InsertSeqOffWithLoc(cache_node_list, location_id, stream);
        gpu_device->StreamSync(gpu_ctx, stream);
      }
      // _new_hash_table->_hash_table->CountEntries(stream);
      // gpu_device->StreamSync(gpu_ctx, stream);
    }
  }

  LOG(INFO) << "CollCacheManager: waiting for remotes, here is " << _local_location_id;

  // wait until all device's hashtable is ready
  if (RunConfig::coll_skip_hash == false) {
    if (RunConfig::use_flat_hashtable) {
      hash_table_list.signin(_local_location_id, _new_hash_table->_flat_hash_table->_flat_table);
    } else {
      hash_table_list.signin(_local_location_id, _new_hash_table->_simple_hash_table->_o2n_table);
    }
  }
  if (num_cached_nodes > 0) {
    device_cache_data_list.signin(_local_location_id, _device_cache_data[_local_location_id]);
  }

  /**
   * 3. get hashtable entry, cache data from remote devices
   */
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      _barrier->Wait();
      if (!RunConfig::cross_process) {
        auto cuda_err = cudaDeviceEnablePeerAccess(dev_id, 0);
        if (cuda_err != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CALL(cuda_err);
        }
      }
      _device_cache_data[dev_id] = device_cache_data_list.extract(dev_id);
      if (RunConfig::coll_skip_hash) {
        continue;
      }
      _remote_new_hash_table[dev_id] = new CacheEntryManager;
      CacheEntryManager &remote_cache_manager = *_remote_new_hash_table[dev_id];
      if (RunConfig::use_flat_hashtable) {
        _remote_hash_table_flat[dev_id] = (BucketFlat * )hash_table_list.extract(dev_id);
        remote_cache_manager._flat_hash_table = std::make_shared<FlatHashTable>(_remote_hash_table_flat[dev_id], num_total_nodes, _trainer_ctx, stream);
      } else {
        _remote_hash_table_simple[dev_id] = (BucketO2N * )hash_table_list.extract(dev_id);
        remote_cache_manager._simple_hash_table = std::make_shared<SimpleHashTable>(_remote_hash_table_simple[dev_id], (size_t)(num_total_nodes - num_cpu_nodes), _trainer_ctx, stream);
      }
      if (keys_for_each_source[dev_id] == nullptr || keys_for_each_source[dev_id]->NumItem() == 0 || dev_id == _local_location_id) continue;
      auto keys = Tensor::CopyToExternal(keys_for_each_source[dev_id], _eager_gpu_mem_allocator, _trainer_ctx, stream);
      auto off = Tensor::EmptyExternal(kI32, keys->Shape(), _eager_gpu_mem_allocator, _trainer_ctx, "");
      remote_cache_manager.LookupOffset(keys, off, stream);
      auto off_cpu = off->CopyTo(CPU(), stream, "");
      if (_local_location_id == 0) {
        LOG(ERROR) 
          << keys_for_each_source[dev_id]->CPtr<IdType>()[0] << "," << off_cpu->CPtr<IdType>()[0] << "|"
          << keys_for_each_source[dev_id]->CPtr<IdType>()[1] << "," << off_cpu->CPtr<IdType>()[1] << "|"
          << keys_for_each_source[dev_id]->CPtr<IdType>()[2] << "," << off_cpu->CPtr<IdType>()[2] << "|"
          << keys_for_each_source[dev_id]->CPtr<IdType>()[3] << "," << off_cpu->CPtr<IdType>()[3] << "|"
          << keys_for_each_source[dev_id]->CPtr<IdType>()[4] << "," << off_cpu->CPtr<IdType>()[4] << "|"
        ;
      }
      _new_hash_table->InsertWithLoc(keys, off, dev_id, stream);
      gpu_device->StreamSync(gpu_ctx, stream);
    }
  }
  if (RunConfig::coll_skip_hash) {
    auto clique_size = RunConfig::coll_cache_link_desc.CliqueSize();
    auto clique_master = RoundUp(_local_location_id + 1, clique_size) - clique_size;
    _device_cache_data_clique = std::vector<void*>(_device_cache_data.begin() + clique_master, _device_cache_data.begin() + clique_master + clique_size);
  }
  if (RunConfig::coll_skip_hash == false) {
    _new_hash_table->_cached_keys = keys_for_each_source[location_id];
    _new_hash_table->_cache_space_capacity = _cache_space_capacity;
    _new_hash_table->cpu_location_id = _cpu_location_id;
    _new_hash_table->num_total_key = num_total_nodes;
    _new_hash_table->_free_offsets = Tensor::EmptyExternal(kI32, {_cache_space_capacity}, [](size_t nb){
      return std::make_shared<HostWorkspaceHandle>(CPU(CPU_CLIB_MALLOC_DEVICE), nb);
    }, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    _new_hash_table->_free_offsets->ForceScale(kI32, {_cache_space_capacity - num_cached_nodes}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    cpu::ArrangeArray<IdType>(_new_hash_table->_free_offsets->Ptr<IdType>(), _cache_space_capacity - num_cached_nodes, num_cached_nodes);
  }
  CUDA_CALL(cudaGetLastError());
  // 4. Free index 
  size_t num_remote_nodes = num_total_nodes - num_cached_nodes - num_cpu_nodes;

  LOG(ERROR) << "Asymm Coll cache (policy: " << RunConfig::cache_policy << ") | "
            << "local "  << num_cached_nodes << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_cached_nodes / (double)num_total_nodes) << "~" << ToPercentage(cache_percentage) << ") | "
            << "remote " << num_remote_nodes << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_remote_nodes / (double)num_total_nodes) << ") | "
            << "cpu "    << num_cpu_nodes    << " / " << num_total_nodes << " nodes ( "<< ToPercentage(num_cpu_nodes    / (double)num_total_nodes) << ") | "
            << ToReadableSize(_cache_nbytes) << " | " << t.Passed()
            << " secs ";
  std::cout << "test_result:init:feat_nbytes=" << GetTensorBytes(dtype, {num_total_nodes, dim}) << "\n";
  std::cout << "test_result:init:cache_nbytes=" << _cache_nbytes << "\n";
}

void CacheContext::build(std::function<MemHandle(size_t)> gpu_mem_allocator,
                         int location_id,
                         std::shared_ptr<CollCache> coll_cache_ptr,
                         void *cpu_data, DataType dtype, size_t dim,
                         Context gpu_ctx, double cache_percentage,
                         StreamHandle stream) {
  _coll_cache = coll_cache_ptr;
  _gpu_mem_allocator = gpu_mem_allocator;

  d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  _eager_gpu_mem_allocator = [gpu_ctx](size_t nbytes)-> MemHandle{
    std::shared_ptr<EagerGPUMemoryHandler> ret = std::make_shared<EagerGPUMemoryHandler>();
    ret->dev_id_ = gpu_ctx.device_id;
    ret->nbytes_ = nbytes;
    CUDA_CALL(cudaSetDevice(gpu_ctx.device_id));
    if (nbytes > 1<<21) {
      nbytes = RoundUp(nbytes, (size_t)(1<<21));
    }
    CUDA_CALL(cudaMalloc(&ret->ptr_, nbytes));
    return ret;
  };
  // _gpu_mem_allocator = _eager_gpu_mem_allocator;
  // _eager_gpu_mem_allocator = _gpu_mem_allocator;

  if (cache_percentage == 0) {
    return build_no_cache(location_id, coll_cache_ptr, cpu_data, dtype, dim, gpu_ctx, stream);
  } else if (cache_percentage == 1 && RunConfig::cache_policy != common::kCliquePart) {
    return build_full_cache(location_id, coll_cache_ptr, cpu_data, dtype, dim, gpu_ctx, RunConfig::num_total_item, stream);
  } else if (_coll_cache->_block_access_advise) {
    build_with_advise_new_hash(location_id, coll_cache_ptr, cpu_data, dtype, dim,
                             gpu_ctx, cache_percentage, stream);
#ifdef COLL_HASH_VALID_LEGACY
    build_with_advise(location_id, coll_cache_ptr, cpu_data, dtype, dim,
                             gpu_ctx, cache_percentage, stream);
    LOG(ERROR) << "comparing hashtable after construction";
    compare_hashtable(stream);
    LOG(ERROR) << "comparing hashtable after construction - passed";
#endif
  } else {
#ifdef COLL_HASH_VALID_LEGACY
    return build_without_advise(location_id, coll_cache_ptr, cpu_data, dtype, dim,
                                gpu_ctx, cache_percentage, stream);
#endif
    LOG(FATAL) << "Unimplemented";
  }
}
DevicePointerExchanger::DevicePointerExchanger(BarHandle barrier,
                                               std::string shm_name) {
  int fd = cpu::MmapCPUDevice::CreateShm(4096, shm_name);
  _buffer = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), 4096, fd);
  // _barrier = new ((char *)_buffer + 2048) AtomicBarrier(world_size, is_master);
  _barrier = barrier;
  // even for single process, we require concurrent initialization of each gpu.
  // _cross_process = cross_process;
}

void DevicePointerExchanger::signin(int local_id, void *ptr_to_share) {
  if (RunConfig::cross_process) {
    CUDA_CALL(cudaIpcGetMemHandle(&((cudaIpcMemHandle_t *)_buffer)[local_id], ptr_to_share));
  } else {
    static_cast<void **>(_buffer)[local_id] = ptr_to_share;
  }
  _barrier->Wait();
}
void *DevicePointerExchanger::extract(int location_id) {
  void *ret = nullptr;
  if (RunConfig::cross_process) {
    CUDA_CALL(cudaIpcOpenMemHandle(&ret, ((cudaIpcMemHandle_t *)_buffer)[location_id],
                                   cudaIpcMemLazyEnablePeerAccess));
  } else {
    ret = static_cast<void **>(_buffer)[location_id];
  }
  return ret;
}
void DevicePointerExchanger::close(void *ptr) {
  if (RunConfig::cross_process) {
    CUDA_CALL(cudaIpcCloseMemHandle(ptr));
  }
}

HostPointerExchanger::HostPointerExchanger(BarHandle barrier,
                                               std::string shm_name) {
  // int fd = cpu::MmapCPUDevice::CreateShm(4096, shm_name);
  // _buffer = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), 4096, fd);
  // _barrier = new ((char *)_buffer + 2048) AtomicBarrier(world_size, is_master);
  CHECK(RunConfig::cross_process);
  _barrier = barrier;
  _shm_name = shm_name;
  // even for single process, we require concurrent initialization of each gpu.
  // _cross_process = cross_process;
}

void* HostPointerExchanger::signin(int local_id, size_t nbytes) {
  // register a shared area for local
  nbytes = RoundUp<size_t>(nbytes, 1<<21);
  int local_area_fd = cpu::MmapCPUDevice::CreateShm(nbytes, _shm_name + "_" + std::to_string(local_id));
  void* local_area = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, local_area_fd);
  CUDA_CALL(cudaHostRegister(local_area, nbytes, cudaHostRegisterDefault));
  _barrier->Wait();
  return local_area;
}
void *HostPointerExchanger::extract(int location_id) {
  size_t nbytes = 0;
  int remote_area_fd = cpu::MmapCPUDevice::OpenShm(_shm_name + "_" + std::to_string(location_id), &nbytes);
  void* remote_area = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, remote_area_fd);
  CUDA_CALL(cudaHostRegister(remote_area, nbytes, cudaHostRegisterDefault));
  return remote_area;
}
void HostPointerExchanger::close(void *ptr) {
  CUDA_CALL(cudaHostUnregister(ptr));
}

ExtractSession::ExtractSession(std::shared_ptr<CacheContext> cache_ctx) : _cache_ctx(cache_ctx) {
  if (cache_ctx->IsDirectMapping()) return;
  auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  _group_offset = (IdType*)Device::Get(cpu_ctx)->AllocWorkspace(cpu_ctx, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  _local_location_id = _cache_ctx->_local_location_id;
  _cpu_location_id = _cache_ctx->_cpu_location_id;
  auto log_mem_usage = [](int dev_id, std::string msg){
    if (dev_id == 0) {
      size_t free = 0, total = 0;
      cudaMemGetInfo(&free, &total);
      LOG(WARNING) << msg << ToReadableSize(total - free);
    }
  };
  this->_extract_threads.resize(_cache_ctx->_num_location);
  this->_extract_ctx.resize(_cache_ctx->_num_location);
  auto gpu_ctx = cache_ctx->_trainer_ctx;
  auto & link_desc = RunConfig::coll_cache_link_desc;
  auto & link_src = link_desc.link_src[cache_ctx->_local_location_id];
  int num_link = link_src.size();
  // _cache_ctx->ctx_injector_ = [](){};
  if (RunConfig::concurrent_link_impl == common::kMPS) {

    // check_primary_ctx_active(gpu_ctx.device_id);
    cuda::check_have_affinity_support(gpu_ctx.device_id);
    _remote_syncer = ParallelJobSync::create<SpinJobSync>(num_link);
    _cpu_syncer   = new SpinJobSync;
    _local_syncer = new SpinJobSync;
    for (int link_id = 0; link_id < num_link; link_id++) {
      CHECK(link_src[link_id].size() == 1);
      int src_loc = link_src[link_id][0];
      int num_sm = RunConfig::coll_cache_link_desc.link_sm[_local_location_id][link_id];
      this->launch_thread(link_src[link_id][0], _remote_syncer->get_worker_handle(link_id), [this, num_sm](ExtractionThreadCtx* ctx){
        ctx->create_ctx(_local_location_id, num_sm);
      });
    }
    this->launch_thread(_cpu_location_id, _cpu_syncer->get_worker_handle(), [this](ExtractionThreadCtx* ctx){
      ctx->create_ctx(_local_location_id, RunConfig::coll_cache_link_desc.cpu_sm[_local_location_id]);
    });
    this->launch_thread(_local_location_id, _local_syncer->get_worker_handle(), [this](ExtractionThreadCtx* ctx){
      ctx->create_ctx(_local_location_id, RunConfig::coll_cache_link_desc.local_sm[_local_location_id]);
    });
    this->LaunchWaitAllSyncer();
  } else if (RunConfig::concurrent_link_impl == kMultiKernelNumBlock || RunConfig::concurrent_link_impl == kMultiKernelNumBlockOld) {

    // _remote_syncer = new SemBarJobSync(num_link);
    _remote_syncer = ParallelJobSync::create<SemJobSync>(num_link);
    // _remote_syncer = ParallelJobSync::create<SpinJobSync>(num_link);
    _cpu_syncer   = new SpinJobSync;
    _local_syncer = new SpinJobSync;

    for (int link_id = 0; link_id < num_link; link_id++) {
      CHECK(link_src[link_id].size() == 1);
      int src_loc = link_src[link_id][0];
      this->launch_thread(src_loc, _remote_syncer->get_worker_handle(link_id), [this](ExtractionThreadCtx* ctx){
        ctx->create_stream(_local_location_id);
      });
    }
    this->launch_thread(_cpu_location_id, _cpu_syncer->get_worker_handle(), [this](ExtractionThreadCtx* ctx){
      ctx->create_stream(_local_location_id);
    });
    this->launch_thread(_local_location_id, _local_syncer->get_worker_handle(), [this](ExtractionThreadCtx* ctx){
      ctx->create_stream(_local_location_id);
    });
    this->LaunchWaitAllSyncer();
  } else if (RunConfig::concurrent_link_impl == kMPSForLandC) {
    _remote_syncer = ParallelJobSync::create<SpinJobSync>(num_link);
    _cpu_syncer = new SpinJobSync;
    _local_syncer = new SpinJobSync;



    for (int link_id = 0; link_id < num_link; link_id++) {
      CHECK(link_src[link_id].size() == 1);
      int src_loc = link_src[link_id][0];
      this->launch_thread(src_loc, _remote_syncer->get_worker_handle(link_id), [this](ExtractionThreadCtx* ctx){
        ctx->create_stream(_local_location_id);
      });
    }
    this->launch_thread(_cpu_location_id, _cpu_syncer->get_worker_handle(), [this](ExtractionThreadCtx* ctx){
      ctx->create_ctx(_local_location_id, RunConfig::coll_cache_link_desc.cpu_sm[_local_location_id]);
    });
    this->launch_thread(_local_location_id, _local_syncer->get_worker_handle(), [this](ExtractionThreadCtx* ctx){
      ctx->create_ctx(_local_location_id, RunConfig::coll_cache_link_desc.local_sm[_local_location_id]);
    });
    this->LaunchWaitAllSyncer();
  } else if (RunConfig::concurrent_link_impl == kMPSPhase) {
    cuda::check_have_affinity_support(gpu_ctx.device_id);

    _remote_syncer = ParallelJobSync::create<SemJobSync>(num_link);
    // _remote_syncer = new BarJobSync(num_link);
    _cpu_syncer = new SpinJobSync;


    for (int link_id = 0; link_id < num_link; link_id++) {
      CHECK(link_src[link_id].size() == 1);
      int src_loc = link_src[link_id][0];
      int num_sm = RunConfig::coll_cache_link_desc.link_sm[_local_location_id][link_id];
      this->launch_thread(src_loc, _remote_syncer->get_worker_handle(link_id), [this, num_sm](ExtractionThreadCtx* ctx){
        ctx->create_ctx(_local_location_id, num_sm, -5);
      });
    }
    this->launch_thread(_cpu_location_id, _cpu_syncer->get_worker_handle(), [this, link_desc](ExtractionThreadCtx* ctx){
      ctx->create_ctx(_local_location_id, link_desc.cpu_sm[_local_location_id], -5);
    });
    this->LaunchWaitAllSyncer();
    CUDA_CALL(cudaStreamCreate(&_local_ext_stream));
  } else if (RunConfig::concurrent_link_impl == kSMMaskPhase) {
    _remote_syncer = ParallelJobSync::create<SemJobSync>(num_link);
    // _remote_syncer = new BarJobSync(num_link);
    _cpu_syncer = new SpinJobSync;
    auto print_binary = [this](uint64_t mask){
      if (_local_location_id != 0) return;
      std::bitset<64> bs(mask);
      std::cerr << " mask is " << bs << "\n";
    };
    auto sm_id_range_to_mask = [](int small_include, int large_exclude) -> uint64_t {
      CHECK(small_include % 2 == 0);
      CHECK(large_exclude % 2 == 0);
      // note that zero means use that core
      return ~((1ull << (large_exclude/2)) - (1ull << (small_include/2)));
    };
    int current_used_sm = 0;
    auto mask_next_sms = [&current_used_sm, &sm_id_range_to_mask, &print_binary](int num_sm) -> uint64_t {
      uint64_t ret = sm_id_range_to_mask(current_used_sm, current_used_sm + num_sm);
      current_used_sm += num_sm;
      print_binary(ret);
      return ret;
    };
    {
      uint64_t cpu_sm_mask = mask_next_sms(link_desc.cpu_sm[_local_location_id]);
      this->launch_thread(_cpu_location_id, _cpu_syncer->get_worker_handle(), [this, link_desc, cpu_sm_mask](ExtractionThreadCtx* ctx){
        ctx->create_stream_sm_mask_v1(_local_location_id, cpu_sm_mask, -5);
      });
    }
    for (int link_id = 0; link_id < num_link; link_id++) {
      CHECK(link_src[link_id].size() == 1);
      int src_loc = link_src[link_id][0];
      int num_sm = RunConfig::coll_cache_link_desc.link_sm[_local_location_id][link_id];
      uint64_t mask = mask_next_sms(num_sm);
      this->launch_thread(src_loc, _remote_syncer->get_worker_handle(link_id), [this, mask](ExtractionThreadCtx* ctx){
        ctx->create_stream_sm_mask_v1(_local_location_id, mask, -5);
      });
    }
    this->LaunchWaitAllSyncer();
    CUDA_CALL(cudaStreamCreate(&_local_ext_stream));
  } else if (RunConfig::concurrent_link_impl != common::kNoConcurrentLink) {
    _concurrent_stream_array.resize(RunConfig::num_device - 1);
    for (auto & stream : _concurrent_stream_array) {
      cudaStream_t & cu_s = reinterpret_cast<cudaStream_t &>(stream);
      CUDA_CALL(cudaStreamCreate(&cu_s));
    }
  }
}

void ExtractionThreadCtx::v2_thread_func() {
  if (this->cu_ctx_ != nullptr) {
    CU_CALL(cuCtxSetCurrent(this->cu_ctx_));
  }
  while (true) {
    syncer_->on_wait_next_job();
    func_(stream_);
    syncer_->on_job_done();
  }
}
void ExtractionThreadCtx::v2_forward_one_step(std::function<void(cudaStream_t)> new_func) {
  func_ = new_func;
  syncer_->on_send_job();
}
void ExtractionThreadCtx::v2_forward_nop() {
  syncer_->on_send_job();
}
std::thread ExtractionThreadCtx::v2_launch() {
  return std::thread([this](){
    this->v2_thread_func();
  });
}
ExtractionThreadCtx::ExtractionThreadCtx() {}

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void preserve_offset(
    IdType* offset_output,
    const IdType* nodes, const size_t num_nodes,
    const HashTableEntryOffset* hash_table_offset) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t dst_idx = block_start + threadIdx.x; dst_idx < block_end;
       dst_idx += BLOCK_SIZE) {
    if (dst_idx < num_nodes) {
      const IdType node_id = nodes[dst_idx];
      offset_output[dst_idx] = hash_table_offset[node_id];
    }
  }
}


template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename T_MARK>
__global__ void mark_offset_inuse(
    const IdType* offset_inuse_list, const size_t num_input,
    T_MARK* inuse_mark) {

  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += BLOCK_SIZE) {
    if (idx < num_input) {
      IdType offset = offset_inuse_list[idx];
      inuse_mark[offset] = 1;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE, bool use_empty_feat = false>
__global__ void mark_evict_nodes(
    const IdType* evict_node_list, const size_t num_input,
    const IdType fall_back_location_id,
    const IdType old_location_id,
    HashTableEntryLocation* _location, HashTableEntryOffset* _offset,
    size_t empty_feat = 0) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += BLOCK_SIZE) {
    if (idx < num_input) {
      const IdType node_id = evict_node_list[idx];
      assert(_location[node_id] == old_location_id);
      _location[node_id] = fall_back_location_id;
      if (use_empty_feat) {
        _offset[node_id] = node_id % (1 << empty_feat);
      } else {
        _offset[node_id] = node_id;
      }
    }
  }
}

struct _LocationPack {
  const HashTableEntryLocation* data[9];
};

template <size_t BLOCK_SIZE, size_t TILE_SIZE, bool use_empty_feat = false>
__global__ void mark_remote_evict_nodes(
    _LocationPack _remote_location,
    HashTableEntryLocation* local_location_table,
    HashTableEntryOffset* local_offset_table,
    const HashTableEntryLocation local_loc, const HashTableEntryLocation cpu_loc,
    const size_t num_input,
    size_t empty_feat = 0) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t node_id = block_start + threadIdx.x; node_id < block_end; node_id += BLOCK_SIZE) {
    if (node_id < num_input) {
      auto location_id = local_location_table[node_id];
      if (location_id != local_loc && location_id != cpu_loc && _remote_location.data[location_id][node_id] != location_id) {
        local_location_table[node_id] = cpu_loc;
        if (use_empty_feat) {
          local_offset_table[node_id] = node_id % (1 << empty_feat);
        } else {
          local_offset_table[node_id] = node_id;
        }
      }
    }
  }
}


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize>
__global__ void refresh_hash_table_local(
    HashTableEntryLocation* hash_table_location, HashTableEntryOffset* hash_table_offset, 
    const IdType* local_nodes, const size_t num_node,
    const IdType* local_nodes_corresponding_offset,
    const int local_location_id) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += BLOCK_SIZE) {
    if (idx < num_node) {
      IdType node_id = local_nodes[idx];
      hash_table_location[node_id] = local_location_id;
      hash_table_offset[node_id] = local_nodes_corresponding_offset[idx];
    }
  }
}

#ifdef DEAD_CODE
template<typename T>
void check_uniq(const T* array, size_t num) {
  std::set<T> uniq_set;
  for (size_t i = 0; i < num; i++) {
    uniq_set.insert(array[i]);
  }
  CHECK(uniq_set.size() == num);
}

template<typename T>
void check_no_intersec(const T* array1, size_t num1, const T* array2, size_t num2) {
  std::set<T> uniq_set;
  for (size_t i = 0; i < num1; i++) {
    uniq_set.insert(array1[i]);
  }
  CHECK(uniq_set.size() == num1);
  for (size_t i = 0; i < num2; i++) {
    uniq_set.insert(array2[i]);
  }
  CHECK(uniq_set.size() == num1 + num2);
}

template<typename T>
void check_covers(const T* large, size_t num_large, const T* small, size_t num_small) {
  std::set<T> uniq_set;
  for (size_t i = 0; i < num_large; i++) {
    uniq_set.insert(large[i]);
  }
  CHECK(uniq_set.size() == num_large);
  for (size_t i = 0; i < num_small; i++) {
    uniq_set.insert(small[i]);
  }
  CHECK(uniq_set.size() == num_large);
}
#endif

};
void CheckCpuEqual(const void * a_in, const void* b_in, const size_t nbytes) {
  CHECK(nbytes % 4 == 0);
  const size_t n_elem = nbytes / 4;
  auto a = (const uint32_t*)a_in;
  auto b = (const uint32_t*)b_in;
  // {
  //   SAM_CUDA_PREPARE_1D(n_elem);
  //   check_eq<><<<grid, block, 0, (cudaStream_t)stream>>>((const uint32_t*)a, (const uint32_t*)b, n_elem);
  // }

  for (size_t offset = 0; offset < n_elem; offset += 1) {
    CHECK_EQ(a[offset], b[offset]) << " at offset " << offset << ", " << a[offset] << "!=" << b[offset] << "\n";
  }
}
#ifdef COLL_HASH_VALID_LEGACY
void CacheContext::compare_hashtable(StreamHandle stream) {
  IdType validate_batch_size = 1 << 19;
  auto keys = Tensor::EmptyExternal(kI32, {validate_batch_size}, this->_eager_gpu_mem_allocator, this->_trainer_ctx, "");
  auto old_val = Tensor::EmptyExternal(kI32, {validate_batch_size}, this->_eager_gpu_mem_allocator, this->_trainer_ctx, "");
  auto new_val = Tensor::EmptyExternal(kI32, {validate_batch_size}, this->_eager_gpu_mem_allocator, this->_trainer_ctx, "");
  auto cu_stream = reinterpret_cast<cudaStream_t>(stream);
  for (IdType i = 0; i < RoundUp<size_t>(RunConfig::num_total_item, validate_batch_size); i += validate_batch_size) {
    IdType this_batch_len = (i + validate_batch_size > RunConfig::num_total_item) ? (RunConfig::num_total_item - i) : (validate_batch_size);
    keys->ForceScale(kI32, {this_batch_len}, _trainer_ctx, "");
    cuda::ArrangeArray<IdType>(keys->Ptr<IdType>(), this_batch_len, i, 1, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    {
      const DataIter<const IdType*> src_data_iter(keys->CPtr<IdType>(), this->_hash_table_offset, 1);
      DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), old_val->Ptr<IdType>(), 1);
      Combine(src_data_iter, dst_data_iter, this_batch_len, _trainer_ctx, kI32, 1, stream);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
      _new_hash_table->LookupOffset(keys, new_val, stream);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
      // auto old_val_cpu = old_val->CopyTo(CPU(), stream, "");
      // auto new_val_cpu = new_val->CopyTo(CPU(), stream, "");
      // CUDA_CALL(cudaStreamSynchronize(cu_stream));
      // CheckCpuEqual(old_val_cpu->Data(), new_val_cpu->Data(), old_val_cpu->NumBytes());
      CheckCudaEqual(old_val->Data(), new_val->Data(), old_val->NumBytes(), stream);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
    }
    {
      const DataIter<const IdType*> src_data_iter(keys->CPtr<IdType>(), this->_hash_table_location, 1);
      DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), old_val->Ptr<IdType>(), 1);
      Combine(src_data_iter, dst_data_iter, this_batch_len, _trainer_ctx, kI32, 1, stream);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
      _new_hash_table->LookupLoc(keys, new_val, stream);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
      // auto old_val_cpu = old_val->CopyTo(CPU(), stream, "");
      // auto new_val_cpu = new_val->CopyTo(CPU(), stream, "");
      // CUDA_CALL(cudaStreamSynchronize(cu_stream));
      // CheckCpuEqual(old_val_cpu->Data(), new_val_cpu->Data(), old_val_cpu->NumBytes());
      CheckCudaEqual(old_val->Data(), new_val->Data(), old_val->NumBytes(), stream);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
    }
  }
}
#endif

struct SelectEvictedRemoteKeys {
  TensorPtr old_nid_to_block;
  TensorPtr old_block_access;
  coll_cache::TensorView<uint8_t> old_block_access_view;

  TensorPtr new_nid_to_block;
  TensorPtr new_block_placement = nullptr;
  TensorPtr new_block_access;
  coll_cache::TensorView<uint8_t> new_block_access_view;
  int local_loc, cpu_loc;
  SelectEvictedRemoteKeys(
      TensorPtr old_nid_to_block, TensorPtr old_block_access,
      TensorPtr new_nid_to_block, TensorPtr new_block_access, TensorPtr new_block_placement,
      int local_loc, int cpu_loc) : 
      old_nid_to_block(old_nid_to_block), old_block_access(old_block_access),
      new_nid_to_block(new_nid_to_block), new_block_access(new_block_access), // new_block_placement(new_block_placement), 
      local_loc(local_loc), cpu_loc(cpu_loc) {
    CHECK_NOTNULL(old_nid_to_block);
    CHECK_NOTNULL(old_block_access);
    CHECK_NOTNULL(new_nid_to_block);
    CHECK_NOTNULL(new_block_access);
    // LOG(ERROR) << old_nid_to_block->CPtr<IdType>()[0] << old_block_access->CPtr<uint8_t>()[0];
    old_block_access_view = coll_cache::TensorView<uint8_t>(old_block_access)[local_loc];
    new_block_access_view = coll_cache::TensorView<uint8_t>(new_block_access)[local_loc];
  }
  bool operator()(const IdType & key){
    auto old_block_id = old_nid_to_block->CPtr<IdType>()[key];
    int old_loc = old_block_access_view[old_block_id].ref();
    if (old_loc == local_loc || old_loc == cpu_loc) return false;
    auto new_block_id = new_nid_to_block->CPtr<IdType>()[key];
    return new_block_access_view[new_block_id].ref() != old_loc;
    // return (new_block_placement->CPtr<uint8_t>()[new_block_id] & (1 << old_loc)) == 0;
  }
};

#ifdef COLL_HASH_VALID_LEGACY
void RefreshSession::refresh_after_solve(bool foreground) {
  auto _new_hash_table = _cache_ctx->_new_hash_table;
  Context gpu_ctx = _cache_ctx->_trainer_ctx;
  auto _hash_table_location = _cache_ctx->_hash_table_location;
  auto _hash_table_offset = _cache_ctx->_hash_table_offset;
  auto num_total_nodes = RunConfig::num_total_item;
  auto _local_location_id = _cache_ctx->_local_location_id;
  auto cu_stream = reinterpret_cast<cudaStream_t>(stream);

#ifdef DEAD_CODE
  {
    auto hs_loc = Tensor::CopyBlob(_cache_ctx->_hash_table_location, kI32, {RunConfig::num_total_item}, gpu_ctx, CPU(), "", stream);
    size_t local_nodes = 0;
    for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
      if (hs_loc->CPtr<IdType>()[node_id] == _local_location_id) local_nodes++;
    }
    CHECK(local_nodes == _cache_ctx->_cache_nodes) << local_nodes << "!=" << _cache_ctx->_cache_nodes;
  }
#endif

  TensorPtr block_access_advise_cpu = Tensor::CopyLine(_cache_ctx->_coll_cache->_block_access_advise, _local_location_id, CPU(CPU_CLIB_MALLOC_DEVICE), stream); // small
  size_t num_blocks = _cache_ctx->_coll_cache->_block_placement->Shape()[0];

  size_t per_src_size[9] = {0};

  for (size_t i = 0; i < num_blocks; i++) {
    IdType src = block_access_advise_cpu->CPtr<uint8_t>()[i];
    per_src_size[src] += (_cache_ctx->_coll_cache->_block_density->CPtr<double>()[i] + 0.1) * RunConfig::num_total_item / 100;
  }

  TensorPtr node_list_of_src[9] = {nullptr};
  #pragma omp parallel for
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making remote node list for " << dev_id;
      if (per_src_size[dev_id] == 0) continue;
      node_list_of_src[dev_id] = Tensor::Empty(kI32, {per_src_size[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
      size_t next_idx = 0;
      for (size_t node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
        IdType block_id = _cache_ctx->_coll_cache->_nid_to_block->CPtr<IdType>()[node_id];
        if (block_access_advise_cpu->CPtr<uint8_t>()[block_id] != dev_id) continue;
        node_list_of_src[dev_id]->Ptr<IdType>()[next_idx] = node_id;
        next_idx++;
      }
      per_src_size[dev_id] = next_idx;
      // check_uniq(node_list_of_src[dev_id]->CPtr<IdType>(), next_idx);
    }
  }
  TensorPtr node_list_of_src_cmp[9] = {nullptr};
  {
    auto coll_cache = _cache_ctx->_coll_cache;
    CacheEntryManager::DetectKeysForAllSource(coll_cache->_nid_to_block, block_access_advise_cpu, _local_location_id, coll_cache->_block_density, RunConfig::num_total_item, node_list_of_src_cmp);

    for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
      for (auto dev_id : link) {
        if (node_list_of_src_cmp[dev_id] == nullptr) {
          CHECK(per_src_size[dev_id] == 0);
        } else {
          CHECK_EQ(node_list_of_src_cmp[dev_id]->NumItem(), per_src_size[dev_id]) << node_list_of_src_cmp[dev_id]->NumItem() << "!=" << per_src_size[dev_id];
          CheckCpuEqual(node_list_of_src_cmp[dev_id]->CPtr<IdType>(), node_list_of_src[dev_id]->CPtr<IdType>(), node_list_of_src_cmp[dev_id]->NumBytes());
        }
      }
    }
  }

  // figure out new local cache id list
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making local node list";
  // new_local_node is ordered by key
  TensorPtr new_local_node_list_cpu = Tensor::Empty(kI32, {_cache_ctx->_cache_space_capacity}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  size_t num_new_local_node = 0;
  for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
    IdType block_id = _cache_ctx->_coll_cache->_nid_to_block->Ptr<IdType>()[node_id];
    if ((_cache_ctx->_coll_cache->_block_placement->Ptr<uint8_t>()[block_id] & (1 << _local_location_id)) != 0) {
      new_local_node_list_cpu->Ptr<IdType>()[num_new_local_node] = node_id;
      num_new_local_node++;
    }
  }
  // check_uniq(new_local_node_list_cpu->Ptr<IdType>(), num_new_local_node);
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new local node = " << num_new_local_node;
  {
    CHECK_EQ(num_new_local_node, node_list_of_src_cmp[_local_location_id]->NumItem());
    CheckCpuEqual(node_list_of_src_cmp[_local_location_id]->CPtr<IdType>(), new_local_node_list_cpu->CPtr<IdType>(), node_list_of_src_cmp[_local_location_id]->NumBytes());
  }

  CHECK(num_new_local_node <= _cache_ctx->_cache_space_capacity) << "cache space can not hold refreshed new cache";
  TensorPtr new_local_node_list_gpu = Tensor::CopyToExternal(new_local_node_list_cpu, _cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding preserved node and new insert nodes";
  // figure out preserved node id list for later extract it's used offset, and newly inserted node id list for later insertion
  TensorPtr preserved_node_list_gpu = Tensor::EmptyExternal(kI32, {num_new_local_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  TensorPtr new_insert_node_list_gpu = Tensor::EmptyExternal(kI32, {num_new_local_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  size_t num_preserved_node, num_new_insert_node_;
  // preserved_key, new_insert_key should be ordered by key
  cuda::CubSelectBySideNe<IdType>(gpu_ctx, (const IdType *)new_local_node_list_gpu->CPtr<IdType>(), num_new_local_node, _cache_ctx->_hash_table_location, new_insert_node_list_gpu->Ptr<IdType>(), num_new_insert_node_, _local_location_id, _cache_ctx->_gpu_mem_allocator, stream);
  cuda::CubSelectBySideEq<IdType>(gpu_ctx, (const IdType *)new_local_node_list_gpu->CPtr<IdType>(), num_new_local_node, _cache_ctx->_hash_table_location, preserved_node_list_gpu->Ptr<IdType>(), num_preserved_node, _local_location_id, _cache_ctx->_gpu_mem_allocator, stream);
  CHECK(num_preserved_node + num_new_insert_node_ == num_new_local_node);
  new_local_node_list_gpu = nullptr;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "preserved node = " << num_preserved_node;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new insert node = " << num_new_insert_node_;
  TensorPtr _new_insert_keys;
  {
    TensorPtr _new_local_keys = node_list_of_src_cmp[_local_location_id];
    _new_insert_keys = _new_hash_table->DetectKeysWithPlacement(
        _new_local_keys->CPtr<IdType>(), 
        _new_local_keys->Shape()[0], 
        _cache_ctx->_coll_cache->_old_nid_to_block, 
        _cache_ctx->_coll_cache->_old_block_placement, 
        CacheEntryManager::PlaceOn<false>(_local_location_id));
    LOG(ERROR) << " new hashtable find new local key done";
    CHECK_EQ(num_new_insert_node_, _new_insert_keys->NumItem());
    TensorPtr _new_insert_node_list_cpu = Tensor::CopyTo(new_insert_node_list_gpu, CPU(), stream, "");
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    CheckCpuEqual(_new_insert_node_list_cpu->Data(), _new_insert_keys->Data(), _new_insert_keys->NumBytes());
    // preserved keys is not necessary when using new hashtable
    LOG(ERROR) << " new hashtable new local key done";
    AnonymousBarrier::_refresh_instance->Wait();
  }

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "gathering unused offsets";
  // get free offset by select preserved node's offset, mark these offset, and select unused offset
  TensorPtr preserved_node_offset_list_gpu = Tensor::EmptyExternal(kI32, {num_preserved_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  auto local_offset_vocab_inuse_mark = Tensor::EmptyExternal(kI32, {_cache_ctx->_cache_space_capacity}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  // auto local_offset_vocab_inuse_mark = Tensor::EmptyExternal(kU8, {_cache_ctx->_cache_space_capacity}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  cuda::ArrangeArray<IdType>(local_offset_vocab_inuse_mark->Ptr<IdType>(), _cache_ctx->_cache_space_capacity, 0, 0, stream);
#ifdef DEAD_CODE
  TensorPtr inuse_offset = nullptr;
  TensorPtr preserved_node_offset_list_cpu = nullptr;
#endif
  if (num_preserved_node > 0) {
#ifdef DEAD_CODE
    {
      TensorPtr preserved_node_list_cpu = Tensor::CopyTo(preserved_node_list_gpu, CPU(), stream, "");
      check_uniq(preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node);
      TensorPtr new_insert_node_list_cpu = Tensor::CopyTo(new_insert_node_list_gpu, CPU(), stream, "");
      check_uniq(new_insert_node_list_cpu->CPtr<IdType>(), num_new_insert_node_);
      check_no_intersec(preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node, new_insert_node_list_cpu->CPtr<IdType>(), num_new_insert_node_);
    }
#endif
    SAM_CUDA_PREPARE_1D(num_preserved_node);
    preserve_offset<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(preserved_node_offset_list_gpu->Ptr<IdType>(), preserved_node_list_gpu->Ptr<IdType>(), num_preserved_node, _cache_ctx->_hash_table_offset);
#ifdef DEAD_CODE
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    {
      preserved_node_offset_list_cpu = Tensor::CopyTo(preserved_node_offset_list_gpu, CPU(), stream, "");
      check_uniq(preserved_node_offset_list_cpu->CPtr<IdType>(), num_preserved_node);
    }
#endif
    mark_offset_inuse<Constant::kCudaBlockSize, Constant::kCudaTileSize, IdType><<<grid, block, 0, cu_stream>>>(preserved_node_offset_list_gpu->CPtr<IdType>(), num_preserved_node, local_offset_vocab_inuse_mark->Ptr<IdType>());
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
#ifdef DEAD_CODE
    {
      TensorPtr local_offset_vocab_inuse_mark_cpu = Tensor::CopyTo(local_offset_vocab_inuse_mark, CPU(), stream, "");
      size_t sum = 0;
      for (IdType i = 0; i < _cache_ctx->_cache_space_capacity; i++) {
        sum += local_offset_vocab_inuse_mark_cpu->CPtr<IdType>()[i];
      }
      CHECK(sum == num_preserved_node);
    }
    {
      size_t num_inuse_offset = 0;
      cuda::CubCountByEq<IdType>(gpu_ctx, local_offset_vocab_inuse_mark->CPtr<IdType>(), _cache_ctx->_cache_space_capacity, num_inuse_offset, 1, _cache_ctx->_gpu_mem_allocator, stream);
      CHECK(num_inuse_offset == num_preserved_node) << num_inuse_offset << "!=" << num_preserved_node;
      inuse_offset = Tensor::EmptyExternal(common::kI32, {num_preserved_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
      cuda::CubSelectIndexByEqSide<IdType, IdType>(gpu_ctx,
        local_offset_vocab_inuse_mark->CPtr<IdType>(), _cache_ctx->_cache_space_capacity, inuse_offset->Ptr<IdType>(), num_inuse_offset, 1, _cache_ctx->_gpu_mem_allocator, stream);
      CHECK(num_inuse_offset == num_preserved_node) << num_inuse_offset << "!=" << num_preserved_node;
      inuse_offset = Tensor::CopyTo(inuse_offset, CPU(), stream, "");
      if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num in use offset = " << num_inuse_offset;
    }
#endif
  }
  preserved_node_list_gpu = nullptr;
  preserved_node_offset_list_gpu = nullptr;
  CHECK(num_preserved_node < _cache_ctx->_cache_space_capacity);
  auto nouse_offset = Tensor::EmptyExternal(common::kI32, {_cache_ctx->_cache_space_capacity - num_preserved_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");

  size_t num_nouse_offset = 0;
  // the selected no use offset, should be ordered from small to large
  cuda::CubSelectIndexByEqSide<IdType, IdType>(gpu_ctx,
    local_offset_vocab_inuse_mark->CPtr<IdType>(), _cache_ctx->_cache_space_capacity, nouse_offset->Ptr<IdType>(), num_nouse_offset, 0, _cache_ctx->_gpu_mem_allocator, stream);
  CHECK(num_nouse_offset + num_preserved_node == _cache_ctx->_cache_space_capacity) << num_nouse_offset << "+" << num_preserved_node << "!=" << _cache_ctx->_cache_space_capacity;
  local_offset_vocab_inuse_mark = nullptr;
#ifdef DEAD_CODE
  {
    auto nouse_offset_cpu = Tensor::CopyTo(nouse_offset, CPU(), stream, "");
    check_no_intersec(inuse_offset->CPtr<IdType>(), num_preserved_node, nouse_offset_cpu->CPtr<IdType>(), num_nouse_offset);
    check_covers(inuse_offset->CPtr<IdType>(), num_preserved_node, preserved_node_offset_list_cpu->CPtr<IdType>(), num_preserved_node);
  }
#endif
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num no use offset = " << num_nouse_offset;

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding evicted nodes";
  // copy old cache node list to cpu, and find those evicted
  TensorPtr cache_node_list_cpu = _cache_ctx->_local_node_list_tensor;
  TensorPtr evicted_node_list_cpu = Tensor::Empty(kI32, {_cache_ctx->_cache_nodes - num_preserved_node}, CPU(CPU_CLIB_MALLOC_DEVICE), "");

  size_t num_eviced_node = 0;
  for (IdType i = 0; i < _cache_ctx->_cache_nodes; i++) {
    IdType node_id = cache_node_list_cpu->Ptr<IdType>()[i];
    IdType block_id = _cache_ctx->_coll_cache->_nid_to_block->Ptr<IdType>()[node_id];
    if ((_cache_ctx->_coll_cache->_block_placement->Ptr<uint8_t>()[block_id] & (1 << _local_location_id)) == 0) {
      evicted_node_list_cpu->Ptr<IdType>()[num_eviced_node] = node_id;
      num_eviced_node++;
    }
  }
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num evicted node = " << num_eviced_node;
  LOG(ERROR) << "old num evicted node = " << num_eviced_node;

  TensorPtr _new_evict_keys_gpu;
  {
    AnonymousBarrier::_refresh_instance->Wait();
    TensorPtr _new_evict_keys_cpu = _new_hash_table->DetectKeysWithPlacement(
        _new_hash_table->_cached_keys->CPtr<IdType>(), 
        _new_hash_table->_cached_keys->Shape()[0], 
        _cache_ctx->_coll_cache->_nid_to_block, 
        _cache_ctx->_coll_cache->_block_placement, 
        CacheEntryManager::PlaceOn<false>(_local_location_id));
    CHECK_EQ(num_eviced_node, _new_evict_keys_cpu->NumItem()) << num_eviced_node << "!=" << _new_evict_keys_cpu->NumItem();
    CheckCpuEqual(evicted_node_list_cpu->Data(), _new_evict_keys_cpu->Data(), _new_evict_keys_cpu->NumBytes());
    LOG(ERROR) << "evict keys checked";
    AnonymousBarrier::_refresh_instance->Wait();

    _new_evict_keys_gpu = Tensor::CopyToExternal(_new_evict_keys_cpu, _cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    auto _new_evict_offsets_gpu = Tensor::EmptyExternal(kI32, {num_eviced_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    _new_hash_table->LookupOffset(_new_evict_keys_gpu, _new_evict_offsets_gpu, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    LOG(ERROR) << "evict keys offset fetched";
    LOG(ERROR) << "num _evicted node = " << num_eviced_node;
    AnonymousBarrier::_refresh_instance->Wait();
    auto _new_evict_offsets_cpu = _new_evict_offsets_gpu->CopyTo(CPU(), stream, "");
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    LOG(ERROR) << "evict keys offset to cpu";
    AnonymousBarrier::_refresh_instance->Wait();
    LOG(ERROR) << "#new return offset=" << _new_evict_offsets_cpu->NumItem();
    _new_hash_table->ReturnOffset(_new_evict_offsets_cpu);
    LOG(ERROR) << "evict keys offset returned";
    AnonymousBarrier::_refresh_instance->Wait();
    _new_hash_table->SortFreeOffsets();
    LOG(ERROR) << "evict keys offset sorted";
    AnonymousBarrier::_refresh_instance->Wait();

    CHECK_EQ(_new_hash_table->_free_offsets->NumItem(), num_nouse_offset);
    auto nouse_offset_cpu = Tensor::CopyTo(nouse_offset, CPU(), stream, "");
    CheckCpuEqual(nouse_offset_cpu->Data(), _new_hash_table->_free_offsets->Data(), nouse_offset_cpu->NumBytes());
  }

#ifdef DEAD_CODE
  {
    TensorPtr preserved_node_list_cpu = Tensor::CopyTo(preserved_node_list_gpu, CPU(), stream, "");
    check_no_intersec(preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node, evicted_node_list_cpu->CPtr<IdType>(), num_eviced_node);
  }
#endif

  CHECK(num_eviced_node + num_preserved_node == _cache_ctx->_cache_nodes);
  CHECK(num_eviced_node <= num_nouse_offset);

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting nodes";
  // evict it

#ifdef DEAD_CODE
  {
    auto hs_loc = Tensor::CopyBlob(_cache_ctx->_hash_table_location, kI32, {RunConfig::num_total_item}, gpu_ctx, CPU(), "", stream);
    size_t local_nodes = 0;
    for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
      if (hs_loc->CPtr<IdType>()[node_id] == _local_location_id) local_nodes++;
    }
    CHECK(local_nodes == _cache_ctx->_cache_nodes) << local_nodes << "!=" << _cache_ctx->_cache_nodes;

    for (IdType idx = 0; idx < num_eviced_node; idx++) {
      CHECK(hs_loc->CPtr<IdType>()[evicted_node_list_cpu->CPtr<IdType>()[idx]] == _local_location_id);
    }
  }
#endif

  if (num_eviced_node > 0) {
    TensorPtr evicted_node_list_gpu = Tensor::CopyToExternal(evicted_node_list_cpu, _cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);
    SAM_CUDA_PREPARE_1D(num_eviced_node);
    if (RunConfig::option_empty_feat) {
      mark_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, true>
        <<<grid, block, 0, cu_stream>>>(
          evicted_node_list_gpu->CPtr<IdType>(), num_eviced_node, 
          _cache_ctx->_cpu_location_id, 
          _local_location_id,
          _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, RunConfig::option_empty_feat);
    } else {
      mark_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(
          evicted_node_list_gpu->CPtr<IdType>(), num_eviced_node, 
          _cache_ctx->_cpu_location_id, 
          _local_location_id,
          _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset);
    }
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }
  {
    _cache_ctx->_new_hash_table->Evict(_new_evict_keys_gpu, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    _cache_ctx->compare_hashtable(stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    AnonymousBarrier::_refresh_instance->Wait();
  }
#ifdef DEAD_CODE
  {
    auto hs_loc = Tensor::CopyBlob(_cache_ctx->_hash_table_location, kI32, {RunConfig::num_total_item}, gpu_ctx, CPU(), "", stream);
    size_t local_nodes = 0;
    for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
      if (hs_loc->CPtr<IdType>()[node_id] == _local_location_id) local_nodes++;
    }
    CHECK(local_nodes == num_preserved_node) << local_nodes << "!=" << num_preserved_node;
  }
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evict local node done, checking hashtable";
  {
    TensorPtr preserved_node_list_cpu = Tensor::CopyTo(preserved_node_list_gpu, CPU(), stream, "");
    auto cache_node_list = Tensor::EmptyExternal(kI32, {_cache_ctx->_cache_nodes}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    size_t num_cached_nodes;
    cuda::CubSelectIndexByEqSide<IdType, IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, cache_node_list->Ptr<IdType>(), num_cached_nodes, _local_location_id, _cache_ctx->_gpu_mem_allocator, stream);
    auto cache_node_list_cpu = Tensor::CopyTo(cache_node_list, CPU(), stream, "");
    CHECK(num_cached_nodes == num_preserved_node) << num_cached_nodes << "!=" << num_preserved_node;
    check_covers(cache_node_list_cpu->CPtr<IdType>(), num_cached_nodes, preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node);
    if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "extracting offset of preserved nodes";
    auto offset_list = Tensor::EmptyExternal(kI32, {num_preserved_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    {
      SAM_CUDA_PREPARE_1D(num_preserved_node);
      preserve_offset<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(offset_list->Ptr<IdType>(), cache_node_list->Ptr<IdType>(), num_preserved_node, _cache_ctx->_hash_table_offset);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
    }
    offset_list = Tensor::CopyTo(offset_list, CPU(), stream, "");
    check_covers(preserved_node_offset_list_cpu->CPtr<IdType>(), num_preserved_node, offset_list->CPtr<IdType>(), num_preserved_node);
  }
#endif

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting remote nodes";
  {
    _LocationPack _location_param;
    memcpy(_location_param.data, _cache_ctx->_remote_hash_table_location.data(), sizeof(HashTableEntryLocation*) * _cache_ctx->_remote_hash_table_location.size());
    SAM_CUDA_PREPARE_1D(RunConfig::num_total_item);
    if (RunConfig::option_empty_feat) {
      mark_remote_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, true>
        <<<grid, block, 0, cu_stream>>>(_location_param, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, 
          _cache_ctx->_local_location_id, _cache_ctx->_cpu_location_id, RunConfig::num_total_item, RunConfig::option_empty_feat);
    } else {
      mark_remote_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, false>
        <<<grid, block, 0, cu_stream>>>(_location_param, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, 
          _cache_ctx->_local_location_id, _cache_ctx->_cpu_location_id, RunConfig::num_total_item);
    }
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }
  {
    LOG(ERROR) << "remote nodes to evict detecting";
    auto remote_keys_to_evict = _cache_ctx->_new_hash_table->DetectKeysWithCond(
        cub::CountingInputIterator<IdType>(0), 
        RunConfig::num_total_item, 
        SelectEvictedRemoteKeys(
            _cache_ctx->_coll_cache->_old_nid_to_block,
            _cache_ctx->_coll_cache->_old_block_access_advise,
            _cache_ctx->_coll_cache->_nid_to_block,
            _cache_ctx->_coll_cache->_block_access_advise,
            nullptr, // _cache_ctx->_coll_cache->_block_placement,
            _local_location_id, _cache_ctx->_cpu_location_id
          ),
        _new_hash_table->_hash_table->max_efficient_size
      );
    AnonymousBarrier::_refresh_instance->Wait();
    LOG(ERROR) << "remote nodes to evict detected";
    auto remote_keys_to_evict_gpu = Tensor::CopyToExternal(remote_keys_to_evict, _cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    AnonymousBarrier::_refresh_instance->Wait();
    LOG(ERROR) << "calling evict remote";
    _cache_ctx->_new_hash_table->EvictRemote(remote_keys_to_evict_gpu, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    AnonymousBarrier::_refresh_instance->Wait();
  }

  size_t current_progress = _cache_ctx->progress.load();

  AnonymousBarrier::_refresh_instance->Wait();

  Timer t0;
  while (!foreground && _cache_ctx->progress.load() < current_progress + 1) {
    if (t0.PassedSec() >= 20) break;
  }

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new local nodes";
  // fixme: wait for current extraction 
  size_t num_new_insert_node = num_new_local_node - num_preserved_node;
  const DataIter<const IdType*> src_data_iter(new_insert_node_list_gpu->CPtr<IdType>(), _cache_ctx->_device_cache_data[_cache_ctx->_cpu_location_id], _cache_ctx->_dim);
  DataIter<FreeOffIter> dst_data_iter(FreeOffIter(nouse_offset->Ptr<IdType>()), _cache_ctx->_device_cache_data[_cache_ctx->_local_location_id], _cache_ctx->_dim);
  Combine(src_data_iter, dst_data_iter, num_new_insert_node, gpu_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);

  LOG(INFO) << "CollCacheManager: fix location and offset of local nodes in hash table";
  if (num_new_insert_node > 0) {
    SAM_CUDA_PREPARE_1D(num_new_insert_node);
    refresh_hash_table_local<><<<grid, block, 0, cu_stream>>>(
        _hash_table_location, _hash_table_offset, 
        new_insert_node_list_gpu->CPtr<IdType>(), num_new_insert_node, nouse_offset->CPtr<IdType>(), _local_location_id);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }
  {
    auto reserved_offset = _new_hash_table->ReserveOffsetFront(_new_insert_keys->NumItem())->CopyToExternal(_cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);
    _new_insert_keys = _new_insert_keys->CopyToExternal(_cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);
    _new_hash_table->InsertWithLoc(_new_insert_keys, reserved_offset, _local_location_id, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));

  }
  new_insert_node_list_gpu = nullptr;
#ifdef DEAD_CODE
  if (num_new_insert_node > 0){
    auto new_insert_offset_list = Tensor::EmptyExternal(kI32, {num_new_insert_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    SAM_CUDA_PREPARE_1D(num_new_insert_node);
    preserve_offset<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(new_insert_offset_list->Ptr<IdType>(), new_insert_node_list_gpu->CPtr<IdType>(), num_new_insert_node, _cache_ctx->_hash_table_offset);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));

    new_insert_offset_list = Tensor::CopyTo(new_insert_offset_list, CPU(), stream, "");
    auto nouse_offset_cpu = Tensor::CopyTo(nouse_offset, CPU(), stream, "");
    check_uniq(new_insert_offset_list->CPtr<IdType>(), num_new_insert_node);
    check_covers(nouse_offset_cpu->CPtr<IdType>(), num_nouse_offset, new_insert_offset_list->CPtr<IdType>(), num_new_insert_node);
    check_no_intersec(new_insert_offset_list->CPtr<IdType>(), num_new_insert_node, inuse_offset->CPtr<IdType>(), num_preserved_node);
  }
#endif

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new remote nodes";
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      size_t num_remote_nodes = per_src_size[dev_id];
      if (num_remote_nodes == 0) continue;
      TensorPtr remote_node_list_tensor = Tensor::CopyToExternal(node_list_of_src[dev_id], _cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);

      SAM_CUDA_PREPARE_1D(num_remote_nodes);
      init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
          _hash_table_location, _hash_table_offset, 
          _cache_ctx->_remote_hash_table_offset[dev_id], remote_node_list_tensor->CPtr<IdType>(), num_remote_nodes, dev_id);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
      {
        auto _old_access_view = coll_cache::TensorView<uint8_t>(_cache_ctx->_coll_cache->_old_block_access_advise)[_local_location_id];
        auto _remote_new_keys = _new_hash_table->DetectKeysWithCond(node_list_of_src_cmp[dev_id]->CPtr<IdType>(), node_list_of_src_cmp[dev_id]->NumItem(), [this, _old_access_view, dev_id](const IdType & key) mutable{
          auto block_id = _cache_ctx->_coll_cache->_old_nid_to_block->CPtr<IdType>()[key];
          return _old_access_view[block_id].ref() != dev_id;
        })->CopyToExternal(_cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);
        auto _remote_offsets = Tensor::EmptyExternal(kI32, _remote_new_keys->Shape(), _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
        _cache_ctx->_remote_new_hash_table[dev_id]->LookupOffset(_remote_new_keys, _remote_offsets, stream);
        _new_hash_table->InsertWithLoc(_remote_new_keys, _remote_offsets, dev_id, stream);
      }
    }
  }

  AnonymousBarrier::_refresh_instance->Wait();

  _cache_ctx->compare_hashtable(stream);
  CUDA_CALL(cudaStreamSynchronize(cu_stream));

  if (_cache_ctx->_local_location_id == 0) LOG(INFO) << "refresh done";

  _cache_ctx->_cache_nodes = num_new_local_node;
  _cache_ctx->_local_node_list_tensor = new_local_node_list_cpu;
  _new_hash_table->_cached_keys = node_list_of_src_cmp[_local_location_id];
}
#endif

#ifdef COLL_HASH_VALID_LEGACY
void RefreshSession::refresh_after_solve_old(bool foreground) {
  Context gpu_ctx = _cache_ctx->_trainer_ctx;
  auto _hash_table_location = _cache_ctx->_hash_table_location;
  auto _hash_table_offset = _cache_ctx->_hash_table_offset;
  auto num_total_nodes = RunConfig::num_total_item;
  auto _local_location_id = _cache_ctx->_local_location_id;
  auto cu_stream = reinterpret_cast<cudaStream_t>(stream);

#ifdef DEAD_CODE
  {
    auto hs_loc = Tensor::CopyBlob(_cache_ctx->_hash_table_location, kI32, {RunConfig::num_total_item}, gpu_ctx, CPU(), "", stream);
    size_t local_nodes = 0;
    for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
      if (hs_loc->CPtr<IdType>()[node_id] == _local_location_id) local_nodes++;
    }
    CHECK(local_nodes == _cache_ctx->_cache_nodes) << local_nodes << "!=" << _cache_ctx->_cache_nodes;
  }
#endif

  TensorPtr block_access_advise_cpu = Tensor::CopyLine(_cache_ctx->_coll_cache->_block_access_advise, _local_location_id, CPU(CPU_CLIB_MALLOC_DEVICE), stream); // small
  size_t num_blocks = _cache_ctx->_coll_cache->_block_placement->Shape()[0];

  size_t per_src_size[9] = {0};

  for (size_t i = 0; i < num_blocks; i++) {
    IdType src = block_access_advise_cpu->CPtr<uint8_t>()[i];
    per_src_size[src] += (_cache_ctx->_coll_cache->_block_density->CPtr<double>()[i] + 0.1) * RunConfig::num_total_item / 100;
  }

  TensorPtr node_list_of_src[9] = {nullptr};
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making remote node lists";
  #pragma omp parallel for
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      // if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making remote node list for " << dev_id;
      if (per_src_size[dev_id] == 0) continue;
      node_list_of_src[dev_id] = Tensor::Empty(kI32, {per_src_size[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
      size_t next_idx = 0;
      for (size_t node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
        IdType block_id = _cache_ctx->_coll_cache->_nid_to_block->CPtr<IdType>()[node_id];
        if (block_access_advise_cpu->CPtr<uint8_t>()[block_id] != dev_id) continue;
        CHECK(next_idx < per_src_size[dev_id]);
        node_list_of_src[dev_id]->Ptr<IdType>()[next_idx] = node_id;
        next_idx++;
      }
      per_src_size[dev_id] = next_idx;
      // check_uniq(node_list_of_src[dev_id]->CPtr<IdType>(), next_idx);
    }
  }

  // figure out new local cache id list
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making local node list";
  TensorPtr new_local_node_list_cpu = Tensor::Empty(kI32, {_cache_ctx->_cache_space_capacity}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  size_t num_new_local_node = 0;
  for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
    IdType block_id = _cache_ctx->_coll_cache->_nid_to_block->Ptr<IdType>()[node_id];
    if ((_cache_ctx->_coll_cache->_block_placement->Ptr<uint8_t>()[block_id] & (1 << _local_location_id)) != 0) {
      CHECK(num_new_local_node < _cache_ctx->_cache_space_capacity) << num_new_local_node << " >=" << _cache_ctx->_cache_space_capacity;
      new_local_node_list_cpu->Ptr<IdType>()[num_new_local_node] = node_id;
      num_new_local_node++;
    }
  }
  // check_uniq(new_local_node_list_cpu->Ptr<IdType>(), num_new_local_node);
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new local node = " << num_new_local_node;

  CHECK(num_new_local_node <= _cache_ctx->_cache_space_capacity) << "cache space can not hold refreshed new cache";
  TensorPtr new_local_node_list_gpu = Tensor::CopyToExternal(new_local_node_list_cpu, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, stream);

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding preserved node and new insert nodes";
  // figure out preserved node id list for later extract it's used offset, and newly inserted node id list for later insertion
  TensorPtr preserved_node_list_gpu = Tensor::EmptyExternal(kI32, {num_new_local_node}, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, "");
  TensorPtr new_insert_node_list_gpu = Tensor::EmptyExternal(kI32, {num_new_local_node}, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, "");
  size_t num_preserved_node, num_new_insert_node_;
  cuda::CubSelectBySideNe<IdType>(gpu_ctx, (const IdType *)new_local_node_list_gpu->CPtr<IdType>(), num_new_local_node, _cache_ctx->_hash_table_location, new_insert_node_list_gpu->Ptr<IdType>(), num_new_insert_node_, _local_location_id, _cache_ctx->_eager_gpu_mem_allocator, stream);
  cuda::CubSelectBySideEq<IdType>(gpu_ctx, (const IdType *)new_local_node_list_gpu->CPtr<IdType>(), num_new_local_node, _cache_ctx->_hash_table_location, preserved_node_list_gpu->Ptr<IdType>(), num_preserved_node, _local_location_id, _cache_ctx->_eager_gpu_mem_allocator, stream);
  CHECK(num_preserved_node + num_new_insert_node_ == num_new_local_node);
  new_local_node_list_gpu = nullptr;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "preserved node = " << num_preserved_node;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new insert node = " << num_new_insert_node_;

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "gathering unused offsets";
  // get free offset by select preserved node's offset, mark these offset, and select unused offset
  TensorPtr preserved_node_offset_list_gpu = Tensor::EmptyExternal(kI32, {num_preserved_node}, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, "");
  auto local_offset_vocab_inuse_mark = Tensor::EmptyExternal(kI32, {_cache_ctx->_cache_space_capacity}, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, "");
  // auto local_offset_vocab_inuse_mark = Tensor::EmptyExternal(kU8, {_cache_ctx->_cache_space_capacity}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  cuda::ArrangeArray<IdType>(local_offset_vocab_inuse_mark->Ptr<IdType>(), _cache_ctx->_cache_space_capacity, 0, 0, stream);
#ifdef DEAD_CODE
  TensorPtr inuse_offset = nullptr;
  TensorPtr preserved_node_offset_list_cpu = nullptr;
#endif
  if (num_preserved_node > 0) {
#ifdef DEAD_CODE
    {
      TensorPtr preserved_node_list_cpu = Tensor::CopyTo(preserved_node_list_gpu, CPU(), stream, "");
      check_uniq(preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node);
      TensorPtr new_insert_node_list_cpu = Tensor::CopyTo(new_insert_node_list_gpu, CPU(), stream, "");
      check_uniq(new_insert_node_list_cpu->CPtr<IdType>(), num_new_insert_node_);
      check_no_intersec(preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node, new_insert_node_list_cpu->CPtr<IdType>(), num_new_insert_node_);
    }
#endif
    SAM_CUDA_PREPARE_1D(num_preserved_node);
    preserve_offset<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(preserved_node_offset_list_gpu->Ptr<IdType>(), preserved_node_list_gpu->Ptr<IdType>(), num_preserved_node, _cache_ctx->_hash_table_offset);
#ifdef DEAD_CODE
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    {
      preserved_node_offset_list_cpu = Tensor::CopyTo(preserved_node_offset_list_gpu, CPU(), stream, "");
      check_uniq(preserved_node_offset_list_cpu->CPtr<IdType>(), num_preserved_node);
    }
#endif
    mark_offset_inuse<Constant::kCudaBlockSize, Constant::kCudaTileSize, IdType><<<grid, block, 0, cu_stream>>>(preserved_node_offset_list_gpu->CPtr<IdType>(), num_preserved_node, local_offset_vocab_inuse_mark->Ptr<IdType>());
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
#ifdef DEAD_CODE
    {
      TensorPtr local_offset_vocab_inuse_mark_cpu = Tensor::CopyTo(local_offset_vocab_inuse_mark, CPU(), stream, "");
      size_t sum = 0;
      for (IdType i = 0; i < _cache_ctx->_cache_space_capacity; i++) {
        sum += local_offset_vocab_inuse_mark_cpu->CPtr<IdType>()[i];
      }
      CHECK(sum == num_preserved_node);
    }
    {
      size_t num_inuse_offset = 0;
      cuda::CubCountByEq<IdType>(gpu_ctx, local_offset_vocab_inuse_mark->CPtr<IdType>(), _cache_ctx->_cache_space_capacity, num_inuse_offset, 1, _cache_ctx->_gpu_mem_allocator, stream);
      CHECK(num_inuse_offset == num_preserved_node) << num_inuse_offset << "!=" << num_preserved_node;
      inuse_offset = Tensor::EmptyExternal(common::kI32, {num_preserved_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
      cuda::CubSelectIndexByEqSide<IdType, IdType>(gpu_ctx,
        local_offset_vocab_inuse_mark->CPtr<IdType>(), _cache_ctx->_cache_space_capacity, inuse_offset->Ptr<IdType>(), num_inuse_offset, 1, _cache_ctx->_gpu_mem_allocator, stream);
      CHECK(num_inuse_offset == num_preserved_node) << num_inuse_offset << "!=" << num_preserved_node;
      inuse_offset = Tensor::CopyTo(inuse_offset, CPU(), stream, "");
      if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num in use offset = " << num_inuse_offset;
    }
#endif
  }
  preserved_node_list_gpu = nullptr;
  preserved_node_offset_list_gpu = nullptr;
  CHECK(num_preserved_node < _cache_ctx->_cache_space_capacity);
  auto nouse_offset = Tensor::EmptyExternal(common::kI32, {_cache_ctx->_cache_space_capacity - num_preserved_node}, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, "");

  size_t num_nouse_offset = 0;
  cuda::CubSelectIndexByEqSide<IdType, IdType>(gpu_ctx,
    local_offset_vocab_inuse_mark->CPtr<IdType>(), _cache_ctx->_cache_space_capacity, nouse_offset->Ptr<IdType>(), num_nouse_offset, 0, _cache_ctx->_eager_gpu_mem_allocator, stream);
  CHECK(num_nouse_offset + num_preserved_node == _cache_ctx->_cache_space_capacity) << num_nouse_offset << "+" << num_preserved_node << "!=" << _cache_ctx->_cache_space_capacity;
  local_offset_vocab_inuse_mark = nullptr;
#ifdef DEAD_CODE
  {
    auto nouse_offset_cpu = Tensor::CopyTo(nouse_offset, CPU(), stream, "");
    check_no_intersec(inuse_offset->CPtr<IdType>(), num_preserved_node, nouse_offset_cpu->CPtr<IdType>(), num_nouse_offset);
    check_covers(inuse_offset->CPtr<IdType>(), num_preserved_node, preserved_node_offset_list_cpu->CPtr<IdType>(), num_preserved_node);
  }
#endif
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num no use offset = " << num_nouse_offset;

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding evicted nodes";
  // copy old cache node list to cpu, and find those evicted
  TensorPtr cache_node_list_cpu = _cache_ctx->_local_node_list_tensor;
  TensorPtr evicted_node_list_cpu;
  if (_cache_ctx->_cache_nodes > num_preserved_node) evicted_node_list_cpu = Tensor::Empty(kI32, {_cache_ctx->_cache_nodes - num_preserved_node}, CPU(CPU_CLIB_MALLOC_DEVICE), "");

  size_t num_eviced_node = 0;
  for (IdType i = 0; i < _cache_ctx->_cache_nodes; i++) {
    IdType node_id = cache_node_list_cpu->Ptr<IdType>()[i];
    IdType block_id = _cache_ctx->_coll_cache->_nid_to_block->Ptr<IdType>()[node_id];
    if ((_cache_ctx->_coll_cache->_block_placement->Ptr<uint8_t>()[block_id] & (1 << _local_location_id)) == 0) {
      CHECK(num_eviced_node < _cache_ctx->_cache_nodes - num_preserved_node);
      evicted_node_list_cpu->Ptr<IdType>()[num_eviced_node] = node_id;
      num_eviced_node++;
    }
  }
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num evicted node = " << num_eviced_node;

#ifdef DEAD_CODE
  {
    TensorPtr preserved_node_list_cpu = Tensor::CopyTo(preserved_node_list_gpu, CPU(), stream, "");
    check_no_intersec(preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node, evicted_node_list_cpu->CPtr<IdType>(), num_eviced_node);
  }
#endif

  CHECK(num_eviced_node + num_preserved_node == _cache_ctx->_cache_nodes);
  CHECK(num_eviced_node <= num_nouse_offset);

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting nodes";
  // evict it

#ifdef DEAD_CODE
  {
    auto hs_loc = Tensor::CopyBlob(_cache_ctx->_hash_table_location, kI32, {RunConfig::num_total_item}, gpu_ctx, CPU(), "", stream);
    size_t local_nodes = 0;
    for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
      if (hs_loc->CPtr<IdType>()[node_id] == _local_location_id) local_nodes++;
    }
    CHECK(local_nodes == _cache_ctx->_cache_nodes) << local_nodes << "!=" << _cache_ctx->_cache_nodes;

    for (IdType idx = 0; idx < num_eviced_node; idx++) {
      CHECK(hs_loc->CPtr<IdType>()[evicted_node_list_cpu->CPtr<IdType>()[idx]] == _local_location_id);
    }
  }
#endif

  if (num_eviced_node > 0) {
    TensorPtr evicted_node_list_gpu = Tensor::CopyToExternal(evicted_node_list_cpu, _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, stream);
    SAM_CUDA_PREPARE_1D(num_eviced_node);
    if (RunConfig::option_empty_feat) {
      mark_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, true>
        <<<grid, block, 0, cu_stream>>>(
          evicted_node_list_gpu->CPtr<IdType>(), num_eviced_node, 
          _cache_ctx->_cpu_location_id, 
          _local_location_id,
          _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, RunConfig::option_empty_feat);
    } else {
      mark_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, false>
        <<<grid, block, 0, cu_stream>>>(
          evicted_node_list_gpu->CPtr<IdType>(), num_eviced_node, 
          _cache_ctx->_cpu_location_id, 
          _local_location_id,
          _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset);
    }
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }
#ifdef DEAD_CODE
  {
    auto hs_loc = Tensor::CopyBlob(_cache_ctx->_hash_table_location, kI32, {RunConfig::num_total_item}, gpu_ctx, CPU(), "", stream);
    size_t local_nodes = 0;
    for (IdType node_id = 0; node_id < RunConfig::num_total_item; node_id++) {
      if (hs_loc->CPtr<IdType>()[node_id] == _local_location_id) local_nodes++;
    }
    CHECK(local_nodes == num_preserved_node) << local_nodes << "!=" << num_preserved_node;
  }
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evict local node done, checking hashtable";
  {
    TensorPtr preserved_node_list_cpu = Tensor::CopyTo(preserved_node_list_gpu, CPU(), stream, "");
    auto cache_node_list = Tensor::EmptyExternal(kI32, {_cache_ctx->_cache_nodes}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    size_t num_cached_nodes;
    cuda::CubSelectIndexByEqSide<IdType, IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, cache_node_list->Ptr<IdType>(), num_cached_nodes, _local_location_id, _cache_ctx->_gpu_mem_allocator, stream);
    auto cache_node_list_cpu = Tensor::CopyTo(cache_node_list, CPU(), stream, "");
    CHECK(num_cached_nodes == num_preserved_node) << num_cached_nodes << "!=" << num_preserved_node;
    check_covers(cache_node_list_cpu->CPtr<IdType>(), num_cached_nodes, preserved_node_list_cpu->CPtr<IdType>(), num_preserved_node);
    if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "extracting offset of preserved nodes";
    auto offset_list = Tensor::EmptyExternal(kI32, {num_preserved_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    {
      SAM_CUDA_PREPARE_1D(num_preserved_node);
      preserve_offset<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(offset_list->Ptr<IdType>(), cache_node_list->Ptr<IdType>(), num_preserved_node, _cache_ctx->_hash_table_offset);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
    }
    offset_list = Tensor::CopyTo(offset_list, CPU(), stream, "");
    check_covers(preserved_node_offset_list_cpu->CPtr<IdType>(), num_preserved_node, offset_list->CPtr<IdType>(), num_preserved_node);
  }
#endif

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting remote nodes";
  {
    _LocationPack _location_param;
    memcpy(_location_param.data, _cache_ctx->_remote_hash_table_location.data(), sizeof(HashTableEntryLocation*) * _cache_ctx->_remote_hash_table_location.size());
    SAM_CUDA_PREPARE_1D(RunConfig::num_total_item);
    if (RunConfig::option_empty_feat) {
      mark_remote_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, true>
        <<<grid, block, 0, cu_stream>>>(_location_param, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, 
          _cache_ctx->_local_location_id, _cache_ctx->_cpu_location_id, RunConfig::num_total_item, RunConfig::option_empty_feat);
    } else {
      mark_remote_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize, false>
        <<<grid, block, 0, cu_stream>>>(_location_param, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, 
          _cache_ctx->_local_location_id, _cache_ctx->_cpu_location_id, RunConfig::num_total_item);
    }
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }

  size_t current_progress = _cache_ctx->progress.load();

  AnonymousBarrier::_refresh_instance->Wait();

  Timer t0;
  while (!foreground && _cache_ctx->progress.load() < current_progress + 1) {
    if (t0.PassedSec() >= 20) break;
  }

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new local nodes";
  // fixme: wait for current extraction 
  size_t num_new_insert_node = num_new_local_node - num_preserved_node;
  
  if (RunConfig::option_empty_feat == 0) {
    const DataIter<const IdType*> src_data_iter(new_insert_node_list_gpu->CPtr<IdType>(), _cache_ctx->_device_cache_data[_cache_ctx->_cpu_location_id], _cache_ctx->_dim);
    DataIter<FreeOffIter> dst_data_iter(FreeOffIter(nouse_offset->Ptr<IdType>()), _cache_ctx->_device_cache_data[_cache_ctx->_local_location_id], _cache_ctx->_dim);
    Combine(src_data_iter, dst_data_iter, num_new_insert_node, gpu_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);
  } else {
    const DataIter<MockSrcOffIter> src_data_iter(MockSrcOffIter(new_insert_node_list_gpu->Ptr<IdType>()), _cache_ctx->_device_cache_data[_cache_ctx->_cpu_location_id], _cache_ctx->_dim);
    DataIter<FreeOffIter> dst_data_iter(FreeOffIter(nouse_offset->Ptr<IdType>()), _cache_ctx->_device_cache_data[_cache_ctx->_local_location_id], _cache_ctx->_dim);
    Combine(src_data_iter, dst_data_iter, num_new_insert_node, gpu_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);
  }

  LOG(INFO) << "CollCacheManager: fix location and offset of local nodes in hash table";
  if (num_new_insert_node > 0) {
    SAM_CUDA_PREPARE_1D(num_new_insert_node);
    refresh_hash_table_local<><<<grid, block, 0, cu_stream>>>(
        _hash_table_location, _hash_table_offset, 
        new_insert_node_list_gpu->CPtr<IdType>(), num_new_insert_node, nouse_offset->CPtr<IdType>(), _local_location_id);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }
  new_insert_node_list_gpu = nullptr;
#ifdef DEAD_CODE
  if (num_new_insert_node > 0){
    auto new_insert_offset_list = Tensor::EmptyExternal(kI32, {num_new_insert_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
    SAM_CUDA_PREPARE_1D(num_new_insert_node);
    preserve_offset<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(new_insert_offset_list->Ptr<IdType>(), new_insert_node_list_gpu->CPtr<IdType>(), num_new_insert_node, _cache_ctx->_hash_table_offset);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));

    new_insert_offset_list = Tensor::CopyTo(new_insert_offset_list, CPU(), stream, "");
    auto nouse_offset_cpu = Tensor::CopyTo(nouse_offset, CPU(), stream, "");
    check_uniq(new_insert_offset_list->CPtr<IdType>(), num_new_insert_node);
    check_covers(nouse_offset_cpu->CPtr<IdType>(), num_nouse_offset, new_insert_offset_list->CPtr<IdType>(), num_new_insert_node);
    check_no_intersec(new_insert_offset_list->CPtr<IdType>(), num_new_insert_node, inuse_offset->CPtr<IdType>(), num_preserved_node);
  }
#endif

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new remote nodes";
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      size_t num_remote_nodes = per_src_size[dev_id];
      if (num_remote_nodes == 0) continue;
      TensorPtr remote_node_list_tensor = Tensor::CopyToExternal(node_list_of_src[dev_id], _cache_ctx->_eager_gpu_mem_allocator, gpu_ctx, stream);

      SAM_CUDA_PREPARE_1D(num_remote_nodes);
      init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
          _hash_table_location, _hash_table_offset, 
          _cache_ctx->_remote_hash_table_offset[dev_id], remote_node_list_tensor->CPtr<IdType>(), num_remote_nodes, dev_id);
      CUDA_CALL(cudaStreamSynchronize(cu_stream));
    }
  }

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(INFO) << "refresh done";

  _cache_ctx->_cache_nodes = num_new_local_node;
  _cache_ctx->_local_node_list_tensor = new_local_node_list_cpu;
}
#endif

void RefreshSession::refresh_after_solve_new(bool foreground) {
  auto _new_hash_table = _cache_ctx->_new_hash_table;
  Context gpu_ctx = _cache_ctx->_trainer_ctx;
  auto _local_location_id = _cache_ctx->_local_location_id;
  auto cu_stream = reinterpret_cast<cudaStream_t>(stream);

  // Step.1 Detect key for each source
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making key list for each source";
  TensorPtr key_list_of_each_src[9] = {nullptr};
  auto coll_cache = _cache_ctx->_coll_cache;
  TensorPtr block_access_advise_cpu = Tensor::CopyLine(_cache_ctx->_coll_cache->_block_access_advise, _local_location_id, CPU(CPU_CLIB_MALLOC_DEVICE), stream); // small
  CacheEntryManager::DetectKeysForAllSource(coll_cache->_nid_to_block, block_access_advise_cpu, _local_location_id, coll_cache->_block_density, RunConfig::num_total_item, key_list_of_each_src, RunConfig::num_device);
  TensorPtr old_rkeys = _new_hash_table->_remote_keys;
  TensorPtr new_rkeys = ConcatAllRemote(key_list_of_each_src, RunConfig::num_device, _local_location_id);

  size_t num_new_local_key = key_list_of_each_src[_local_location_id]->NumItem();
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new local node = " << num_new_local_key;
  CHECK(num_new_local_key <= _cache_ctx->_cache_space_capacity) << "cache space can not hold refreshed new cache";

  // Step.2 Detect new local keys to insert
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding preserved node and new insert nodes";
  TensorPtr _new_insert_keys;
  {
    auto _old_nid_to_block = _cache_ctx->_coll_cache->_old_nid_to_block->CPtr<IdType>();
    auto _old_block_placement = _cache_ctx->_coll_cache->_old_block_placement->CPtr<uint8_t>();
    _new_insert_keys = _new_hash_table->DetectKeysWithCond(key_list_of_each_src[_local_location_id]->CPtr<IdType>(), num_new_local_key, [_old_nid_to_block, _old_block_placement, _local_location_id](const IdType & key){
      return (_old_block_placement[_old_nid_to_block[key]] & (1 << _local_location_id)) == 0;
    }, num_new_local_key);
  }
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << " new hashtable find new local key done";
  size_t num_preserved_key = key_list_of_each_src[_local_location_id]->NumItem() - _new_insert_keys->NumItem();
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "preserved node = " << num_preserved_key;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new insert node = " << _new_insert_keys->NumItem();
  // CUDA_CALL(cudaStreamSynchronize(cu_stream));
  // preserved keys is not necessary when using new hashtable

  // Step.3 Detect old local keys to evict, and corresponding cache offset, then evict
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding evicted nodes";
  TensorPtr _new_evict_keys;
  {
    auto _nid_to_block = _cache_ctx->_coll_cache->_nid_to_block->CPtr<IdType>();
    auto _block_placement = _cache_ctx->_coll_cache->_block_placement->CPtr<uint8_t>();
    _new_evict_keys = _new_hash_table->DetectKeysWithCond(_new_hash_table->_cached_keys->CPtr<IdType>(), _new_hash_table->_cached_keys->Shape()[0], [_nid_to_block, _block_placement, _local_location_id](const IdType & key){
      return (_block_placement[_nid_to_block[key]] & (1 << _local_location_id)) == 0;
    }, _new_hash_table->_cached_keys->NumItem() - num_preserved_key);
  }
  size_t num_eviced_node = _new_evict_keys->NumItem();
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num evicted node = " << num_eviced_node << ", should equal to " << _new_hash_table->_cached_keys->Shape()[0] - (key_list_of_each_src[_local_location_id]->NumItem() - _new_insert_keys->NumItem());
  CHECK_EQ(num_eviced_node + num_preserved_key, _new_hash_table->_cached_keys->NumItem());

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "gathering unused offsets";
  if (num_eviced_node > 0) {
    _new_evict_keys = CopyTo1DReuse(__keys_buffer, _new_evict_keys, gpu_ctx, stream);
    auto _new_evict_offsets_gpu = Empty1DReuse(__offs_buffer, kI32, {num_eviced_node}, gpu_ctx);
    _new_hash_table->LookupOffset(_new_evict_keys, _new_evict_offsets_gpu, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    auto _new_evict_offsets_cpu = _new_evict_offsets_gpu->CopyTo(CPU(), stream, "");
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    _new_hash_table->ReturnOffset(_new_evict_offsets_cpu);
  }
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "num no use offset = " << _new_hash_table->_free_offsets->NumItem();
#ifdef COLL_HASH_VALID_LEGACY
  _new_hash_table->SortFreeOffsets();
#endif

  CHECK(num_eviced_node + key_list_of_each_src[_local_location_id]->NumItem() - _new_insert_keys->NumItem() == _new_hash_table->_cached_keys->NumItem());
  CHECK(num_eviced_node <= _new_hash_table->_free_offsets->NumItem());

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting nodes";

  _new_hash_table->Evict(_new_evict_keys, stream);
  CUDA_CALL(cudaStreamSynchronize(cu_stream));

  // new hashtable do not depends on remote hashtable to detech remote keys to evict
  // AnonymousBarrier::_refresh_instance->Wait();

  // Step.4 Evict remote keys
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting remote nodes";
  auto remote_keys_to_evict = _new_hash_table->DetectKeysWithCond(
      old_rkeys->CPtr<IdType>(),
      old_rkeys->NumItem(), 
      SelectEvictedRemoteKeys(
          _cache_ctx->_coll_cache->_old_nid_to_block,
          _cache_ctx->_coll_cache->_old_block_access_advise,
          _cache_ctx->_coll_cache->_nid_to_block,
          _cache_ctx->_coll_cache->_block_access_advise,
          nullptr, // _cache_ctx->_coll_cache->_block_placement,
          _local_location_id, _cache_ctx->_cpu_location_id
        ),
       old_rkeys->NumItem()
      // _new_hash_table->_hash_table->max_efficient_size - _new_hash_table->_cached_keys->NumItem()
    );
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "evicting remote nodes - key detected, now evict " << remote_keys_to_evict->NumItem();
  if (remote_keys_to_evict->NumItem() > 0) {
    remote_keys_to_evict = CopyTo1DReuse(__keys_buffer, remote_keys_to_evict, gpu_ctx, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
    _new_hash_table->EvictRemote(remote_keys_to_evict, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }

  // Step.5 wait for one lookup batch, eliminate cache read dependency
  size_t current_progress = _cache_ctx->progress.load();

  AnonymousBarrier::_refresh_instance->Wait();

  Timer t0;
  while (!foreground && _cache_ctx->progress.load() < current_progress + 1) {
    if (t0.PassedSec() >= 20) break;
  }

  // Step.6 Update cache content, insert new local keys to hashtable
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new local nodes";

  TensorPtr reserved_offset;
  if (_new_insert_keys->NumItem() > 0) {
    _new_insert_keys = CopyTo1DReuse(__keys_buffer, _new_insert_keys, gpu_ctx, low_pri_stream);
    reserved_offset = CopyTo1DReuse(__offs_buffer, _new_hash_table->ReserveOffsetFront(_new_insert_keys->NumItem()), gpu_ctx, low_pri_stream);
    const size_t local_combine_batch_size = 4092; // 4K ~ 0.2ms
    SWITCH_BOOL(RunConfig::option_empty_feat != 0, use_empty_feat, {
      for (size_t cur_batch_begin = 0; cur_batch_begin < _new_insert_keys->NumItem(); cur_batch_begin += local_combine_batch_size) {
        size_t cur_batch_size = (cur_batch_begin + local_combine_batch_size < _new_insert_keys->NumItem()) ? local_combine_batch_size : (_new_insert_keys->NumItem() - cur_batch_begin);
        IdxStoreDst<use_empty_feat> idx_store(_new_insert_keys->CPtr<IdType>() + cur_batch_begin, reserved_offset->Ptr<IdType>() + cur_batch_begin);
        DataIterPerLoc<decltype(idx_store)> data_iter(idx_store, _cache_ctx->_device_cache_data[_cache_ctx->_cpu_location_id], _cache_ctx->_device_cache_data[_cache_ctx->_local_location_id], _cache_ctx->_dim);
        Combine(data_iter, cur_batch_size, gpu_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, low_pri_stream, 20);
        usleep(3000);
      }
    });
  }

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new local nodes - combine done";
  if (_new_insert_keys->NumItem() > 0) {
    _new_hash_table->InsertWithLoc(_new_insert_keys, reserved_offset, _local_location_id, stream);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }

  AnonymousBarrier::_refresh_instance->Wait();
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new local nodes done";

  // Step.7 Update remote keys to hashtable
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "inserting new remote nodes";
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      if (key_list_of_each_src[dev_id] == nullptr || key_list_of_each_src[dev_id]->NumItem() == 0) continue;
      auto _old_access_view = coll_cache::TensorView<uint8_t>(_cache_ctx->_coll_cache->_old_block_access_advise)[_local_location_id];
      auto _remote_new_keys = _new_hash_table->DetectKeysWithCond(key_list_of_each_src[dev_id]->CPtr<IdType>(), key_list_of_each_src[dev_id]->NumItem(), [this, _old_access_view, dev_id](const IdType & key) mutable{
        auto block_id = _cache_ctx->_coll_cache->_old_nid_to_block->CPtr<IdType>()[key];
        return _old_access_view[block_id].ref() != dev_id;
      }, key_list_of_each_src[dev_id]->NumItem());
      if (_remote_new_keys->NumItem() > 0) {
        _remote_new_keys = CopyTo1DReuse(__keys_buffer, _remote_new_keys, gpu_ctx, stream);
        auto _remote_offsets = Empty1DReuse(__offs_buffer, kI32, _remote_new_keys->Shape(), gpu_ctx);
        _cache_ctx->_remote_new_hash_table[dev_id]->LookupOffset(_remote_new_keys, _remote_offsets, stream);
        _new_hash_table->InsertWithLoc(_remote_new_keys, _remote_offsets, dev_id, stream);
      }
    }
  }

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(INFO) << "refresh done";

  _new_hash_table->_cached_keys = key_list_of_each_src[_local_location_id];
  _new_hash_table->_remote_keys = new_rkeys;
}

void RefreshSession::refresh_after_solve_main(bool foreground) {
  auto cu_stream = reinterpret_cast<cudaStream_t>(stream);
#ifdef COLL_HASH_VALID_LEGACY
  refresh_after_solve_old(foreground);
#endif
  refresh_after_solve_new(foreground);

#ifdef COLL_HASH_VALID_LEGACY
  LOG(ERROR) << "comparing hashtable after refresh";
  _cache_ctx->compare_hashtable(stream);
  CUDA_CALL(cudaStreamSynchronize(cu_stream));
  LOG(ERROR) << "comparing hashtable after refresh - done";
#endif
}

TensorPtr RefreshSession::Empty1DReuse(TensorPtr& preserved_buffer, DataType dtype, std::vector<size_t> shape, Context ctx) {
  CHECK(shape.size() == 1);
  if (preserved_buffer && preserved_buffer->CanForceScale(dtype, shape)) {
    preserved_buffer->ForceScale(dtype, shape, ctx, "");
    return preserved_buffer;
  }
  preserved_buffer = Tensor::EmptyExternal(dtype, shape, _cache_ctx->_eager_gpu_mem_allocator, ctx, "");
  return preserved_buffer;
}
TensorPtr RefreshSession::CopyTo1DReuse(TensorPtr& preserved_buffer, TensorPtr src, Context ctx, StreamHandle stream) {
  auto ret = Empty1DReuse(preserved_buffer, src->Type(), src->Shape(), ctx);
  Device::Get(ctx)->CopyDataFromTo(src->Data(), 0, ret->MutableData(), 0, src->NumBytes(), src->Ctx(), ctx, stream);
  return ret;
}
void log_mem_usage(int dev_id, std::string msg) {
  if (dev_id == 0) {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    LOG(WARNING) << msg << ToReadableSize(total - free);
  }
};
void ExtractionThreadCtx::create_ctx(int dev_id, int num_sm, int priority) {
  log_mem_usage(dev_id, "before create ctx, mem is ");
  cu_ctx_ = cuda::create_ctx_with_sm_count(dev_id, num_sm);
  log_mem_usage(dev_id, "after create ctx, mem is ");
  check_current_ctx_is(cu_ctx_);
  if (priority == 0) {
    CUDA_CALL(cudaStreamCreate(&stream_));
  } else {
    CUDA_CALL(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
  }
  log_mem_usage(dev_id, "after create stream, mem is ");
}
void ExtractionThreadCtx::create_stream(int dev_id, int priority) {
  CUDA_CALL(cudaSetDevice(dev_id));
  log_mem_usage(dev_id, "before create stream, mem is ");
  if (priority == 0) {
    CUDA_CALL(cudaStreamCreate(&stream_));
  } else {
    CUDA_CALL(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
  }
  log_mem_usage(dev_id, "after create stream, mem is ");
}


#define CU_UUID_CONST static const
#define CU_CHAR(x) (char)((x) & 0xff)
// Define the symbol as exportable to other translation units, and
// initialize the value.  Inner set of parens is necessary because
// "bytes" array needs parens within the struct initializer, which
// also needs parens.  
#define CU_DEFINE_UUID(name, a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)          \
    CU_UUID_CONST CUuuid name =                                                \
    {                                                                          \
      {                                                                        \
        CU_CHAR(a), CU_CHAR((a) >> 8), CU_CHAR((a) >> 16), CU_CHAR((a) >> 24), \
        CU_CHAR(b), CU_CHAR((b) >> 8),                                         \
        CU_CHAR(c), CU_CHAR((c) >> 8),                                         \
        CU_CHAR(d0),                                                           \
        CU_CHAR(d1),                                                           \
        CU_CHAR(d2),                                                           \
        CU_CHAR(d3),                                                           \
        CU_CHAR(d4),                                                           \
        CU_CHAR(d5),                                                           \
        CU_CHAR(d6),                                                           \
        CU_CHAR(d7)                                                            \
      }                                                                        \
    }


#define ASSERT_GPU_ERROR(cmd)\
{\
    CUresult error = cmd;\
    if (error == CUDA_ERROR_DEINITIALIZED) {\
        fprintf(stderr, "[WARN] cuda result %d: cuda driver is shutting down at %s:%d\n", error, __FILE__, __LINE__); \
    } else if (error != CUDA_SUCCESS) {\
        const char* str;\
        cuGetErrorString(error, &str);\
        std::string err_str(str);\
        fprintf(stderr, "[ERR] cuda error %d: %s at %s:%d\n", error, str, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

CU_DEFINE_UUID(CU_ETID_SmDisableMask,
    0x8b7e90eb, 0x8cf2, 0x4a00, 0xb1, 0xd1, 0x08, 0xaa, 0x53, 0x55, 0x90, 0xdb);


void set_stream_sm_mask(CUstream s, unsigned int maskUpper, unsigned int maskLower) {
  const void* exportTable;
  ASSERT_GPU_ERROR(cuGetExportTable(&exportTable, &CU_ETID_SmDisableMask));
  CUresult (*set_mask)(CUstream, unsigned int, unsigned int) = (CUresult (*)(CUstream, unsigned int, unsigned int))(*(unsigned long int*)((uint8_t*)(exportTable) + (8*1)));
  ASSERT_GPU_ERROR(set_mask(s, maskUpper, maskLower));
  // printf("set mask: %08x, %08x\n", maskUpper, maskLower);
}

void ExtractionThreadCtx::create_stream_sm_mask_v1(int dev_id, uint64_t mask, int priority) {
  CUDA_CALL(cudaSetDevice(dev_id));
  log_mem_usage(dev_id, "before create stream, mem is ");
  if (priority == 0) {
    CUDA_CALL(cudaStreamCreate(&stream_));
  } else {
    CUDA_CALL(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
  }
  set_stream_sm_mask(stream_, mask >> 32, mask & 0x0ffffffff);
  log_mem_usage(dev_id, "after create stream, mem is ");
}
} // namespace coll_cache_lib
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

#define SWITCH_TYPE(type, Type, ...)      \
  switch(type) {                     \
    case kF32: { typedef float   Type; { __VA_ARGS__ }; break; } \
    case kF64: { typedef double  Type; { __VA_ARGS__ }; break; } \
    case kF16: { typedef short   Type; { __VA_ARGS__ }; break; } \
    case kU8:  { typedef uint8_t Type; { __VA_ARGS__ }; break; } \
    case kI32: { typedef int32_t Type; { __VA_ARGS__ }; break; } \
    case kI64: { typedef int64_t Type; { __VA_ARGS__ }; break; } \
    default: CHECK(false);           \
  }


#define SAM_CUDA_PREPARE_1D(num_item) \
  const size_t num_tiles = RoundUpDiv((num_item), Constant::kCudaTileSize); \
  const dim3 grid(num_tiles); \
  const dim3 block(Constant::kCudaBlockSize);


namespace coll_cache_lib {

using namespace common;
// per-gpu cache handler

namespace {

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
  __host__ __device__ T* operator[](const size_t & idx) const {
    const size_t src_offset = _offset_iter[idx];
    const int location = _hash_table_location[src_offset];
    const auto _remote_raw_data = (T*)_device_cache_data[location];
    const auto offset = _hash_table_offset[src_offset];
    return _remote_raw_data + offset * dim;
  }
};
template<typename OffsetIter_T>
struct DataIter {
  OffsetIter_T offset_iter;
  void* output;
  size_t dim;
  DataIter() {}
  DataIter(OffsetIter_T offset_iter, void* output, size_t dim) : 
    offset_iter(offset_iter), output(output), dim(dim) {}
  DataIter(OffsetIter_T offset_iter, const void* output, size_t dim) : 
    offset_iter(offset_iter), output(const_cast<void*>(output)), dim(dim) {}
  template<typename T>
  __host__ __device__ T* operator[](const size_t & idx) {
    size_t offset = offset_iter[idx];
    return ((T*)output) + offset * dim;
  }
  template<typename T>
  __host__ __device__ const T* operator[](const size_t & idx) const {
    size_t offset = offset_iter[idx];
    return ((T*)output) + offset * dim;
  }
};

template <typename T, typename SrcDataIter_T, typename DstDataIter_T>
__global__ void extract_data(const SrcDataIter_T full_src, DstDataIter_T dst_index,
                             const size_t num_node,
                             size_t dim) {
  size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * gridDim.x;

  while (i < num_node) {
    size_t col = threadIdx.x;
    T* dst = dst_index.template operator[]<T>(i);
    const T* src = full_src.template operator[]<T>(i);
    while (col < dim) {
      dst[col] = src[col];
      col += blockDim.x;
    }
    i += stride;
  }
}

template<int NUM_LINK, typename SrcDataIter_T, typename DstDataIter_T>
struct ExtractConcurrentParam {
  SrcDataIter_T full_src_array[NUM_LINK];
  DstDataIter_T dst_index_array[NUM_LINK];
  IdType num_node_array[NUM_LINK];
  IdType block_num_prefix_sum[NUM_LINK + 1];
  const IdType* link_mapping;
  const IdType* sub_block_mappling;
  size_t dim;
};

template <int NUM_LINK, typename T, typename SrcDataIter_T, typename DstDataIter_T>
__global__ void extract_data_concurrent(ExtractConcurrentParam<NUM_LINK, SrcDataIter_T, DstDataIter_T> packed_param) {
  // block -> which link
  // block -> local block idx in this link
  // block -> num block of this link
  const IdType link_idx = packed_param.link_mapping[blockIdx.x];
  const IdType local_block_idx_x = packed_param.sub_block_mappling[blockIdx.x];
  const IdType local_grid_dim_x = packed_param.block_num_prefix_sum[link_idx + 1] - packed_param.block_num_prefix_sum[link_idx];
  size_t i = local_block_idx_x * blockDim.y + threadIdx.y;
  const size_t stride = blockDim.y * local_grid_dim_x;


  const IdType num_node = packed_param.num_node_array[link_idx];
  const auto full_src = packed_param.full_src_array[link_idx];
  auto dst_index = packed_param.dst_index_array[link_idx];

  // if ((packed_param.num_node_array[0] % 20) == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
  //   printf("Block[%d]/[%d], link[%d], Local Block idx=%d, local_grid_dim=%d, block duty size=%d, stride=%d\n",
  //     blockIdx.x, gridDim.x, link_idx, local_block_idx_x, local_grid_dim_x, num_node, stride);
  // }

  // if ((packed_param.num_node_array[0] % 20) == 0 && threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("\n");
  // }

  while (i < num_node) {
    size_t col = threadIdx.x;
    T* dst = dst_index.template operator[]<T>(i);
    const T* src = full_src.template operator[]<T>(i);
    while (col < packed_param.dim) {
      dst[col] = src[col];
      col += blockDim.x;
    }
    i += stride;
  }
}

struct LocationIter {
  SrcKey* src_key;
  LocationIter() {}
  LocationIter(SrcKey* src_key) : src_key(src_key) {}
  LocationIter(const SrcKey* src_key) : src_key(const_cast<SrcKey*>(src_key)) {}
  __host__ __device__ int & operator[](const size_t & idx) { return src_key[idx]._location_id; }
  __host__ __device__ const int & operator[](const size_t & idx) const { return src_key[idx]._location_id; }
};
struct SrcOffIter {
  DstVal* dst_val;
  SrcOffIter() {}
  SrcOffIter(DstVal* dst_val) : dst_val(dst_val) {}
  SrcOffIter(const DstVal* dst_val) : dst_val(const_cast<DstVal*>(dst_val)) {}
  __host__ __device__ IdType & operator[](const size_t & idx) { return dst_val[idx]._src_offset; }
  __host__ __device__ const IdType & operator[](const size_t & idx) const { return dst_val[idx]._src_offset; }
};
struct FreeOffIter {
  IdType* off_list;
  FreeOffIter() {}
  FreeOffIter(IdType* off_list) : off_list(off_list) {}
  FreeOffIter(const IdType* off_list) : off_list(const_cast<IdType*>(off_list)) {}
  __host__ __device__ IdType & operator[](const size_t & idx) { return off_list[idx]; }
  __host__ __device__ const IdType & operator[](const size_t & idx) const { return off_list[idx]; }
};

struct DstOffIter {
  DstVal* dst_val;
  DstOffIter() {}
  DstOffIter(DstVal* dst_val) : dst_val(dst_val) {}
  DstOffIter(const DstVal* dst_val) : dst_val(const_cast<DstVal*>(dst_val)) {}
  __host__ __device__ IdType & operator[](const size_t & idx) { return dst_val[idx]._dst_offset; }
  __host__ __device__ const IdType & operator[](const size_t & idx) const { return dst_val[idx]._dst_offset; }
};
struct DirectOffIter {
  __host__ __device__ size_t operator[](const size_t & idx) const { return idx; }
};
struct MockOffIter {
  size_t empty_feat;
  MockOffIter() { empty_feat = RunConfig::option_empty_feat; }
  __host__ __device__ size_t operator[](const size_t & idx) const { return idx % (1 << empty_feat); }
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


template <size_t BLOCK_SIZE=Constant::kCudaBlockSize, size_t TILE_SIZE=Constant::kCudaTileSize, 
    typename LocationIter_T>
__global__ void find_boundary(
    LocationIter_T location_iter, const size_t len,
    IdType* boundary_list) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t src_offset = block_start + threadIdx.x; src_offset < block_end;
       src_offset += BLOCK_SIZE) {
    if (src_offset < len) {
      if (src_offset == len-1 || location_iter[src_offset] != location_iter[src_offset+1]) {
        boundary_list[location_iter[src_offset]+1] = src_offset+1;
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
#endif


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

}  // namespace

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

void ExtractSession::GetMissCacheIndex(
    SrcKey* & output_src_index, DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  if (num_nodes == 0) return;
  if (output_src_index_handle == nullptr || output_src_index_handle->nbytes() < num_nodes * sizeof(SrcKey)) {
    output_src_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
    output_dst_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(DstVal));
    output_src_index_alter_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
    output_dst_index_alter_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(DstVal));
  }

  output_src_index = output_src_index_handle->ptr<SrcKey>();
  output_dst_index = output_dst_index_handle->ptr<DstVal>();


  SrcKey * output_src_index_alter = output_src_index_alter_handle->ptr<SrcKey>();
  DstVal * output_dst_index_alter = output_dst_index_alter_handle->ptr<DstVal>();

  const size_t num_tiles = RoundUpDiv(num_nodes, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - getting miss/hit index...";
  Timer t0;
  LocationIter location_iter(output_src_index);
  SrcOffIter src_offset_iter(output_dst_index);
  DstOffIter dst_offset_iter(output_dst_index);
  get_miss_cache_index<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
    location_iter, src_offset_iter, dst_offset_iter, nodes, num_nodes, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset);
  device->StreamSync(_cache_ctx->_trainer_ctx, stream);
  // std::cout << "coll get index "<< t0.Passed() << "\n";
  
  Timer t1;
  cub::DoubleBuffer<int> keys(reinterpret_cast<int*>(output_src_index), reinterpret_cast<int*>(output_src_index_alter));
  cub::DoubleBuffer<Id64Type> vals(reinterpret_cast<Id64Type*>(output_dst_index), reinterpret_cast<Id64Type*>(output_dst_index_alter));

  size_t workspace_bytes;
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - sorting according to group...";
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      nullptr, workspace_bytes, keys, vals, num_nodes, 0, sizeof(SrcKey) * 8,
      cu_stream));

  if (workspace_handle == nullptr || workspace_handle->nbytes() < workspace_bytes) {
    workspace_handle = _cache_ctx->_gpu_mem_allocator(workspace_bytes);
  }
  void *workspace = workspace_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, num_nodes, 0, sizeof(SrcKey) * 8,
      cu_stream));
  device->StreamSync(_cache_ctx->_trainer_ctx, stream);
  LOG(DEBUG) << "CollCacheManager: GetMissCacheIndex - sorting according to group - done...";
  // std::cout << "coll sort index "<< t1.Passed() << "\n";

  Timer t2;
  if (reinterpret_cast<SrcKey*>(keys.Current()) != output_src_index) {
    output_src_index = reinterpret_cast<SrcKey*>(keys.Current());
    output_dst_index = reinterpret_cast<DstVal*>(vals.Current());
    // output_src_index_handle = output_src_index_alter_handle;
    // output_dst_index_handle = output_dst_index_alter_handle;
  }

  // std::cout << "coll free workspace "<< t2.Passed() << "\n";
}

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
  get_location<Constant::kCudaBlockSize, Constant::kCudaTileSize><<<grid, block, 0, cu_stream>>>(
    location_iter, nodes, num_nodes, _cache_ctx->_hash_table_location);
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

void ExtractSession::SplitGroup(const SrcKey * src_index, const size_t len, IdType * & group_offset, StreamHandle stream){
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  const size_t num_tiles = RoundUpDiv(len, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  Timer t0;
  group_offset = this->_group_offset;
  // group_offset = (IdType*)Device::Get(cpu_ctx)->AllocWorkspace(cpu_ctx, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  std::memset(group_offset, 0, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  group_offset[_cache_ctx->_num_location] = len;
  if (len == 0) return;
  LOG(DEBUG) << "CollCache: SplitGroup: legacy finding offset...";
  const LocationIter loc_iter(src_index);
  find_boundary<><<<grid, block, 0, cu_stream>>>(loc_iter, len, group_offset);
  device->StreamSync(_cache_ctx->_trainer_ctx, stream);
  LOG(DEBUG) << "CollCache: SplitGroup: legacy fixing offset...";
  /** Old implementation is buggy */
  // for (int i = cache_ctx->_num_location - 1; i > 0; i--) {
  //   if (group_offset[i] < group_offset[i-1]) {
  //     group_offset[i] = group_offset[i+1];
  //   }
  // }
  for (int i = 1; i < _cache_ctx->_num_location; i++) {
    if (group_offset[i+1] == 0) {
      group_offset[i+1] = group_offset[i];
    }
  }
  // std::cout << "coll split group "<< t0.Passed() << "\n";
  LOG(DEBUG) << "CollCache: SplitGroup: legacy fixing done...";
}

namespace {
template <typename SrcDataIter_T, typename DstDataIter_T>
void Combine(const SrcDataIter_T src_data_iter, DstDataIter_T dst_data_iter,
    const size_t num_node, Context _trainer_ctx, DataType _dtype, IdType _dim, StreamHandle stream, IdType limit_block=0, bool async=false);
}

void ExtractSession::CombineOneGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block, bool async) {
  const DataIter<const SrcOffIter> src_data_iter(SrcOffIter(dst_index), src_data, _cache_ctx->_dim);
  DataIter<const DstOffIter>       dst_data_iter(DstOffIter(dst_index), output, _cache_ctx->_dim);
  Combine<>(src_data_iter, dst_data_iter, num_node, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream, limit_block, async);
}

template<int NUM_LINK>
void ExtractSession::CombineConcurrent(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream) {
  CHECK(NUM_LINK == RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size());
  ExtractConcurrentParam<NUM_LINK, DataIter<SrcOffIter>, DataIter<DstOffIter>> param;
  IdType total_required_num_sm = 0;
  TensorPtr link_mapping = Tensor::Empty(kI32, {108}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  TensorPtr sub_block_mappling = Tensor::Empty(kI32, {108}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  IdType total_num_node = 0;
  for (int i = 0; i < NUM_LINK; i++) {
    CHECK(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i].size() == 1);
    int dev_id = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i][0];
    int num_sm = RunConfig::coll_cache_link_desc.link_sm[_cache_ctx->_local_location_id][i];
    param.full_src_array[i] = DataIter<SrcOffIter>(SrcOffIter(dst_index + group_offset[dev_id]), _cache_ctx->_device_cache_data[dev_id], _cache_ctx->_dim);
    param.dst_index_array[i] = DataIter<DstOffIter>(DstOffIter(dst_index + group_offset[dev_id]), output, _cache_ctx->_dim);
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

template<int NUM_LINK>
void ExtractSession::CombineFused(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream) {
  dim3 block(1024, 1);
  while (static_cast<size_t>(block.x) >= 2 * _cache_ctx->_dim) {
    block.x /= 2;
    block.y *= 2;
  }

  CHECK(NUM_LINK == RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size());
  ExtractConcurrentParam<NUM_LINK, DataIter<SrcOffIter>, DataIter<DstOffIter>> param;
  IdType total_required_num_block = 0;
  TensorPtr link_mapping = Tensor::Empty(kI32, {RoundUpDiv(group_offset[this->_cache_ctx->_num_location] - group_offset[0], block.y * 4) * 2}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  TensorPtr sub_block_mappling = Tensor::Empty(kI32, {RoundUpDiv(group_offset[this->_cache_ctx->_num_location] - group_offset[0], block.y * 4) * 2}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  IdType total_num_node = 0;
  for (int i = 0; i < NUM_LINK; i++) {
    CHECK(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i].size() == 1);
    int dev_id = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id][i][0];
    param.full_src_array[i] = DataIter<SrcOffIter>(SrcOffIter(dst_index + group_offset[dev_id]), _cache_ctx->_device_cache_data[dev_id], _cache_ctx->_dim);
    param.dst_index_array[i] = DataIter<DstOffIter>(DstOffIter(dst_index + group_offset[dev_id]), output, _cache_ctx->_dim);
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
template <typename SrcDataIter_T, typename DstDataIter_T>
void Combine(const SrcDataIter_T src_data_iter, DstDataIter_T dst_data_iter,
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
      extract_data<type><<<grid, block, 0, cu_stream>>>(
          src_data_iter, dst_data_iter, num_node, _dim);
  });

  if (async == false) {
    device->StreamSync(_trainer_ctx, stream);
  }
}
}

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

void ExtractSession::ExtractFeat(const IdType* nodes, const size_t num_nodes,
                  void* output, StreamHandle stream, uint64_t task_key) {
#ifdef DEAD_CODE
  if (IsDirectMapping()) {
    // fast path
    // direct mapping from node id to freature, no need to go through hashtable
    LOG(DEBUG) << "CollCache: ExtractFeat: Direct mapping, going fast path... ";
    Timer t0;
    const DataIter<const IdType*> src_data_iter(nodes, _device_cache_data[0], _dim);
    DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), output, _dim);
    Combine(src_data_iter, dst_data_iter, num_nodes, _trainer_ctx, _dtype, _dim, stream);
    double combine_time = t0.Passed();
    if (task_key != 0xffffffffffffffff) {
      Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      if (_cpu_location_id == -1) {
        // full cache
        Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_time);
      } else {
        // no cache
        Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
        Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_time);
      }
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_cache_time);
      Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
    }
  } else if (IsLegacy()) {
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
  } else 
#endif
  if (RunConfig::coll_cache_no_group != common::kAlwaysGroup) {
    // CHECK(false) << "Multi source extraction is not supported now";
    auto trainer_gpu_device = Device::Get(_cache_ctx->_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    // Timer t0;
    // double get_index_time = t0.Passed();
    Timer t0;
    if (RunConfig::coll_cache_no_group == common::kOrderedNoGroup) {
      IdType* sorted_nodes;
      SortByLocation(sorted_nodes, nodes, num_nodes, stream);
      nodes = sorted_nodes;
    }
    double get_index_time = t0.Passed();
    Timer t1;
    CombineNoGroup(nodes, num_nodes, output, _cache_ctx->_trainer_ctx, _cache_ctx->_dtype, _cache_ctx->_dim, stream);
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
    SrcKey * src_index = nullptr;
    DstVal * dst_index = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, get miss cache index... ";
    Timer t0;
    GetMissCacheIndex(src_index, dst_index, nodes, num_nodes, stream);
    // std::cout << "Get Idx " << t0.Passed() << "\n";
    IdType * group_offset = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, splitting group... ";
    SplitGroup(src_index, num_nodes, group_offset, stream);
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

      auto call_combine = [src_index, group_offset, dst_index, nodes, this, output, num_nodes](int location_id, StreamHandle stream){
        if (num_nodes == 0) return;
        CombineOneGroup(src_index + group_offset[location_id], 
                        dst_index + group_offset[location_id], 
                        nodes + group_offset[location_id], 
                        group_offset[location_id+1] - group_offset[location_id], 
                        _cache_ctx->_device_cache_data[location_id], output, stream, 0, true);
        CUDA_CALL(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
      };
      // launch cpu extraction
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->forward_one_step([&combine_times, call_combine, loc_id = _cache_ctx->_cpu_location_id](cudaStream_t cu_s){
        Timer t_cpu;
        call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
        combine_times[0] = t_cpu.Passed();
      });
      // launch local extraction
      auto & link_src = RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id];
      auto num_link = link_src.size();
      Timer t_local;
      this->_extract_ctx[_cache_ctx->_local_location_id]->forward_one_step([call_combine, loc_id = _cache_ctx->_local_location_id](cudaStream_t cu_s){
        call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
      });
      this->_extract_ctx[_cache_ctx->_local_location_id]->wait_one_step();
      combine_times[2] = t_local.Passed();
      // launch remote extraction
      Timer t_remote;
      for (int i = 0; i < num_link; i++) {
        CHECK(link_src[i].size() == 1);
        int loc_id = link_src[i][0];
        if (group_offset[loc_id+1] - group_offset[loc_id] == 0) continue;
        this->_extract_ctx[loc_id]->forward_one_step([call_combine, loc_id](cudaStream_t cu_s){
          call_combine(loc_id, reinterpret_cast<StreamHandle>(cu_s));
        });
      }
      for (int i = 0; i < num_link; i++) {
        int loc_id = link_src[i][0];
        if (group_offset[loc_id+1] - group_offset[loc_id] == 0) continue;
        this->_extract_ctx[loc_id]->wait_one_step();
      }
      combine_times[1] = t_remote.Passed();
      this->_extract_ctx[_cache_ctx->_cpu_location_id]->wait_one_step();
    } else {
      // cpu first, then concurrent remote, then local
      auto call_combine = [src_index, group_offset, dst_index, nodes, this, output](int location_id, StreamHandle stream){
        CombineOneGroup(src_index + group_offset[location_id], 
                        dst_index + group_offset[location_id], 
                        nodes + group_offset[location_id], 
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
            case 1: CombineFused<1>(src_index, dst_index, group_offset, output, stream); break;
            case 2: CombineFused<2>(src_index, dst_index, group_offset, output, stream); break;
            case 3: CombineFused<3>(src_index, dst_index, group_offset, output, stream); break;
            case 4: CombineFused<4>(src_index, dst_index, group_offset, output, stream); break;
            case 5: CombineFused<5>(src_index, dst_index, group_offset, output, stream); break;
            case 6: CombineFused<6>(src_index, dst_index, group_offset, output, stream); break;
            case 7: CombineFused<7>(src_index, dst_index, group_offset, output, stream); break;
            default: CHECK(false);
          }
        } else if (RunConfig::concurrent_link_impl == kFusedLimitNumBlock) {
          switch(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size()) {
            case 1: CombineConcurrent<1>(src_index, dst_index, group_offset, output, stream); break;
            case 2: CombineConcurrent<2>(src_index, dst_index, group_offset, output, stream); break;
            case 3: CombineConcurrent<3>(src_index, dst_index, group_offset, output, stream); break;
            case 4: CombineConcurrent<4>(src_index, dst_index, group_offset, output, stream); break;
            case 5: CombineConcurrent<5>(src_index, dst_index, group_offset, output, stream); break;
            case 6: CombineConcurrent<6>(src_index, dst_index, group_offset, output, stream); break;
            case 7: CombineConcurrent<7>(src_index, dst_index, group_offset, output, stream); break;
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
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // _cache_ctx->_coll_cache->_profiler->LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
    // cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
  } else {
    // CHECK(false) << "Multi source extraction is not supported now";
    auto trainer_gpu_device = Device::Get(_cache_ctx->_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    SrcKey * src_index = nullptr;
    DstVal * dst_index = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, get miss cache index... ";
    Timer t0;
    GetMissCacheIndex(src_index, dst_index, nodes, num_nodes, stream);
    // std::cout << "Get Idx " << t0.Passed() << "\n";
    Timer t1;
    IdType * group_offset = nullptr;
    LOG(DEBUG) << "CollCache: ExtractFeat: coll, splitting group... ";
    SplitGroup(src_index, num_nodes, group_offset, stream);
    double get_index_time = t0.Passed();
    
    // std::cout << "Split GrOup " <<t1.Passed() << "\n";
    double combine_times[3] = {0, 0, 0};
    // cpu first, then remote, then local 

    auto call_combine = [src_index, group_offset, dst_index, nodes, this, output, stream, num_nodes](int location_id){
      if (num_nodes == 0) return;
      CombineOneGroup(src_index + group_offset[location_id], 
                      dst_index + group_offset[location_id], 
                      nodes + group_offset[location_id], 
                      group_offset[location_id+1] - group_offset[location_id], 
                      _cache_ctx->_device_cache_data[location_id], output, stream);
    };
    _cache_ctx->_barrier->Wait();
    Timer t_cpu;
    call_combine(_cache_ctx->_cpu_location_id);
    if (group_offset[_cache_ctx->_cpu_location_id+1] - group_offset[_cache_ctx->_cpu_location_id] != 0) {
      _cache_ctx->_barrier->Wait();
    }
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

CollCacheManager CollCacheManager::BuildNoCache(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  StreamHandle stream) {
  CollCacheManager cm(trainer_ctx, dtype, dim, 0);
  cm._cpu_location_id = 0;

  Timer t;

  cm._cache_nbytes = 0;

  // auto cpu_device = Device::Get(_extractor_ctx);
  auto trainer_gpu_device = Device::Get(trainer_ctx);

  // _cpu_hashtable = static_cast<IdType *>(
  //     cpu_device->AllocDataSpace(_extractor_ctx, sizeof(IdType) * _num_nodes));
  cm._device_cache_data.resize(1);
  cm._device_cache_data[0] = cpu_src_data;
  cm._hash_table_location = nullptr;
  cm._hash_table_offset = nullptr;

  LOG(INFO) << "Collaborative GPU cache (policy: " << "no cache"
            << ") | " << t.Passed()
            << " secs )";
  std::cout << "test_result:init:cache_nbytes=" << cm._cache_nbytes << "\n";
  return cm;
}

CollCacheManager CollCacheManager::BuildFullCache(Context trainer_ctx,
                  void* cpu_src_data, DataType dtype, size_t dim,
                  size_t num_total_nodes,
                  StreamHandle stream) {
  CollCacheManager cm(trainer_ctx, dtype, dim, 0);
  cm._cpu_location_id = -1;

  Timer t;

  cm._cache_nbytes = GetTensorBytes(cm._dtype, {num_total_nodes, cm._dim});

  auto trainer_gpu_device = Device::Get(trainer_ctx);

  void* local_cache = trainer_gpu_device->AllocDataSpace(trainer_ctx, cm._cache_nbytes);
  trainer_gpu_device->CopyDataFromTo(cpu_src_data, 0, local_cache, 0, cm._cache_nbytes, CPU(), trainer_ctx, stream);
  trainer_gpu_device->StreamSync(trainer_ctx, stream);

  cm._device_cache_data.resize(1);
  cm._device_cache_data[0] = local_cache;
  cm._hash_table_location = nullptr;
  cm._hash_table_offset = nullptr;

  LOG(INFO) << "Collaborative GPU cache (policy: " << "full cache"
            << ") " << num_total_nodes << " nodes ( "
            << ToReadableSize(cm._cache_nbytes) << " | " << t.Passed()
            << " secs )";
  std::cout << "test_result:init:cache_nbytes=" << cm._cache_nbytes << "\n";
  return cm;
}

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
  auto node_list_buffer_handle = _eager_gpu_mem_allocator(num_total_nodes * sizeof(IdType));
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
      const DataIter<MockOffIter> src_data_iter(MockOffIter(), cpu_data, dim);
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

void CacheContext::build_with_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream) {
  auto hash_table_offset_list = DevicePointerExchanger(_barrier, Constant::kCollCacheHashTableOffsetPtrShmName);
  auto device_cache_data_list = DevicePointerExchanger(_barrier, Constant::kCollCacheDeviceCacheDataPtrShmName);
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

  _remote_hash_table_offset.resize(_num_location, nullptr);
  _remote_hash_table_location.resize(_num_location, nullptr);
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
  auto node_list_buffer_handle = _eager_gpu_mem_allocator(num_total_nodes * sizeof(IdType));
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
    _cache_space_capacity = num_cached_nodes + num_total_nodes * 0.0001;
    _cache_nbytes = GetTensorBytes(_dtype, {_cache_space_capacity, _dim});
    _cache_nodes = num_cached_nodes;
    _device_cache_data_local_handle = _eager_gpu_mem_allocator(_cache_nbytes);
    _device_cache_data[_local_location_id] = _device_cache_data_local_handle->ptr();
    if (num_cached_nodes > 0) {
      if (RunConfig::option_empty_feat == 0) {
        const DataIter<const IdType*> src_data_iter(cache_node_list, cpu_data, dim);
        DataIter<DirectOffIter> dst_data_iter(DirectOffIter(), _device_cache_data[_local_location_id], dim);
        Combine(src_data_iter, dst_data_iter, num_cached_nodes, gpu_ctx, dtype, dim, stream);
      } else {
        const DataIter<MockOffIter> src_data_iter(MockOffIter(), cpu_data, dim);
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
  }

  LOG(INFO) << "CollCacheManager: waiting for remotes, here is " << _local_location_id;

  // wait until all device's hashtable is ready
  hash_table_location_list.signin(_local_location_id, _hash_table_location);
  hash_table_offset_list.signin(_local_location_id, _hash_table_offset);
  if (num_cached_nodes > 0) {
    device_cache_data_list.signin(_local_location_id, _device_cache_data[_local_location_id]);
  }

  /**
   * 3. get hashtable entry, cache data from remote devices
   */
  for (auto & link : RunConfig::coll_cache_link_desc.link_src[_local_location_id]) {
    for (auto dev_id : link) {
      LOG(ERROR) << "Device " << _local_location_id << " init p2p of link " << dev_id;
      IdType * remote_node_list = node_list_buffer;
      size_t num_remote_nodes;
      cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, remote_node_list, num_remote_nodes, dev_id, _eager_gpu_mem_allocator, stream);
      if (!RunConfig::cross_process) {
        auto cuda_err = cudaDeviceEnablePeerAccess(dev_id, 0);
        if (cuda_err != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CALL(cuda_err);
        }
      }
      _device_cache_data[dev_id] = device_cache_data_list.extract(dev_id);
      _remote_hash_table_location[dev_id] = (HashTableEntryLocation *)hash_table_location_list.extract(dev_id);
      _remote_hash_table_offset[dev_id] = (HashTableEntryOffset * )hash_table_offset_list.extract(dev_id);

      if (num_remote_nodes == 0) continue;
      SAM_CUDA_PREPARE_1D(num_remote_nodes);
      init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
          _hash_table_location, _hash_table_offset, 
          _remote_hash_table_offset[dev_id], remote_node_list, num_remote_nodes, dev_id);
      gpu_device->StreamSync(gpu_ctx, stream);
      // hash_table_offset_list.close((void*)_remote_hash_table_offset[dev_id]);
      // CUDA_CALL(cudaIpcCloseMemHandle((void*)_remote_hash_table_offset[dev_id]))
    }
  }
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

  if (_coll_cache->_block_access_advise) {
    return build_with_advise(location_id, coll_cache_ptr, cpu_data, dtype, dim,
                             gpu_ctx, cache_percentage, stream);
  } else {
    return build_without_advise(location_id, coll_cache_ptr, cpu_data, dtype, dim,
                                gpu_ctx, cache_percentage, stream);
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

ExtractSession::ExtractSession(std::shared_ptr<CacheContext> cache_ctx) : _cache_ctx(cache_ctx) {
  auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  _group_offset = (IdType*)Device::Get(cpu_ctx)->AllocWorkspace(cpu_ctx, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  // _cache_ctx->ctx_injector_ = [](){};
  if (RunConfig::concurrent_link_impl == common::kMPS) {
    this->_extract_threads.resize(_cache_ctx->_num_location);
    this->_extract_ctx.resize(_cache_ctx->_num_location);
    auto gpu_ctx = cache_ctx->_trainer_ctx;
    auto & link_desc = RunConfig::coll_cache_link_desc;
    auto & link_src = link_desc.link_src[cache_ctx->_local_location_id];
    int _local_location_id = cache_ctx->_local_location_id;

    auto ctx_creation_lambda = [gpu_ctx, this](int num_sm, int src_location_id){
      auto ext_ctx_ptr = std::make_shared<ExtractionThreadCtx>();
      this->_extract_ctx[src_location_id] = ext_ctx_ptr;
      this->_extract_threads[src_location_id] = std::thread([ext_ctx_ptr](){
        ext_ctx_ptr->thread_func();
      });
      ext_ctx_ptr->forward_one_step([ext_ctx_ptr, dev_id=gpu_ctx.device_id, num_sm](cudaStream_t s){
        ext_ctx_ptr->cu_ctx_ = cuda::create_ctx_with_sm_count(dev_id, num_sm);
        check_current_ctx_is(ext_ctx_ptr->cu_ctx_);
        CUDA_CALL(cudaStreamCreate(&ext_ctx_ptr->stream_));
      });
      ext_ctx_ptr->wait_one_step();
    };

    // check_primary_ctx_active(gpu_ctx.device_id);
    cuda::check_have_affinity_support(gpu_ctx.device_id);
    int num_link = link_src.size();
    for (int link_id = 0; link_id < num_link; link_id++) {
      CHECK(link_src[link_id].size() == 1);
      ctx_creation_lambda(link_desc.link_sm[_local_location_id][link_id], link_src[link_id][0]);
    }
    ctx_creation_lambda(link_desc.local_sm[_local_location_id], gpu_ctx.device_id);
    ctx_creation_lambda(link_desc.cpu_sm[_local_location_id], _cache_ctx->_cpu_location_id);
    // _cache_ctx->ctx_injector_ = [this, _local_location_id](){
    //   CU_CALL(cuCtxSetCurrent(_extract_ctx[_local_location_id]->cu_ctx_));
    // };
  } else if (RunConfig::concurrent_link_impl != common::kNoConcurrentLink) {
    _concurrent_stream_array.resize(RunConfig::num_device - 1);
    for (auto & stream : _concurrent_stream_array) {
      cudaStream_t & cu_s = reinterpret_cast<cudaStream_t &>(stream);
      CUDA_CALL(cudaStreamCreate(&cu_s));
    }
  }
}

void ExtractionThreadCtx::thread_func() {
  if (this->cu_ctx_ != nullptr) {
    CU_CALL(cuCtxSetCurrent(this->cu_ctx_));
  }
  while (true) {
    int local_done_steps = done_steps.load();
    while (todo_steps.load() == local_done_steps) {}
    func_(stream_);
    done_steps.fetch_add(1);
  }
}
void ExtractionThreadCtx::forward_one_step(std::function<void(cudaStream_t)> new_func) {
  {
    func_ = new_func;
    todo_steps.fetch_add(1);
  }
}
void ExtractionThreadCtx::wait_one_step() {
  int local_todo_steps = todo_steps.load();
  while(local_todo_steps > done_steps.load()) {}
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void mark_evict_nodes(
    const IdType* evict_node_list, const size_t num_input,
    const IdType fall_back_location_id,
    const IdType old_location_id,
    HashTableEntryLocation* _location, HashTableEntryOffset* _offset) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t idx = block_start + threadIdx.x; idx < block_end; idx += BLOCK_SIZE) {
    if (idx < num_input) {
      const IdType node_id = evict_node_list[idx];
      assert(_location[node_id] == old_location_id);
      _location[node_id] = fall_back_location_id;
      _offset[node_id] = node_id;
    }
  }
}

struct _LocationPack {
  const HashTableEntryLocation* data[9];
};

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void mark_remote_evict_nodes(
    _LocationPack _remote_location,
    HashTableEntryLocation* local_location_table,
    HashTableEntryOffset* local_offset_table,
    const HashTableEntryLocation local_loc, const HashTableEntryLocation cpu_loc,
    const size_t num_input) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t node_id = block_start + threadIdx.x; node_id < block_end; node_id += BLOCK_SIZE) {
    if (node_id < num_input) {
      auto location_id = local_location_table[node_id];
      if (location_id != local_loc && location_id != cpu_loc && _remote_location.data[location_id][node_id] != location_id) {
        local_location_table[node_id] = cpu_loc;
        local_offset_table[node_id] = node_id;
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

void RefreshSession::refresh_after_solve() {
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

  // figure out new local cache id list
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "making local node list";
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

  CHECK(num_new_local_node <= _cache_ctx->_cache_space_capacity) << "cache space can not hold refreshed new cache";
  TensorPtr new_local_node_list_gpu = Tensor::CopyToExternal(new_local_node_list_cpu, _cache_ctx->_gpu_mem_allocator, gpu_ctx, stream);

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "finding preserved node and new insert nodes";
  // figure out preserved node id list for later extract it's used offset, and newly inserted node id list for later insertion
  TensorPtr preserved_node_list_gpu = Tensor::EmptyExternal(kI32, {num_new_local_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  TensorPtr new_insert_node_list_gpu = Tensor::EmptyExternal(kI32, {num_new_local_node}, _cache_ctx->_gpu_mem_allocator, gpu_ctx, "");
  size_t num_preserved_node, num_new_insert_node_;
  cuda::CubSelectBySideNe<IdType>(gpu_ctx, (const IdType *)new_local_node_list_gpu->CPtr<IdType>(), num_new_local_node, _cache_ctx->_hash_table_location, new_insert_node_list_gpu->Ptr<IdType>(), num_new_insert_node_, _local_location_id, _cache_ctx->_gpu_mem_allocator, stream);
  cuda::CubSelectBySideEq<IdType>(gpu_ctx, (const IdType *)new_local_node_list_gpu->CPtr<IdType>(), num_new_local_node, _cache_ctx->_hash_table_location, preserved_node_list_gpu->Ptr<IdType>(), num_preserved_node, _local_location_id, _cache_ctx->_gpu_mem_allocator, stream);
  CHECK(num_preserved_node + num_new_insert_node_ == num_new_local_node);
  new_local_node_list_gpu = nullptr;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "preserved node = " << num_preserved_node;
  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "new insert node = " << num_new_insert_node_;

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
    mark_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(
        evicted_node_list_gpu->CPtr<IdType>(), num_eviced_node, 
        _cache_ctx->_cpu_location_id, 
        _local_location_id,
        _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset);
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
    SAM_CUDA_PREPARE_1D(num_preserved_node);
    mark_remote_evict_nodes<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(_location_param, _cache_ctx->_hash_table_location, _cache_ctx->_hash_table_offset, 
        _cache_ctx->_local_location_id, _cache_ctx->_cpu_location_id, RunConfig::num_total_item);
    CUDA_CALL(cudaStreamSynchronize(cu_stream));
  }

  size_t current_progress = _cache_ctx->progress.load();

  AnonymousBarrier::_refresh_instance->Wait();

  Timer t0;
  while (_cache_ctx->progress.load() < current_progress + 1) {
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
    }
  }

  AnonymousBarrier::_refresh_instance->Wait();

  if (_cache_ctx->_local_location_id == 0) LOG(ERROR) << "refresh done";

  _cache_ctx->_cache_nodes = num_new_local_node;
  _cache_ctx->_local_node_list_tensor = new_local_node_list_cpu;
}

} // namespace coll_cache_lib
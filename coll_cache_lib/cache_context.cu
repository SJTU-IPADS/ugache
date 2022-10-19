#include "common.h"
#include "cpu/cpu_device.h"
#include "cpu/mmap_cpu_device.h"
#include "run_config.h"
#include "logging.h"
#include "coll_cache/ndarray.h"
#include "coll_cache/optimal_solver_class.h"
// #include "atomic_barrier.h"
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "facade.h"
#include "timer.h"
#include "cache_context.h"
#include "cuda/cub_sort_wrapper.cuh"

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
  size_t dim;
};

template <int NUM_LINK, typename T, typename SrcDataIter_T, typename DstDataIter_T>
__global__ void extract_data_concurrent(ExtractConcurrentParam<NUM_LINK, SrcDataIter_T, DstDataIter_T> packed_param) {
  // block -> which link
  // block -> local block idx in this link
  // block -> num block of this link
  const IdType link_idx = packed_param.link_mapping[blockIdx.x];
  const IdType local_block_idx_x = blockIdx.x - packed_param.block_num_prefix_sum[link_idx];
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


void ExtractSession::GetMissCacheIndex(
    SrcKey* & output_src_index, DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);

  output_src_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
  output_dst_index_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(DstVal));

  output_src_index = output_src_index_handle->ptr<SrcKey>();
  output_dst_index = output_dst_index_handle->ptr<DstVal>();

  auto output_src_index_alter_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(SrcKey));
  auto output_dst_index_alter_handle = _cache_ctx->_gpu_mem_allocator(num_nodes * sizeof(DstVal));

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

  auto workspace_handle = _cache_ctx->_gpu_mem_allocator(workspace_bytes);
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
    output_src_index_handle = output_src_index_alter_handle;
    output_dst_index_handle = output_dst_index_alter_handle;
  }

  // std::cout << "coll free workspace "<< t2.Passed() << "\n";
}


void ExtractSession::SplitGroup(const SrcKey * src_index, const size_t len, IdType * & group_offset, StreamHandle stream){
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(_cache_ctx->_trainer_ctx);
  auto cpu_ctx = CPU(CPU_CUDA_HOST_MALLOC_DEVICE);
  const size_t num_tiles = RoundUpDiv(len, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  Timer t0;
  group_offset = (IdType*)Device::Get(cpu_ctx)->AllocWorkspace(cpu_ctx, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  std::memset(group_offset, 0, sizeof(IdType) * (_cache_ctx->_num_location + 1));
  group_offset[_cache_ctx->_num_location] = len;
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
  TensorPtr link_mapping = Tensor::Empty(kI32, {108}, CPU(), "");
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
    }
    total_num_node += param.num_node_array[i];
  }
  link_mapping = Tensor::CopyToExternal(link_mapping, _cache_ctx->_gpu_mem_allocator, _cache_ctx->_trainer_ctx, stream);
  param.block_num_prefix_sum[NUM_LINK] = total_required_num_sm;
  param.link_mapping = link_mapping->CPtr<IdType>();
  param.dim = _cache_ctx->_dim;

  if (total_num_node == 0) return;
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
  } else if (RunConfig::coll_cache_no_group) {
    // CHECK(false) << "Multi source extraction is not supported now";
    auto trainer_gpu_device = Device::Get(_trainer_ctx);
    auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));
    Timer t0;
    double get_index_time = t0.Passed();
    Timer t1;
    CombineNoGroup(nodes, num_nodes, output, _trainer_ctx, _dtype, _dim, stream);
    double combine_time = t1.Passed();
    if (task_key != 0xffffffffffffffff) {
      // size_t num_hit = group_offset[1];
      Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
      Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[2]);
      Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
  } else 
#endif
  if (RunConfig::coll_cache_concurrent_link) {
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
    double get_index_time = t0.Passed();
    
    // std::cout << "Split GrOup " <<t1.Passed() << "\n";
    double combine_times[3] = {0, 0, 0};
    // cpu first, then concurrent remote, then local
    auto call_combine = [src_index, group_offset, dst_index, nodes, this, output, stream](int location_id){
      CombineOneGroup(src_index + group_offset[location_id], 
                      dst_index + group_offset[location_id], 
                      nodes + group_offset[location_id], 
                      group_offset[location_id+1] - group_offset[location_id], 
                      _cache_ctx->_device_cache_data[location_id], output, stream);
    };
    Timer t1;
    call_combine(_cache_ctx->_cpu_location_id);
    combine_times[0] = t1.Passed();

    // DistEngine::Get()->GetTrainerBarrier()->Wait();

    {
      // impl1: single kernel, limited num block
      t1.Reset();
      switch(RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id].size()) {
        case 1: CombineConcurrent<1>(src_index, dst_index, group_offset, output, stream); break;
        case 2: CombineConcurrent<2>(src_index, dst_index, group_offset, output, stream); break;
        case 3: CombineConcurrent<3>(src_index, dst_index, group_offset, output, stream); break;
        case 4: CombineConcurrent<4>(src_index, dst_index, group_offset, output, stream); break;
        default: CHECK(false);
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
    }

    t1.Reset();
    call_combine(_cache_ctx->_local_location_id);
    combine_times[2] = t1.Passed();

    output_src_index_handle = nullptr;
    output_dst_index_handle = nullptr;
    if (task_key != 0xffffffffffffffff) {
      // size_t num_miss = group_offset[_cpu_location_id+1]- group_offset[_cpu_location_id];
      // size_t num_local = group_offset[_local_location_id+1] - group_offset[_local_location_id];
      // size_t num_remote = num_nodes - num_miss - num_local;
      // // size_t num_hit = group_offset[1];
      // Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
      // Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
      // Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[2]);
      // Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
      // Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    }
    cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
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

    auto call_combine = [src_index, group_offset, dst_index, nodes, this, output, stream](int location_id){
      CombineOneGroup(src_index + group_offset[location_id], 
                      dst_index + group_offset[location_id], 
                      nodes + group_offset[location_id], 
                      group_offset[location_id+1] - group_offset[location_id], 
                      _cache_ctx->_device_cache_data[location_id], output, stream);
    };
    Timer t_cpu;
    call_combine(_cache_ctx->_cpu_location_id);
    combine_times[0] = t_cpu.Passed();

    // DistEngine::Get()->GetTrainerBarrier()->Wait();
    {
      t1.Reset();
      for (auto & link : RunConfig::coll_cache_link_desc.link_src[_cache_ctx->_local_location_id]) {
        for (auto dev_id : link) {
          call_combine(dev_id);
          // IdType offset = group_offset[dev_id];
          // IdType link_num_node = group_offset[dev_id+1] - offset;
          // CombineOneGroup(src_index + offset, dst_index + offset, nodes + offset, link_num_node, cache_ctx->_device_cache_data[dev_id], output, stream);
        }
      }
      combine_times[1] = t1.Passed();
    }

    t1.Reset();
    call_combine(_cache_ctx->_local_location_id);
    combine_times[2] = t1.Passed();

    output_src_index_handle = nullptr;
    output_dst_index_handle = nullptr;
    // if (task_key != 0xffffffffffffffff) {
    //   size_t num_miss = group_offset[_cpu_location_id+1]- group_offset[_cpu_location_id];
    //   size_t num_local = group_offset[_local_location_id+1] - group_offset[_local_location_id];
    //   size_t num_remote = num_nodes - num_miss - num_local;
    //   // size_t num_hit = group_offset[1];
    //   Profiler::Get().LogStep(task_key, kLogL1FeatureBytes, GetTensorBytes(_dtype, {num_nodes, _dim}));
    //   Profiler::Get().LogStep(task_key, kLogL1MissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    //   Profiler::Get().LogStep(task_key, kLogL1RemoteBytes, GetTensorBytes(_dtype, {num_remote, _dim}));
    //   Profiler::Get().LogStep(task_key, kLogL3CacheGetIndexTime, get_index_time);
    //   Profiler::Get().LogStep(task_key, kLogL3CacheCombineMissTime,combine_times[0]);
    //   Profiler::Get().LogStep(task_key, kLogL3CacheCombineRemoteTime,combine_times[1]);
    //   Profiler::Get().LogStep(task_key, kLogL3CacheCombineCacheTime,combine_times[2]);
    //   Profiler::Get().LogEpochAdd(task_key, kLogEpochFeatureBytes,GetTensorBytes(_dtype, {num_nodes, _dim}));
    //   Profiler::Get().LogEpochAdd(task_key, kLogEpochMissBytes, GetTensorBytes(_dtype, {num_miss, _dim}));
    // }
    cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);
  }
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
  _local_location_id = location_id;
  size_t num_total_nodes = coll_cache_ptr->_nid_to_block->Shape()[0];
  size_t num_blocks = coll_cache_ptr->_block_placement->Shape()[0];

  CHECK(RunConfig::coll_cache_concurrent_link == false) << "Not sure old init method support concurrent link";

  Timer t;

  auto gpu_device = Device::Get(gpu_ctx);
  auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));

  _device_cache_data.resize(_num_location, nullptr);
  _device_cache_data[_cpu_location_id] = cpu_data;
  _hash_table_location_handle = _gpu_mem_allocator(sizeof(HashTableEntryLocation) * num_total_nodes);
  _hash_table_offset_handle   = _gpu_mem_allocator(sizeof(HashTableEntryOffset)   * num_total_nodes);

  _hash_table_location = _hash_table_location_handle->ptr<HashTableEntryLocation>();
  _hash_table_offset   = _hash_table_offset_handle->ptr<HashTableEntryOffset>();


  LOG(INFO) << "CollCacheManager: Initializing hashtable location...";

  auto cu_stream = static_cast<cudaStream_t>(stream);
  // 1. Build a mapping from node id to target device
  {
    TensorPtr node_to_block_gpu = Tensor::CopyToExternal(coll_cache_ptr->_nid_to_block, _gpu_mem_allocator, gpu_ctx, stream);   // large
    TensorPtr block_placement_gpu = Tensor::CopyToExternal(coll_cache_ptr->_block_placement, _gpu_mem_allocator, gpu_ctx, stream); // small
    // build a map from placement combinations to source decision
    size_t placement_to_src_nbytes = sizeof(int) * (1 << RunConfig::num_device);
    int * placement_to_src_cpu = (int*) cpu_device->AllocWorkspace(CPU(), placement_to_src_nbytes);
    PreDecideSrc(RunConfig::num_device, _local_location_id, _cpu_location_id, placement_to_src_cpu);
    MemHandle placement_to_src_gpu_handle = _gpu_mem_allocator(placement_to_src_nbytes);
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
  auto node_list_buffer_handle = _gpu_mem_allocator(num_total_nodes * sizeof(IdType));
  IdType* node_list_buffer = node_list_buffer_handle->ptr<IdType>();
  // IdType * group_offset;
  size_t num_cached_nodes;
  size_t num_cpu_nodes;
  {
    IdType* cache_node_list = node_list_buffer;
    // now we want to select nodes with hash_table_location==local id
    cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, cache_node_list, num_cached_nodes, _local_location_id, _gpu_mem_allocator, stream);
    cuda::CubCountByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, num_cpu_nodes, _cpu_location_id, _gpu_mem_allocator, stream);
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
    _device_cache_data_local_handle = _gpu_mem_allocator(_cache_nbytes);
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
    cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, remote_node_list, num_remote_nodes, i, _gpu_mem_allocator, stream);
    if (num_remote_nodes == 0) continue;
    CUDA_CALL(cudaDeviceEnablePeerAccess(i, 0));
    _device_cache_data[i] = device_cache_data_list.extract(i);
    auto remote_hash_table_offset = (const HashTableEntryOffset * )hash_table_offset_list.extract(i);

    SAM_CUDA_PREPARE_1D(num_remote_nodes);
    init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
        _hash_table_location, _hash_table_offset, 
        remote_hash_table_offset, remote_node_list, num_remote_nodes, i);
    gpu_device->StreamSync(gpu_ctx, stream);
    // CUDA_CALL(cudaIpcCloseMemHandle((void*)remote_hash_table_offset))
  }

  // 4. Free index
  node_list_buffer_handle = nullptr;
  // trainer_gpu_device->FreeDataSpace(trainer_ctx, loc_list);
  // cpu_device->FreeWorkspace(CPU(CPU_CUDA_HOST_MALLOC_DEVICE), group_offset);

  size_t num_remote_nodes = num_total_nodes - num_cached_nodes - num_cpu_nodes;

  if (RunConfig::coll_cache_concurrent_link) {
    _concurrent_stream_array.resize(RunConfig::num_device - 1);
    for (auto & stream : _concurrent_stream_array) {
      cudaStream_t & cu_s = reinterpret_cast<cudaStream_t &>(stream);
      CUDA_CALL(cudaStreamCreate(&cu_s));
    }
  }

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
  _trainer_ctx = gpu_ctx;
  _dtype = dtype;
  _dim = dim;
  // cpu counts as a location
  _num_location = RunConfig::num_device + 1,
  _cpu_location_id = RunConfig::num_device;
  LOG(INFO) << "Coll cache init with " << RunConfig::num_device << " gpus, " << _num_location << " locations";

  LOG(ERROR) << "Building Coll Cache with ... num gpu device is " << RunConfig::num_device;

  _local_location_id = location_id;
  size_t num_total_nodes = coll_cache_ptr-> _nid_to_block->Shape()[0];
  size_t num_blocks = coll_cache_ptr->_block_placement->Shape()[0];

  // RunConfig::coll_cache_link_desc = coll_cache::AsymmLinkDesc::AutoBuild(trainer_ctx);
  if (RunConfig::coll_cache_concurrent_link) {
    _concurrent_stream_array.resize(RunConfig::num_device - 1);
    for (auto & stream : _concurrent_stream_array) {
      cudaStream_t & cu_s = reinterpret_cast<cudaStream_t &>(stream);
      CUDA_CALL(cudaStreamCreate(&cu_s));
    }
  }

  Timer t;

  auto gpu_device = Device::Get(gpu_ctx);
  auto cpu_device = Device::Get(CPU(CPU_CUDA_HOST_MALLOC_DEVICE));

  _device_cache_data.resize(_num_location, nullptr);
  _device_cache_data[_cpu_location_id] = cpu_data;
  _hash_table_location_handle = _gpu_mem_allocator(sizeof(HashTableEntryLocation) * num_total_nodes);
  _hash_table_offset_handle   = _gpu_mem_allocator(sizeof(HashTableEntryOffset)   * num_total_nodes);

  _hash_table_location = _hash_table_location_handle->ptr<HashTableEntryLocation>();
  _hash_table_offset   = _hash_table_offset_handle->ptr<HashTableEntryOffset>();


  LOG(INFO) << "CollCacheManager: Initializing hashtable location...";

  auto cu_stream = static_cast<cudaStream_t>(stream);
  // 1. Build a mapping from node id to target device
  {
    TensorPtr node_to_block_gpu = Tensor::CopyToExternal(coll_cache_ptr->_nid_to_block, _gpu_mem_allocator, gpu_ctx, stream);   // large
    TensorPtr block_access_advise_gpu = Tensor::CopyLineToExternel(coll_cache_ptr->_block_access_advise, location_id, _gpu_mem_allocator, gpu_ctx, stream); // small

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
  auto node_list_buffer_handle = _gpu_mem_allocator(num_total_nodes * sizeof(IdType));
  IdType* node_list_buffer = (IdType*)node_list_buffer_handle->ptr();
  // IdType * group_offset;
  size_t num_cached_nodes;
  size_t num_cpu_nodes;
  {
    IdType* cache_node_list = node_list_buffer;
    // now we want to select nodes with hash_table_location==local id
    cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, cache_node_list, num_cached_nodes, _local_location_id, _gpu_mem_allocator, stream);
    cuda::CubCountByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, num_cpu_nodes, _cpu_location_id, _gpu_mem_allocator, stream);
    // CHECK_NE(num_cached_nodes, 0);
    _cache_nbytes = GetTensorBytes(_dtype, {num_cached_nodes, _dim});
    _device_cache_data_local_handle = _gpu_mem_allocator(_cache_nbytes);
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
      cuda::CubSelectIndexByEq<IdType>(gpu_ctx, (const IdType *)_hash_table_location, num_total_nodes, remote_node_list, num_remote_nodes, dev_id, _gpu_mem_allocator, stream);
      if (num_remote_nodes == 0) continue;
      CUDA_CALL(cudaDeviceEnablePeerAccess(dev_id, 0));
      _device_cache_data[dev_id] = device_cache_data_list.extract(dev_id);
      auto remote_hash_table_offset = (const HashTableEntryOffset * )hash_table_offset_list.extract(dev_id);

      SAM_CUDA_PREPARE_1D(num_remote_nodes);
      init_hash_table_remote<><<<grid, block, 0, cu_stream>>>(
          _hash_table_location, _hash_table_offset, 
          remote_hash_table_offset, remote_node_list, num_remote_nodes, dev_id);
      gpu_device->StreamSync(gpu_ctx, stream);
      // CUDA_CALL(cudaIpcCloseMemHandle((void*)remote_hash_table_offset))
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

} // namespace coll_cache_lib
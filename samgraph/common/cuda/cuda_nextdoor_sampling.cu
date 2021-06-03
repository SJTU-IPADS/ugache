#include <cassert>
#include <chrono>
#include <cstdio>

#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

#include "../common.h"
#include "../constant.h"
#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../timer.h"
#include "cuda_function.h"

namespace {

template <typename T> struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

} // namespace


namespace samgraph {
namespace common {
namespace cuda {

__global__ void nextSample(const IdType *indptr, const IdType *indices, 
                           const IdType *input, const size_t num_input, 
                           const size_t fanout, IdType *tmp_src,
                           IdType *tmp_dst, curandState* states) {
  size_t num_task = num_input * fanout;
  size_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  size_t max_span = blockDim.x * gridDim.x;
//  curandState state;
//  curand_init(num_input, threadId, 0, &state);

  for (size_t task_start = threadId; task_start < num_task; task_start += max_span) {
    const IdType rid = input[task_start / fanout];
    const IdType off = indptr[rid];
    const IdType len = indptr[rid + 1] - indptr[rid];
    size_t k = curand(&states[task_start]) % len;
    tmp_src[task_start] = rid;
    tmp_dst[task_start] = indices[off + k];
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_edge(IdType *edge_src, size_t *item_prefix,
                           const size_t num_input, const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<size_t, BLOCK_SIZE>;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  size_t count = 0;
#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (edge_src[index * fanout + j] != Constant::kEmptyKey) {
          ++count;
        }
      }
      // printf("index %lu  count %lu\n", index, count);
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    item_prefix[blockIdx.x] = count;
    // printf("blockIdx.x %d count %lu\n", blockIdx.x, count);
    if (blockIdx.x == 0) {
      item_prefix[gridDim.x] = 0;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_edge(const IdType *tmp_src, const IdType *tmp_dst,
                             IdType *out_src, IdType *out_dst, size_t *num_out,
                             const size_t *item_prefix, const size_t num_input,
                             const size_t fanout) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;

  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const size_t offset = item_prefix[blockIdx.x];

  BlockPrefixCallbackOp<size_t> prefix_op(0);

  // count successful placements
  for (size_t i = 0; i < VALS_PER_THREAD; ++i) {
    const size_t index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    size_t item_per_thread = 0;
    if (index < num_input) {
      for (size_t j = 0; j < fanout; j++) {
        if (tmp_src[index * fanout + j] != Constant::kEmptyKey) {
          item_per_thread++;
        }
      }
    }

    size_t item_prefix_per_thread = item_per_thread;
    BlockScan(temp_space)
        .ExclusiveSum(item_prefix_per_thread, item_prefix_per_thread,
                      prefix_op);
    __syncthreads();

    for (size_t j = 0; j < item_per_thread; j++) {
      out_src[offset + item_prefix_per_thread + j] =
          tmp_src[index * fanout + j];
      out_dst[offset + item_prefix_per_thread + j] =
          tmp_dst[index * fanout + j];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_out = item_prefix[gridDim.x];
    // printf("item_prefix %d\n", item_prefix[gridDim.x]);
  }
}

/**
 * @brief sampling algorithm from nextdoor
 * CSR format example:
        ROW_INDEX = [  0  2  4  7  8  ]
        COL_INDEX = [  0  1  1  3  2  3  4  5  ]
        V         = [ 10 20 30 40 50 60 70 80  ]
 * @param indptr      ROW_INDEX, sampling vertices
 * @param indices     COL_INDEX, neighbors
 * @param input       the indices of sampling vertices
 * @param num_input   the number of sampling vertices
 * @param fanout      the number of neighbors for each sampling vertex
 * @param out_src     src vertices of all neighbors 
 * @param out_dst     dst vertices of all neighbors
 * @param num_out     the number of all neighbors
 * @param ctx         GPU contex
 * @param stream      GPU stream
 * @param task_key
 * @param states      GPU random seeds list
 */
void GPUNextdoorSample(const IdType *indptr, const IdType *indices,
                       const IdType *input, const size_t num_input,
                       const size_t fanout, IdType *out_src, IdType *out_dst,
                       size_t *num_out, Context ctx, StreamHandle stream,
                       uint64_t task_key, curandState *states) {
  LOG(DEBUG) << "GPUSample: begin with num_input " << num_input
             << " and fanout " << fanout;
  Timer t0;

  auto sampler_device = Device::Get(ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  IdType *tmp_src = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  IdType *tmp_dst = static_cast<IdType *>(
      sampler_device->AllocWorkspace(ctx, sizeof(IdType) * num_input * fanout));
  LOG(DEBUG) << "GPUSample: cuda tmp_src malloc "
             << ToReadableSize(num_input * fanout * sizeof(IdType));
  LOG(DEBUG) << "GPUSample: cuda tmp_dst malloc "
             << ToReadableSize(num_input * fanout * sizeof(IdType));

  const size_t max_threads = 5 * 1024 * 1024;
  size_t num_threads = num_input * fanout;
  if (num_threads > max_threads) {
      num_threads = max_threads;
  }

  const size_t blockSize = 512;
  const dim3 nextGrid((num_threads + blockSize - 1) / blockSize);
  const dim3 nextBlock(blockSize);
  nextSample<<<nextGrid, nextBlock, 0, cu_stream>>>(indptr, indices, input, num_input, fanout,
                                            tmp_src, tmp_dst, states);
  sampler_device->StreamSync(ctx, stream);

  double sample_time = t0.Passed();

  const dim3 grid((num_input + Constant::kCudaBlockSize - 1) / Constant::kCudaBlockSize);
  const dim3 block(Constant::kCudaBlockSize);

  // sort tmp_src,tmp_dst -> out_src,out_dst, the size is num_input * fanout
  Timer t1;
  size_t temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, 
                                            tmp_src, out_src, tmp_dst, out_dst, 
                                            num_input * fanout, 0, sizeof(IdType) * 8, 
                                            cu_stream));
  sampler_device->StreamSync(ctx, stream);

  void *d_temp_storage  = sampler_device->AllocWorkspace(ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                            tmp_src, out_src, tmp_dst, out_dst, 
                                            num_input * fanout, 0, sizeof(IdType) * 8, 
                                            cu_stream));
  sampler_device->StreamSync(ctx, stream);
  sampler_device->FreeWorkspace(ctx, d_temp_storage);
  double sort_results_time = t1.Passed();
  LOG(DEBUG) << "GPUSample: sort the temporary results, time cost: "
             << sort_results_time;

  // count the prefix num
  Timer t2;
  size_t *item_prefix = static_cast<size_t *>(sampler_device->AllocWorkspace(ctx, sizeof(size_t) * num_input * fanout));
  LOG(DEBUG) << "GPUSample: cuda prefix_num malloc "
             << ToReadableSize(sizeof(int) * num_input * fanout);
  // TODO: implementation for init_prefix_num
  init_prefix_num<<<grid, block, 0, cu_stream>>>(out_src, out_dst, item_prefix, num_input * fanout);
  sampler_device->StreamSync(ctx, stream);

  temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes,
                                          item_prefix, item_prefix, num_input * fanout,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);

  d_temp_storage = sampler_device->AllocWorkspace(ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                          item_prefix, item_prefix, grid.x + 1,
                                          cu_stream));
  sampler_device->StreamSync(ctx, stream);
  sampler_device->FreeWorkspace(ctx, d_temp_storage);
  double prefix_sum_time = t2.Passed();

  // compact edge
  Timer t3;
  // TODO: compact edges
  compact_edge<<<grid, block, 0, cu_stream>>>(out_src, out_dst, item_prefix, num_input * fanout, num_out);
  sampler_device->StreamSync(ctx, stream);
  double compact_edge_time = t3.Passed();

  sampler_device->FreeWorkspace(ctx, item_prefix);
  sampler_device->FreeWorkspace(ctx, tmp_src);
  sampler_device->FreeWorkspace(ctx, tmp_dst);

  LOG(DEBUG) << "GPUSample: succeed ";
}

} // namespace cuda
} // namespace common
} // namespace samgraph


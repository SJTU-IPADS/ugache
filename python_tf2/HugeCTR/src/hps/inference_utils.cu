/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cub/cub.cuh>
#include <hps/inference_utils.hpp>

#include "coll_cache_lib/common.h"
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Overload CUDA atomic for other 64bit unsinged/signed integer type
__forceinline__ __device__ uint32_t atomicAdd(uint32_t* address, int val) {
  return (uint32_t)atomicAdd((unsigned int*)address, (unsigned int)val);
}

namespace HugeCTR {

// Kernels to combine the value buffer
__global__ void merge_emb_vec(float* d_output_emb_vec, const float* d_missing_emb_vec,
                              const uint64_t* d_missing_index, const size_t len,
                              const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

// Kernels to fill the default value to the output buffer
__global__ void fill_default_emb_vec(float* d_output_emb_vec, const float default_emb_vec,
                                     const uint64_t* d_missing_index, const size_t len,
                                     const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] = default_emb_vec;
  }
}

// Kernels to decompress the value buffer
__global__ void decompress_emb_vec(const float* d_src_emb_vec, const uint64_t* d_src_index,
                                   float* d_dst_emb_vec, const size_t len,
                                   const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t dst_emb_vec = idx / emb_vec_size;
    size_t dst_float = idx % emb_vec_size;
    size_t src_emb_vec = d_src_index[dst_emb_vec];
    d_dst_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_src_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

__global__ void unfold_index(uint64_t* d_src, uint32_t* d_dst, size_t num_src, size_t num_dst) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_src) d_dst[d_src[idx]] = 1;
}

__global__ void map_marking(uint64_t* d_src_index, uint32_t* d_src_mark, uint32_t* d_dst,
                            size_t num_mark, size_t num_dst) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_dst) d_dst[idx] = d_src_mark[d_src_index[idx]] ? 1 : 0;
}

template <typename T>
__global__ void add_up(const T* d_keys_in, const uint32_t* d_values_in, uint32_t* d_dst,
                       size_t num_items) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_items) d_dst[d_keys_in[idx]] += d_values_in[idx];
}

__global__ void min_vec(const uint32_t* d_src1, const uint32_t* d_src2, uint32_t* d_dst,
                        size_t num_items) {
  const size_t global_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (global_id < num_items) d_dst[global_id] = min(d_src1[global_id], d_src2[global_id]);
}

void merge_emb_vec_async(float* d_vals_merge_dst_ptr, const float* d_vals_retrieved_ptr,
                         const uint64_t* d_missing_index_ptr, const size_t missing_len,
                         const size_t emb_vec_size, const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  merge_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_vals_merge_dst_ptr, d_vals_retrieved_ptr, d_missing_index_ptr, missing_len, emb_vec_size);
}

void fill_default_emb_vec_async(float* d_vals_merge_dst_ptr, const float default_emb_vec,
                                const uint64_t* d_missing_index_ptr, const size_t missing_len,
                                const size_t emb_vec_size, const size_t BLOCK_SIZE,
                                cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  fill_default_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_vals_merge_dst_ptr, default_emb_vec, d_missing_index_ptr, missing_len, emb_vec_size);
}

void decompress_emb_vec_async(const float* d_unique_src_ptr, const uint64_t* d_unique_index_ptr,
                              float* d_decompress_dst_ptr, const size_t decompress_len,
                              const size_t emb_vec_size, const size_t BLOCK_SIZE,
                              cudaStream_t stream) {
  if (decompress_len == 0) {
    return;
  }
  size_t decompress_len_in_float = decompress_len * emb_vec_size;
  decompress_emb_vec<<<((decompress_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream>>>(
      d_unique_src_ptr, d_unique_index_ptr, d_decompress_dst_ptr, decompress_len, emb_vec_size);
}

using namespace coll_cache_lib::common;

struct CubSum {
  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__ T operator()(const T& a, const T& b) const {
    return (a + b);
  }
};

struct FlipConverter {
  __host__ __device__ __forceinline__ uint32_t operator()(const uint32_t& a) const { return !a; }
};

template <typename T>
void MathUtil<T>::CubCountByKey(const T* d_keys, uint32_t* d_values, T* d_unique_keys_out,
                                uint32_t* d_aggregates_out, const size_t num_items,
                                uint64_t* d_num_runs_out, cudaStream_t stream, bool flip) {
  // Declare, allocate, and initialize device-accessible pointers for input and output
  CubSum reduction_op;
  FlipConverter flip_op;
  cub::TransformInputIterator<uint32_t, FlipConverter, uint32_t*> itr_flip(d_values, flip_op);

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  auto do_reduce = [&] {
    if (flip) {
      cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys, d_unique_keys_out,
                                     itr_flip, d_aggregates_out, d_num_runs_out, reduction_op,
                                     num_items, stream);
    } else {
      cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys, d_unique_keys_out,
                                     d_values, d_aggregates_out, d_num_runs_out, reduction_op,
                                     num_items, stream);
    }
  };

  do_reduce();
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run reduce-by-key
  do_reduce();
  cudaStreamSynchronize(stream);
  cudaFree(d_temp_storage);
}

template <typename T>
void MathUtil<T>::CubReduceSum(const T* d_in, uint64_t* d_out, size_t num_items,
                               cudaStream_t stream) {
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_temp_storage);
}

template <typename T>
void MathUtil<T>::CubSortPairs(const T* d_keys_in, const uint32_t* d_values_in, T* d_keys_out,
                               uint32_t* d_values_out, size_t num_items, cudaStream_t stream) {
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                  d_values_in, d_values_out, num_items, 0, sizeof(T) * 8, stream);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                  d_values_in, d_values_out, num_items, 0, sizeof(T) * 8, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_temp_storage);
}

template <typename T>
void MathUtil<T>::UnfoldIndexVec(uint64_t* d_src, uint32_t* d_dst, size_t num_src, size_t num_dst,
                                 cudaStream_t stream) {
  if (num_src == 0) return;
  unfold_index<<<RoundUpDiv(num_src, Constant::kCudaBlockSize), Constant::kCudaBlockSize, 0,
                 stream>>>(d_src, d_dst, num_src, num_dst);
}

template <typename T>
void MathUtil<T>::Mark(uint64_t* d_src_index, uint32_t* d_src_mark, uint32_t* d_dst,
                       size_t num_mark, size_t num_dst, cudaStream_t stream) {
  if (num_dst == 0) return;
  map_marking<<<RoundUpDiv(num_dst, Constant::kCudaBlockSize), Constant::kCudaBlockSize, 0,
                stream>>>(d_src_index, d_src_mark, d_dst, num_mark, num_dst);
}

template <typename T>
void MathUtil<T>::AddUp(const T* d_keys_in, const uint32_t* d_values_in, uint32_t* d_dst,
                        size_t num_items, cudaStream_t stream) {
  if (num_items == 0) return;
  add_up<<<RoundUpDiv(num_items, Constant::kCudaBlockSize), Constant::kCudaBlockSize, 0, stream>>>(
      d_keys_in, d_values_in, d_dst, num_items);
}

template <typename T>
void MathUtil<T>::Min(const uint32_t* d_src1, const uint32_t* d_src2, uint32_t* d_dst,
                      size_t num_items, cudaStream_t stream) {
  if (num_items == 0) return;
  min_vec<<<RoundUpDiv(num_items, Constant::kCudaBlockSize), Constant::kCudaBlockSize, 0, stream>>>(
      d_src1, d_src2, d_dst, num_items);
}

template class MathUtil<unsigned int>;
template class MathUtil<long long>;

}  // namespace HugeCTR

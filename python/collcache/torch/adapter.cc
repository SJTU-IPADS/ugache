/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
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
 *
 */

#include "adapter.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "torch/types.h"

#undef LOG
#undef CHECK_NOTNULL
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_EQ
#undef CHECK
#include "coll_cache_lib/common.h"
// #include "../common/cuda/cuda_engine.h"
// #include "../common/dist/dist_engine.h"
#include "coll_cache_lib/profiler.h"
#include "coll_cache_lib/timer.h"

// Use the torch built-in CHECK macros
// #include "../common/logging.h"

namespace {
::torch::Dtype to_torch_data_type(coll_cache_lib::common::DataType dt) {
  switch (dt) {
    case coll_cache_lib::common::kF32: return ::torch::kF32;
    case coll_cache_lib::common::kF64: return ::torch::kF64;
    case coll_cache_lib::common::kF16: return ::torch::kF16;
    case coll_cache_lib::common::kU8:  return ::torch::kU8;
    case coll_cache_lib::common::kI32: return ::torch::kI32;
    case coll_cache_lib::common::kI8:  return ::torch::kI8;
    case coll_cache_lib::common::kI64: return ::torch::kI64;
  }
}

size_t internal_feat_dim = 0;
size_t external_feat_dim = 0;
c10::ScalarType external_dtype;

bool _use_fp16 = false;
cudaStream_t _stream = nullptr;

};

namespace coll_cache_lib {
namespace torch {

extern "C" {

void coll_torch_init(int replica_id, size_t key_space_size, int dev_id, void *cpu_data, size_t dim, double cache_percentage, bool use_fp16) {
  c10::ScalarType t = ::torch::kI8;
  std::function<common::MemHandle(size_t)> allocator = [dev_id](size_t nbytes){
    auto ret = std::make_shared<TorchMemHandle>();
    auto device = "cuda:" + std::to_string(dev_id);
    ret->tensor_hold = ::torch::empty({(long)nbytes},::torch::TensorOptions().dtype(::torch::kI8).device(device));
    ret->nbytes_ = nbytes;
    return ret;
  };
  cudaSetDevice(dev_id);
  cudaStreamCreate(&_stream);

  external_feat_dim = dim;
  if (use_fp16) {
    _use_fp16 = true;
    if (dim % 2 == 0) { std::cerr << "dimension must be even\n"; abort(); }
    dim /= 2;
    external_dtype = ::torch::kF16;
  } else {
    external_dtype = ::torch::kF32;
  }
  internal_feat_dim = dim;
  common::coll_cache_init(replica_id, key_space_size, allocator, cpu_data, common::kF32, dim, cache_percentage, _stream);
}

void coll_torch_init_t(int replica_id, int dev_id, ::torch::Tensor emb, double cache_percentage) {
  c10::ScalarType t = ::torch::kI8;
  std::function<common::MemHandle(size_t)> allocator = [dev_id](size_t nbytes){
    auto ret = std::make_shared<TorchMemHandle>();
    auto device = "cuda:" + std::to_string(dev_id);
    ret->tensor_hold = ::torch::empty({(long)nbytes},::torch::TensorOptions().dtype(::torch::kI8).device(device));
    ret->nbytes_ = nbytes;
    return ret;
  };
  cudaSetDevice(dev_id);
  cudaStreamCreate(&_stream);

  size_t dim = emb.size(1);
  size_t key_space_size = emb.size(0);
  bool use_fp16 = (emb.scalar_type() == ::torch::kF16);

  external_feat_dim = dim;
  if (use_fp16) {
    _use_fp16 = true;
    if (dim % 2 == 0) { std::cerr << "dimension must be even\n"; abort(); }
    dim /= 2;
    external_dtype = ::torch::kF16;
  } else {
    external_dtype = ::torch::kF32;
  }
  internal_feat_dim = dim;
  common::coll_cache_init(replica_id, key_space_size, allocator, emb.data_ptr(), common::kF32, dim, cache_percentage, _stream);
}

::torch::Tensor coll_torch_test(int replica_id,
                                ::torch::Tensor key /*
                                ,cudaStream_t stream
                                */
                                ) {
  ::torch::Tensor tensor = ::torch::empty(
      {key.size(0), (long)external_feat_dim},
      ::torch::TensorOptions()
          .dtype(::torch::kF32)
          .device("cuda:" + std::to_string(common::RunConfig::device_id_list[replica_id])));
  return tensor;
}
::torch::Tensor coll_torch_lookup_key_t_val_ret(int replica_id,
                                                ::torch::Tensor key) {
  auto device = "cuda:" + std::to_string(common::RunConfig::device_id_list[replica_id]);
  auto padded_len = common::RoundUp<long>(key.size(0), 8);
  ::torch::Tensor tensor = ::torch::empty(
      {padded_len, (long)external_feat_dim},
      ::torch::TensorOptions()
          .dtype(external_dtype)
          .device(device));
  common::coll_cache_lookup(replica_id, (uint32_t*)key.data_ptr(), key.size(0), tensor.data_ptr(), _stream);
  return tensor;
}

void coll_torch_record(int replica_id, ::torch::Tensor key) {
  common::coll_cache_record(replica_id, (uint32_t*)key.data_ptr(), key.size(0));
}

PYBIND11_MODULE(c_lib, m) {
  m.def("coll_torch_init", &coll_torch_init);
  m.def("coll_torch_init_t", &coll_torch_init_t);
  m.def("coll_torch_test", &coll_torch_test);
  m.def("coll_torch_record", &coll_torch_record);
  m.def("coll_torch_lookup_key_t_val_ret", &coll_torch_lookup_key_t_val_ret);
}

}


} // namespace torch
}  // namespace coll_cache_lib

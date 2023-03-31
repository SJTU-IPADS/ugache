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

#pragma once
// #include <TH/TH.h>
// #include <THC/THC.h>
#include <torch/torch.h>
#include "../common/operation.h"
#include "coll_cache_lib/run_config.h"

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace coll_cache_lib {
namespace torch {

namespace {
class TorchMemHandle : public common::ExternelGPUMemoryHandler {
 public:
  size_t nbytes_ = 0;
  ::torch::Tensor tensor_hold;
  void* ptr() override { return tensor_hold.data_ptr(); }
  size_t nbytes() override { return nbytes_; }
  ~TorchMemHandle() { /* HCTR_LOG_S(ERROR, WORLD) << "pointer " << tensor_hold.data() << " freed\n";
                      */
  }
};
}

extern "C" {

// ::torch::Tensor samgraph_torch_get_graph_feat(uint64_t key);
// ::torch::Tensor samgraph_torch_get_graph_label(uint64_t key);
// ::torch::Tensor samgraph_torch_get_graph_row(uint64_t key, int layer_idx);
// ::torch::Tensor samgraph_torch_get_graph_col(uint64_t key, int layer_idx);
// ::torch::Tensor samgraph_torch_get_graph_data(uint64_t key, int layer_idx);
// ::torch::Tensor samgraph_torch_get_unsupervised_graph_row(uint64_t key);
// ::torch::Tensor samgraph_torch_get_unsupervised_graph_col(uint64_t key);

// ::torch::Tensor samgraph_torch_get_dataset_feat();
// ::torch::Tensor samgraph_torch_get_dataset_label();
// ::torch::Tensor samgraph_torch_get_graph_input_nodes(uint64_t key);
// ::torch::Tensor samgraph_torch_get_graph_output_nodes(uint64_t key);

void coll_torch_init(int replica_id, size_t key_space_size, int dev_id, void *cpu_data, size_t dim, double cache_percentage, bool use_fp16 = false);

// void coll_torch_init_from_tensor(int replica_id, size_t key_space_size, int dev_id, ::torch::Tensor tensor, common::DataType dtype, size_t dim, double cache_percentage, cudaStream_t stream = nullptr) {
//   coll_torch_init(replica_id, key_space_size, dev_id, tensor.data_ptr(), dtype, dim, cache_percentage, stream);
// }
// void coll_torch_lookup(int replica_id, uint32_t* key, size_t num_keys, void* output, cudaStream_t stream = nullptr) {
//   common::samgraph_lookup(replica_id, key, num_keys, output, stream);
// }
// void coll_torch_lookup_val_t(int replica_id, uint32_t* key, size_t num_keys, ::torch::Tensor tensor, cudaStream_t stream = nullptr) {
//   coll_torch_lookup(replica_id, key, num_keys, tensor.data_ptr(), stream);
// }
// ::torch::Tensor coll_torch_lookup_val_ret(int replica_id, uint32_t* key, size_t num_keys, cudaStream_t stream = nullptr) {
//   ::torch::Tensor tensor = ::torch::empty({(long)num_keys, (long)common::samgraph_feat_dim()}, ::torch::TensorOptions().dtype(::torch::kF32).device(common::RunConfig::device_id_list[replica_id]));
//   coll_torch_lookup_val_t(replica_id, key, num_keys, tensor, stream);
//   return tensor;
// }
::torch::Tensor coll_torch_lookup_key_t_val_ret(int replica_id,
                                                ::torch::Tensor key);

::torch::Tensor coll_torch_test(int replica_id, ::torch::Tensor key);

void coll_torch_record(int replica_id, ::torch::Tensor key);
}

}  // namespace torch
}  // namespace coll_cache_lib

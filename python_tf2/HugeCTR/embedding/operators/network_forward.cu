/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "HugeCTR/include/utils.hpp"
#include "generic_lookup.cuh"
#include "network_forward.hpp"
namespace embedding {
using namespace core;

NetworkForward::NetworkForward(std::shared_ptr<CoreResourceManager> core, int num_gpus)
    : core_(core), num_gpus_(num_gpus) {}

void NetworkForward::compute(const TensorList& network_comm_buffer, const Tensor& network_ids,
                             const Tensor& network_gpu_ids, const Tensor& network_offsets,
                             const Tensor& network_dst_lookup_ids,
                             const TensorList& network_ev_sizes,
                             const TensorList& network_ev_offsets, Tensor& output_buffer,
                             const Tensor& d_ev_size_offset, int batch_size, int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;
  DISPATCH_FLOAT_AND_HALF_FUNCTION(network_comm_buffer.dtype().type(), emb_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), dst_emb_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();

      const int* network_ids_ptr = network_ids.get<int>();
      const int* network_gpu_ids_ptr = network_gpu_ids.get<int>();
      const int* network_offsets_ptr = network_offsets.get<int>();
      const int* network_dst_lookup_ids_ptr = network_dst_lookup_ids.get<int>();
      const int** network_ev_sizes_ptr = network_ev_sizes.get<int>();
      const int** network_ev_offsets_ptr = network_ev_offsets.get<int>();
      const emb_t** network_comm_buffer_ptr = network_comm_buffer.get<emb_t>();
      const int* d_ev_size_offset_ptr = d_ev_size_offset.get<int>();
      dst_emb_t* output_buffer_ptr = output_buffer.get<dst_emb_t>();
      int num_network_dst_lookup_ids = network_dst_lookup_ids.get_num_elements();

      auto multi_to_one_desc = make_MultiToOne<emb_t, dst_emb_t>(
          num_network_dst_lookup_ids * batch_size_per_gpu,
          [=] __device__(int i) {
            int bid = i / num_network_dst_lookup_ids;
            int lookup_id = i % num_network_dst_lookup_ids;
            return bid * network_offsets_ptr[num_network_dst_lookup_ids] +
                   network_offsets_ptr[lookup_id];
          },
          [=] __device__(int i) { return 1; },
          [=] __device__(int i) {
            int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
            return d_ev_size_offset_ptr[dst_lookup_id + 1] - d_ev_size_offset_ptr[dst_lookup_id];
          },
          [=] __device__(int i) {
            int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
            int id = i % network_offsets_ptr[num_network_dst_lookup_ids];

            int network_gpu_id = network_gpu_ids_ptr[id];
            int network_id = network_ids_ptr[id];
            int ev_offset = network_ev_offsets_ptr[network_gpu_id][network_id] * batch_size_per_gpu;
            int ev_size = network_ev_sizes_ptr[network_gpu_id][network_id];

            return network_comm_buffer_ptr[network_gpu_id] + ev_offset + bid * ev_size;
          },
          [=] __device__(int i) {
            int bid = i / num_network_dst_lookup_ids;
            int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

            int ev_offset = d_ev_size_offset_ptr[lookup_id] * batch_size_per_gpu;
            int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
            return output_buffer_ptr + ev_offset + bid * ev_size;
          });
      copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
    });
  });
}
}  // namespace embedding

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

#pragma once
#include "coll_cache_lib/atomic_barrier.h"
#include "coll_cache_lib/facade.h"
#include "hier_parameter_server.hpp"
#include "tensorflow/core/framework/op_kernel.h"

namespace coll_cache_lib {

class LookupManager final {
  class HPSMemHandle : public coll_cache_lib::common::ExternelGPUMemoryHandler {
   public:
    size_t nbytes_ = 0;
    tensorflow::Tensor tensor_hold;
    void* ptr() override { return tensor_hold.data(); }
    size_t nbytes() override { return nbytes_; }
    ~HPSMemHandle() { /* HCTR_LOG_S(ERROR, WORLD) << "pointer " << tensor_hold.data() << " freed\n";
                       */
    }
  };

 public:
  ~LookupManager() = default;
  LookupManager(const LookupManager&) = delete;
  LookupManager& operator=(const LookupManager&) = delete;

  static LookupManager* instance();
  void operator delete(void*);
  void init(const int32_t global_replica_id, tensorflow::OpKernelContext* ctx,
            const char* ps_config_file, const int32_t global_batch_size,
            const int32_t num_replicas_in_sync);
  void config(const int32_t global_replica_id, tensorflow::OpKernelContext* ctx,
              const char* ps_config_file, const int32_t num_replicas_in_sync);
  void forward(const char* model_name, const int32_t table_id, const int32_t global_replica_id,
               const tensorflow::Tensor* values_tensor, tensorflow::Tensor* emb_vector_tensor,
               tensorflow::OpKernelContext* ctx);
  void record_hotness(const char* model_name, const int32_t table_id, const int32_t global_replica_id,
                      const tensorflow::Tensor* values_tensor, tensorflow::OpKernelContext* ctx);

  inline void report_avg() { coll_parameter_server_->report_avg(); }
  // for profiler
  inline void set_step_profile_value(const int global_replica_id, const uint64_t type,
                                     const double value) {
    if (coll_parameter_server_) {
      auto step = this->current_steps_for_each_replica_[global_replica_id] - 1;
      coll_parameter_server_->set_step_profile_value(global_replica_id, step, type, value);
    }
  }
  inline void add_epoch_profile_value(const int global_replica_id, const uint64_t type,
                                      const double value) {
    if (coll_parameter_server_) {
      auto step = this->current_steps_for_each_replica_[global_replica_id] - 1;
      coll_parameter_server_->add_epoch_profile_value(global_replica_id, step, type, value);
    }
  }

  void refresh(const int32_t global_replica_id, cudaStream_t stream = nullptr, bool foreground = false);
 private:
  void init_per_replica(const int32_t global_replica_id);

  LookupManager();
  bool initialized_;
  std::shared_ptr<parameter_server_config> ps_config;
  std::map<std::string, std::map<size_t, std::vector<std::shared_ptr<void>>>> h_values_map_;

  // for coll cache
  std::vector<size_t> current_steps_for_each_replica_;
  std::once_flag atomic_creation_flag_;
  std::shared_ptr<CollCacheParameterServer> coll_parameter_server_ = nullptr;

 public:
  std::vector<tensorflow::OpKernelContext*> tf_ctx_list;
  std::vector<std::thread> coll_refresh_thread;
  std::atomic<bool> *coll_refresh_ongoing = nullptr;
  std::vector<std::shared_ptr<coll_cache_lib::common::FreqRecorder>> coll_freq_recorder_list;
  std::vector<std::thread> coll_record_thread;
  sem_t record_send, record_done; size_t cur_num_key; const void* cur_key_ptr;
};

}  // namespace coll_cache_lib

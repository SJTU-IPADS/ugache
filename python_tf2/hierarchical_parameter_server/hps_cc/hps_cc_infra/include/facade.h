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
#include <cstdint>

#include "coll_cache_lib/profiler.h"
#include "coll_cache_lib/run_config.h"
#include "hps/inference_utils.hpp"
#include "lookup_manager.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace HierarchicalParameterServer {

using namespace HugeCTR;

class Facade final {
 private:
  Facade();
  ~Facade() = default;
  Facade(const Facade&) = delete;
  Facade& operator=(const Facade&) = delete;
  Facade(Facade&&) = delete;
  Facade& operator=(Facade&&) = delete;

  std::once_flag lookup_manager_init_once_flag_;
  std::shared_ptr<LookupManager> lookup_manager_;

 public:
  static Facade* instance();
  void operator delete(void*);
  void init(const int32_t global_replica_id, tensorflow::OpKernelContext* ctx,
            const char* ps_config_file, const int32_t global_batch_size,
            const int32_t num_replicas_in_sync);
  void forward(const char* model_name, const int32_t table_id, const int32_t global_replica_id,
               const tensorflow::Tensor* values_tensor, tensorflow::Tensor* emb_vector_tensor,
               tensorflow::OpKernelContext* ctx);
  void report_avg();
  void report_cache();
  parameter_server_config* ps_config;
  std::shared_ptr<coll_cache_lib::common::Profiler> profiler_;
  std::vector<size_t> current_steps_for_each_replica_;

  // for profiler
  inline void set_step_profile_value(const int global_replica_id, const int64_t type,
                                     double value) {
    if (ps_config->use_coll_cache)
      this->lookup_manager_->set_step_profile_value(global_replica_id, type, value);
    else {
      auto iter_key = current_steps_for_each_replica_[global_replica_id];
      if (type == coll_cache_lib::common::kLogL1TrainTime)
        current_steps_for_each_replica_[global_replica_id]++;
      auto key = iter_key * coll_cache_lib::common::RunConfig::num_device + global_replica_id;
      this->profiler_->LogStep(key, static_cast<coll_cache_lib::common::LogStepItem>(type), value);
    }
  }

  inline void add_epoch_profile_value(const int global_replica_id, const int64_t type,
                                      const double value) {
    this->lookup_manager_->add_epoch_profile_value(global_replica_id, type, value);
  }
};

}  // namespace HierarchicalParameterServer
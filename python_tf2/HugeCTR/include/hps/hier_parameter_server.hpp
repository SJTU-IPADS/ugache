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
#include <coll_cache_lib/facade.h>

#include <common.hpp>
#include <inference_utils.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coll_cache_lib/atomic_barrier.h"
#include "coll_cache_lib/common.h"
#include "coll_cache_lib/run_config.h"
#include "modelloader.hpp"

namespace coll_cache_lib {

class CollCacheParameterServer {
 public:
  using IdType = coll_cache_lib::common::IdType;
  using MemHandle = coll_cache_lib::common::MemHandle;
  using DataType = coll_cache_lib::common::DataType;
  using LogEpochItem = coll_cache_lib::common::LogEpochItem;
  using LogStepItem = coll_cache_lib::common::LogStepItem;
  virtual ~CollCacheParameterServer() = default;
  CollCacheParameterServer(const parameter_server_config& ps_config);
  CollCacheParameterServer(CollCacheParameterServer const&) = delete;
  CollCacheParameterServer& operator=(CollCacheParameterServer const&) = delete;
  void init_per_replica(int global_replica_id, coll_cache_lib::common::ContFreqBuf* freq_rank,
                        std::function<MemHandle(size_t)> gpu_mem_allocator, cudaStream_t stream);

  void lookup(int replica_id, const void* keys, size_t length, void* vectors,
              const std::string& model_name, size_t table_id, cudaStream_t cu_stream,
              uint64_t iter_key);
  inline void set_step_profile_value(int replica_id, uint64_t iter_key, uint64_t item, double val) {
    auto key = iter_key * coll_cache_lib::common::RunConfig::num_device + replica_id;
    this->coll_cache_ptr_->set_step_profile_value(key, static_cast<LogStepItem>(item), val);
  }
  inline void add_epoch_profile_value(int replica_id, uint64_t iter_key, uint64_t item,
                                      double val) {
    auto key = iter_key * coll_cache_lib::common::RunConfig::num_device + replica_id;
    this->coll_cache_ptr_->add_epoch_profile_value(key, static_cast<LogEpochItem>(item), val);
  }
  inline void refresh(int global_replica_id, coll_cache_lib::common::ContFreqBuf* freq_rank, cudaStream_t cu_stream = nullptr,
                      bool foreground = false) {
    auto stream = static_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
    this->coll_cache_ptr_->refresh(global_replica_id, freq_rank, stream, foreground);
  }

  inline parameter_server_config& ref_ps_config() { return ps_config_; }

  void report_avg();
  static void barrier();

 private:
  std::shared_ptr<coll_cache_lib::CollCache> coll_cache_ptr_;
  std::shared_ptr<IModelLoader> raw_data_holder;
  coll_cache_lib::common::DataType dtype = coll_cache_lib::common::kF32;

  parameter_server_config ps_config_;
};

}  // namespace coll_cache_lib
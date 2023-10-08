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

#include "lookup_manager.h"

#include "base/debug/logger.hpp"
#include "coll_cache_lib/atomic_barrier.h"
#include "coll_cache_lib/facade.h"
#include "coll_cache_lib/profiler.h"
#include "coll_cache_lib/logging.h"
#include "coll_cache_lib/timer.h"
#include "hier_parameter_server.hpp"
#include "inference_utils.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace coll_cache_lib {

void CriticalExec(coll_cache_lib::common::BarHandle barrier, int syncer, std::function<void()> func, int local_id) {
  for (int i = 0; i < syncer; i++) {
    if (i == local_id) { func(); }
    barrier->Wait();
  }
}

std::shared_ptr<LookupManager> LookupManager::Create() {
  return std::shared_ptr<LookupManager>(new LookupManager());
}

LookupManager::LookupManager() : initialized_{false} {}

void LookupManager::init(parameter_server_config& ps_config, int32_t global_batch_size,
                         int32_t num_replicas_in_sync) {
  initialized_ = true;
  HCTR_CHECK_HINT(global_batch_size > 0, "global_batch_size must be > 0.");
  HCTR_CHECK_HINT(num_replicas_in_sync > 0, "num_replicas_in_sync must be > 0.");
  HCTR_CHECK_HINT(global_batch_size % num_replicas_in_sync == 0,
                  "global_batch_size must be divisible by num_replicas_in_sync.");
  size_t local_batch_size = global_batch_size / num_replicas_in_sync;

  for (auto& inference_params : ps_config.inference_params_array) {
    sort(inference_params.deployed_devices.begin(), inference_params.deployed_devices.end());
    auto check = [](const std::vector<int>& vec) {
      for (size_t i{0}; i < vec.size(); ++i) {
        if (vec[i] != i) return false;
      }
      return true;
    };
    // 32 bit key is supported
    // HCTR_CHECK_HINT(inference_params.i64_input_key, "inference_params.i64_input_key must be
    // true.");
    HCTR_CHECK_HINT(inference_params.cross_worker_deployed_devices.size() == num_replicas_in_sync,
                    "inference_params.cross_worker_deployed_devices.size() must be equal to "
                    "num_replicas_in_sync.");
    HCTR_CHECK_HINT(
        check(inference_params.cross_worker_deployed_devices),
        "inference_params.cross_worker_deployed_devices should contain exactly from 0 to "
        "num_replicas_in_sync-1.");
    HCTR_CHECK_HINT(local_batch_size <= inference_params.max_batchsize,
                    "global_batch_size / num_replicas_in_sync must be <= max_batchsize configured "
                    "in ps_config.json.");
  }

  // Create the HPS for all models on all the deployed devices
  parameter_server_ = HierParameterServerBase::create(ps_config, ps_config.inference_params_array);
  current_steps_for_each_replica_.resize(num_replicas_in_sync, 0);

  this->coll_freq_recorder_list.resize(num_replicas_in_sync);
  // Initialie the resources for each model
  for (auto& inference_params : ps_config.inference_params_array) {
    // Create the lookup sessions on all the deployed devices
    std::map<size_t, std::shared_ptr<LookupSessionBase>> lookup_sessions;
    for (const auto& device_id : inference_params.deployed_devices) {
      inference_params.device_id = device_id;
      auto lookup_session = LookupSessionBase::create(inference_params, embedding_cache);
      coll_freq_recorder_list[device_id] = lookup_session->freq_recorder_;
    }
    lookup_session_map_.emplace(inference_params.model_name, lookup_sessions);

    // Allocate the host buffer per table per replica to support concurrent query
    std::map<size_t, std::vector<std::shared_ptr<void>>> h_values;
    for (const auto& device_id : inference_params.deployed_devices) {
      std::vector<std::shared_ptr<void>> h_values_per_table;
      for (size_t table_id = 0; table_id < inference_params.embedding_vecsize_per_table.size();
           ++table_id) {
        size_t capacity = inference_params.max_batchsize *
                          inference_params.maxnum_catfeature_query_per_table_per_sample[table_id];
        h_values_per_table.emplace_back(std::shared_ptr<size_t>(new size_t[capacity]));
      }
      h_values.emplace(device_id, h_values_per_table);
    }
    h_values_map_.emplace(inference_params.model_name, h_values);
  }
  this->tf_ctx_list.resize(num_replicas_in_sync);
  this->coll_refresh_thread.resize(num_replicas_in_sync);
  this->coll_record_thread.resize(num_replicas_in_sync);
  this->coll_refresh_ongoing = new std::atomic<bool>[num_replicas_in_sync];
  for (decltype(num_replicas_in_sync) i = 0; i < num_replicas_in_sync; i++) {
    this->coll_refresh_ongoing[i].store(false);
  }
  {
    LOG(INFO) << "replica " << global_replica_id << " calling init pre replica\n";
    init_per_replica(global_replica_id);
    LOG(INFO) << "init per replica done\n";
  }
}

void LookupManager::forward(const std::string& model_name, int32_t table_id,
                            int32_t global_replica_id, size_t num_keys, size_t emb_vec_size,
                            const void* values_ptr, void* emb_vector_ptr) {
  if (coll_parameter_server_) {
    cudaStream_t stream =
        *reinterpret_cast<const cudaStream_t*>(this->tf_ctx_list[global_replica_id]
                                                   ->op_device_context()
                                                   ->stream()
                                                   ->implementation()
                                                   ->GpuStreamMemberHack());
    // HCTR_LOG_S(ERROR, WORLD) << "cp taks time " << t_cp << ",record taks time " << record <<
    // "\n";
    if (coll_parameter_server_->ref_ps_config().coll_cache_enable_refresh) {
      cur_key_ptr = values_ptr; cur_num_key = num_keys; sem_post(&record_send);
    }

    coll_parameter_server_->lookup(global_replica_id, values_ptr, num_keys, emb_vector_ptr,
                                   model_name, table_id, stream,
                                   this->current_steps_for_each_replica_[global_replica_id]);
    if (coll_parameter_server_->ref_ps_config().coll_cache_enable_refresh &&
        this->current_steps_for_each_replica_[global_replica_id] ==
            coll_parameter_server_->ref_ps_config().coll_cache_refresh_iter) {
      if (coll_refresh_ongoing[global_replica_id].load() == false) {
        coll_refresh_ongoing[global_replica_id].store(true);
        if (coll_refresh_thread[global_replica_id].joinable()) {
          coll_refresh_thread[global_replica_id].join();
        }
        coll_refresh_thread[global_replica_id] = std::thread([this, global_replica_id, stream]() {
          refresh(global_replica_id, stream, false /*foreground*/);
          // refresh(global_replica_id, stream, true);
          coll_refresh_ongoing[global_replica_id].store(false);
        });
        // coll_refresh_thread[global_replica_id].join();
      } else {
        HCTR_LOG_S(ERROR, WORLD) << "skip refresh due to refresh already ongoing";
      }
    }
    if (coll_parameter_server_->ref_ps_config().coll_cache_enable_refresh) {
      sem_wait(&record_done);
    }
    this->current_steps_for_each_replica_[global_replica_id]++;
    return;
  }
  this->current_steps_for_each_replica_[global_replica_id]++;
}

void LookupManager::init_per_replica(const int32_t global_replica_id) {
  initialized_ = true;
  const parameter_server_config ps_config = parameter_server_->ref_ps_config();
  const int32_t num_replicas_in_sync =
      ps_config.inference_params_array[0].cross_worker_deployed_devices.size();

  // Create the HPS for all models on all the deployed devices
  std::vector<uint32_t> rank_vec, freq_vec;
  uint32_t *rank_ptr = nullptr, *freq_ptr = nullptr;

  std::call_once(this->atomic_creation_flag_, [&]() {
    HCTR_CHECK_HINT(this->lookup_session_map_.size() == 1, "coll cache supports only 1 model");
    coll_parameter_server_ = std::make_shared<CollCacheParameterServer>(ps_config);
    // this->_tensorflow_ctx_list.resize(num_replicas_in_sync);
  });
  LOG(INFO) << "replica " << global_replica_id
                          << " waits for coll ps creation barrier\n";
  coll_parameter_server_->barrier();

  CriticalExec(coll_cache_lib::common::AnonymousBarrier::_global_instance, 
    ps_config.inference_params_array[0].cross_worker_deployed_devices.size(),
    [this, global_replica_id](){
      HCTR_LOG_S(ERROR, WORLD) << "replica " << global_replica_id << " preparing frequency\n";
      this->coll_freq_recorder_list[global_replica_id]->LocalCombineToShared();
    }, global_replica_id);
  coll_cache_lib::common::ContFreqBuf* freq_rank = nullptr;
  if (global_replica_id == 0) {
    auto freq_recorder = this->lookup_session_map_.begin()->second[0]->freq_recorder_;
    freq_recorder->GlobalCombine();
    freq_recorder->Sort();
    freq_rank = freq_recorder->global_cont_freq_buf;
  }
  if (ps_config.use_multi_worker || global_replica_id == 0) {
    lookup_session_map_.clear();
    // h_values_map_.clear();
  }
  coll_parameter_server_->barrier();

  std::function<coll_cache_lib::MemHandle(size_t)> gpu_mem_allocator =
      [&ctx = tf_ctx_list[global_replica_id],
       global_replica_id](size_t nbytes) -> coll_cache_lib::MemHandle {
    if (nbytes == 0) {
      HCTR_LOG_S(ERROR, WORLD) << "allocating 0 cuda memory?\n";
    }
    auto handle = std::make_shared<HPSMemHandle>();
    TF_CHECK_OK(ctx->allocate_temp(tensorflow::DataType::DT_UINT8,
                                   tensorflow::TensorShape({(long)nbytes}),
                                   &(handle->tensor_hold)));
    handle->nbytes_ = nbytes;
    if (nbytes >= 1 << 21) {
      HCTR_LOG_S(ERROR, WORLD) << global_replica_id << " allocated " << nbytes << " at "
                               << handle->ptr() << "\n";
    }
    return handle;
  };

  CollCacheParameterServer* ps_ptr =
      reinterpret_cast<CollCacheParameterServer*>(coll_parameter_server_.get());

  CHECK(ps_config.inference_params_array.size() == 1);

  LOG(INFO) << "replica " << global_replica_id << " calling init per replica\n";
  cudaStream_t stream;
  stream = *reinterpret_cast<const cudaStream_t*>(this->tf_ctx_list[global_replica_id]
                                                      ->op_device_context()
                                                      ->stream()
                                                      ->implementation()
                                                      ->GpuStreamMemberHack());
  // stream = tensorflow::GetGpuStream(this->tf_ctx_list[global_replica_id]);
  ps_ptr->init_per_replica(global_replica_id, freq_rank, gpu_mem_allocator, stream);
  LOG(INFO) << "replica " << global_replica_id
                          << " calling init per replica done, doing barrier\n";
  coll_parameter_server_->barrier();
  LOG(INFO) << "replica " << global_replica_id
                          << " calling init per replica done, doing barrier done\n";
  sem_init(&record_send, 0, 0);
  sem_init(&record_done, 0, 0);
  if (ps_config.coll_cache_enable_refresh) {
    this->coll_record_thread[global_replica_id] = std::thread([this, global_replica_id](){
      size_t per_key_size = coll_parameter_server_->ref_ps_config().inference_params_array[0].i64_input_key ? 8 : 4;
      void* h_values = h_values_map_.begin()->second.find(global_replica_id)->second.begin()->get();
      while(true) {
        sem_wait(&record_send);
        coll_cache_lib::common::Timer t1;
        if (coll_parameter_server_->ref_ps_config().coll_cache_enable_refresh) {
          size_t num_keys_to_record = cur_num_key * 0.05;
          cudaMemcpy(h_values, cur_key_ptr, num_keys_to_record * per_key_size, cudaMemcpyDeviceToHost);
          if (coll_parameter_server_->ref_ps_config().inference_params_array[0].i64_input_key) {
            coll_freq_recorder_list[global_replica_id]->Record(
                reinterpret_cast<const coll_cache_lib::common::Id64Type*>(h_values),
                num_keys_to_record);
          } else {
            coll_freq_recorder_list[global_replica_id]->Record(
                reinterpret_cast<const coll_cache_lib::common::IdType*>(h_values), num_keys_to_record);
          }
        }
        double record = t1.Passed();
        set_step_profile_value(global_replica_id, coll_cache_lib::common::kLogL1ConvertTime, record);
        sem_post(&record_done);
      }
    });
  }
}

void LookupManager::refresh(const int32_t global_replica_id, cudaStream_t stream, bool foreground) {

  CriticalExec(coll_cache_lib::common::AnonymousBarrier::_refresh_instance, 
    coll_parameter_server_->ref_ps_config().inference_params_array[0].cross_worker_deployed_devices.size(),
    [this, global_replica_id](){
      HCTR_LOG_S(ERROR, WORLD) << "replica " << global_replica_id << " preparing frequency\n";
      this->coll_freq_recorder_list[global_replica_id]->LocalCombineToShared();
    }, global_replica_id);
  coll_cache_lib::common::ContFreqBuf* freq_rank = nullptr;
  if (global_replica_id == 0) {
    auto freq_recorder = this->coll_freq_recorder_list[0];
    freq_recorder->GlobalCombine();
    freq_recorder->Sort();
    freq_rank = freq_recorder->global_cont_freq_buf;
  }
  coll_parameter_server_->refresh(global_replica_id, freq_rank, stream, foreground);
}

void LookupManager::report_avg() {
  if (coll_parameter_server_) {
    coll_parameter_server_->report_avg();
  }
}
}  // namespace coll_cache_lib
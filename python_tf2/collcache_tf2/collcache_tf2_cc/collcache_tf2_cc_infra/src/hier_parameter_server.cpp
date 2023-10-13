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

#include <cmath>
#include <filesystem>
#include <hier_parameter_server.hpp>
#include <inference_utils.hpp>
#include <modelloader.hpp>
#include <regex>
#include <vector>

#include "coll_cache_lib/atomic_barrier.h"
#include "coll_cache_lib/common.h"
#include "coll_cache_lib/device.h"
#include "coll_cache_lib/facade.h"
#include "logging.hpp"
#include "coll_cache_lib/run_config.h"

namespace coll_cache_lib {

// using namespace coll_cache_lib::common;

CollCacheParameterServer::CollCacheParameterServer(std::shared_ptr<parameter_server_config> ps_config)
    : ps_config_(ps_config) {
  COLL_LOG(INFO) << "====================================================HPS Coll "
               "Create====================================================\n";
  const std::vector<InferenceParams>& inference_params_array = ps_config_->inference_params_array;
  COLL_CHECK(ps_config_->embedding_vec_size_.size() == inference_params_array.size()) 
      << "Wrong input: The size of parameter server parameters are not correct.";
  COLL_CHECK(inference_params_array.size() == 1) <<  "Coll cache only support single model for now";
  auto& inference_params = inference_params_array[0];
  COLL_CHECK(inference_params_array[0].sparse_model_files.size() == 1)
      <<  "Coll cache only support single sparse file for now";

  // Connect to volatile database.
  // Create input file stream to read the embedding file
  COLL_CHECK(ps_config_->embedding_vec_size_[inference_params.model_name].size() == inference_params.sparse_model_files.size())
      << "Wrong input: The number of embedding tables in network json file for model "
      << inference_params.model_name
      << " doesn't match the size of 'sparse_model_files' in configuration.";

  coll_cache_lib::common::RunConfig::cache_percentage = inference_params.cache_size_percentage;
  // coll_cache_lib::common::RunConfig::cache_policy = coll_cache_lib::common::kRepCache;
  // coll_cache_lib::common::RunConfig::cache_policy = coll_cache_lib::common::kCliquePart;
  coll_cache_lib::common::RunConfig::cache_policy =
      (coll_cache_lib::common::CachePolicy)ps_config_->coll_cache_policy;
  coll_cache_lib::common::RunConfig::cross_process = false;
  coll_cache_lib::common::RunConfig::device_id_list =
      inference_params.cross_worker_deployed_devices;
  coll_cache_lib::common::RunConfig::num_device =
      inference_params.cross_worker_deployed_devices.size();
  coll_cache_lib::common::RunConfig::cross_process = ps_config_->use_multi_worker;
  coll_cache_lib::common::RunConfig::num_global_step_per_epoch =
      ps_config_->iteration_per_epoch * coll_cache_lib::common::RunConfig::num_device;
  coll_cache_lib::common::RunConfig::num_epoch = ps_config_->epoch;
  coll_cache_lib::common::RunConfig::num_total_item = inference_params.max_vocabulary_size.front();

  COLL_LOG(ERROR)
      << "coll ps creation, with "
      << ps_config_->inference_params_array[0].cross_worker_deployed_devices.size()
      << " devices, using policy " << coll_cache_lib::common::RunConfig::cache_policy << "\n";
  coll_cache_lib::common::AnonymousBarrier::_global_instance->SetWorker(coll_cache_lib::common::RunConfig::num_device);
  this->coll_cache_ptr_ = std::make_shared<coll_cache_lib::CollCache>(
      nullptr, coll_cache_lib::common::AnonymousBarrier::_global_instance);
  COLL_LOG(ERROR) << "coll ps creation done\n";
}

void CollCacheParameterServer::init_per_replica(int global_replica_id,
                                                ContFreqBuf* freq_rank,
                                                std::function<MemHandle(size_t)> gpu_mem_allocator,
                                                cudaStream_t cu_stream) {
  auto& inference_params = ps_config_->inference_params_array[0];
  {
    IModelLoader* rawreader = ModelLoader<uint32_t, float>::CreateLoader();
    IModelLoader::preserved_model_loader = std::shared_ptr<IModelLoader>(rawreader);
    // Create input file stream to read the embedding file
    for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
      if (global_replica_id == 0) {
        rawreader->master_load(inference_params.sparse_model_files[j]);
        CollCacheParameterServer::barrier();
      } else {
        CollCacheParameterServer::barrier();
        rawreader->slave_load(inference_params.sparse_model_files[j]);
      }
      if (inference_params.use_multi_worker) {
        CollCacheParameterServer::barrier();
      }
    }
  }
  raw_data_holder = IModelLoader::preserved_model_loader;

  void* cpu_data = raw_data_holder->getvectors();
  double cache_percentage = ps_config_->inference_params_array[0].cache_size_percentage;
  size_t dim = ps_config_->inference_params_array[0].embedding_vecsize_per_table[0];
  // hps may be used in hugectr or tensorflow, so we don't know how to allocate memory;
  size_t num_key = raw_data_holder->getkeycount();
  COLL_CHECK(num_key == ps_config_->inference_params_array[0].max_vocabulary_size[0]) << 
                  "num key from file must equal with max vocabulary:" << num_key;
  auto stream = reinterpret_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
  COLL_LOG(ERROR) << "Calling build_v2\n";

  {
    int value;
    cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem,
                           coll_cache_lib::common::RunConfig::device_id_list[global_replica_id]);
    COLL_LOG(ERROR) << "cudaDevAttrCanUseHostPointerForRegisteredMem is " << value << "\n";
  }

  this->coll_cache_ptr_->build_v2(global_replica_id, freq_rank, num_key, gpu_mem_allocator, cpu_data,
                                  dtype, dim, cache_percentage, stream);
}

void CollCacheParameterServer::lookup(int replica_id, const void* keys, size_t length, void* output,
                                      const std::string& model_name, size_t table_id,
                                      cudaStream_t cu_stream, uint64_t iter_key) {
  auto stream = reinterpret_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
  auto step_key = iter_key * coll_cache_lib::common::RunConfig::num_device + replica_id;
  this->coll_cache_ptr_->lookup(replica_id, reinterpret_cast<const uint32_t*>(keys), length, output,
                                stream, step_key);
}

void CollCacheParameterServer::barrier() {
  coll_cache_lib::common::AnonymousBarrier::_global_instance->Wait();
}

void CollCacheParameterServer::report_avg() { this->coll_cache_ptr_->report_avg(); }
}  // namespace coll_cache_lib
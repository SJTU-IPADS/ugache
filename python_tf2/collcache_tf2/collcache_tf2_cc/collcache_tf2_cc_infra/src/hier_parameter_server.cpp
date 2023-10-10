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
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if (inference_params_array[i].volatile_db != inference_params_array[0].volatile_db ||
        inference_params_array[i].persistent_db != inference_params_array[0].persistent_db) {
      COLL_LOG(FATAL) << 
          "Inconsistent database setup. coll_cache_lib paramter server does currently not support hybrid "
          "database deployment.";
    }
  }
  if (ps_config_->embedding_vec_size_.size() != inference_params_array.size() ||
      ps_config_->default_emb_vec_value_.size() != inference_params_array.size()) {
    COLL_LOG(FATAL) << 
                   "Wrong input: The size of parameter server parameters are not correct.";
  }

  if (inference_params_array.size() != 1) {
    COLL_LOG(FATAL) <<  "Coll cache only support single model for now";
  }
  auto& inference_params = inference_params_array[0];
  if (inference_params_array[0].sparse_model_files.size() != 1) {
    COLL_LOG(FATAL) <<  "Coll cache only support single sparse file for now";
  }

  // Connect to volatile database.
  // Create input file stream to read the embedding file
  if (ps_config_->embedding_vec_size_[inference_params.model_name].size() !=
      inference_params.sparse_model_files.size()) {
    COLL_LOG(FATAL) << 
                   ("Wrong input: The number of embedding tables in network json file for model " +
                       inference_params.model_name +
                       " doesn't match the size of 'sparse_model_files' in configuration.");
  }

  {
    IModelLoader* rawreader = ModelLoader<uint32_t, float>::CreateLoader(DBTableDumpFormat_t::Raw);
    IModelLoader::preserved_model_loader = std::shared_ptr<IModelLoader>(rawreader);
    // Create input file stream to read the embedding file
    for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
      if (ps_config_->embedding_vec_size_[inference_params.model_name].size() !=
          inference_params.sparse_model_files.size()) {
        COLL_LOG(FATAL) <<
                      "Wrong input: The number of embedding tables in network json file for model " +
                          inference_params.model_name +
                          " doesn't match the size of 'sparse_model_files' in configuration.";
      }
      // Get raw format model loader
      rawreader->load(inference_params.embedding_table_names[j],
                      inference_params.sparse_model_files[j]);
      if (inference_params.use_multi_worker) {
        CollCacheParameterServer::barrier();
      }
      // const std::string tag_name = make_tag_name(
      //     inference_params.model_name, ps_config_->emb_table_name_[inference_params.model_name][j]);
      size_t num_key = rawreader->getkeycount();
      const size_t embedding_size = ps_config_->embedding_vec_size_[inference_params.model_name][j];
      // Populate volatile database(s).
      // if (volatile_db_) {
      //   const size_t volatile_capacity = volatile_db_->capacity(tag_name);
      //   const size_t volatile_cache_amount =
      //       (num_key <= volatile_capacity)
      //           ? num_key
      //           : static_cast<size_t>(
      //                 volatile_db_cache_rate_ * static_cast<double>(volatile_capacity) + 0.5);

      //   if (dynamic_cast<DirectMapBackend<uint32_t>*>(volatile_db_.get()) &&
      //       rawreader->is_mock == false) {
      //     auto keys_ptr = (const uint32_t*)(rawreader->getkeys());
      //     #pragma omp parallel for
      //     for (uint32_t i = 0; i < volatile_cache_amount; i++) {
      //       HCTR_CHECK_HINT(keys_ptr[i] == i, "direct backend requries continious source");
      //     }
      //   }
      //   HCTR_CHECK(volatile_db_->insert(tag_name, volatile_cache_amount,
      //                                   reinterpret_cast<const uint32_t*>(rawreader->getkeys()),
      //                                   reinterpret_cast<const char*>(rawreader->getvectors()),
      //                                   embedding_size * sizeof(float)));
      //   volatile_db_->synchronize();
      //   HCTR_LOG_S(INFO, WORLD) << "Table: " << tag_name << "; cached " << volatile_cache_amount
      //                           << " / " << num_key << " embeddings in volatile database ("
      //                           << volatile_db_->get_name()
      //                           << "); load: " << volatile_db_->size(tag_name) << " / "
      //                           << volatile_capacity << " (" << std::fixed << std::setprecision(2)
      //                           << (static_cast<double>(volatile_db_->size(tag_name)) * 100.0 /
      //                               static_cast<double>(volatile_capacity))
      //                           << "%)." << std::endl;
      // }
    }
  }

  // hps assumes disk key file is long-long
  raw_data_holder = IModelLoader::preserved_model_loader;
  // Get raw format model loader
  size_t num_key = raw_data_holder->getkeycount();
  // const size_t embedding_size = ps_config_->embedding_vec_size_[inference_params.model_name][0];
  // Populate volatile database(s).
  // auto val_ptr = reinterpret_cast<const char*>(raw_data_holder->getvectors());

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
  coll_cache_lib::common::RunConfig::num_total_item = num_key;

  COLL_LOG(ERROR)
      << "coll ps creation, with "
      << ps_config_->inference_params_array[0].cross_worker_deployed_devices.size()
      << " devices, using policy " << coll_cache_lib::common::RunConfig::cache_policy << "\n";
  this->coll_cache_ptr_ = std::make_shared<coll_cache_lib::CollCache>(
      nullptr, coll_cache_lib::common::AnonymousBarrier::_global_instance);
  COLL_LOG(ERROR) << "coll ps creation done\n";
}

#ifdef DEAD_CODE
void CollCacheParameterServer::init_per_replica(int global_replica_id,
                                                IdType* ranking_nodes_list_ptr,
                                                IdType* ranking_nodes_freq_list_ptr,
                                                std::function<MemHandle(size_t)> gpu_mem_allocator,
                                                cudaStream_t cu_stream) {
  void* cpu_data = raw_data_holder->getvectors();
  double cache_percentage = ps_config_->inference_params_array[0].cache_size_percentage;
  size_t dim = ps_config_->inference_params_array[0].embedding_vecsize_per_table[0];
  // hps may be used in hugectr or tensorflow, so we don't know how to allocate memory;
  size_t num_key = raw_data_holder->getkeycount();
  HCTR_CHECK_HINT(num_key == ps_config_->inference_params_array[0].max_vocabulary_size[0],
                  "num key from file must equal with max vocabulary: %d", num_key);
  auto stream = reinterpret_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
  HCTR_LOG(ERROR, WORLD, "Calling build_v2\n");

  {
    int value;
    cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem,
                           coll_cache_lib::common::RunConfig::device_id_list[global_replica_id]);
    HCTR_LOG_S(ERROR, WORLD) << "cudaDevAttrCanUseHostPointerForRegisteredMem is " << value << "\n";
  }

  this->coll_cache_ptr_->build_v2(global_replica_id, ranking_nodes_list_ptr,
                                  ranking_nodes_freq_list_ptr, num_key, gpu_mem_allocator, cpu_data,
                                  dtype, dim, cache_percentage, stream);
}
#endif
void CollCacheParameterServer::init_per_replica(int global_replica_id,
                                                ContFreqBuf* freq_rank,
                                                std::function<MemHandle(size_t)> gpu_mem_allocator,
                                                cudaStream_t cu_stream) {
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
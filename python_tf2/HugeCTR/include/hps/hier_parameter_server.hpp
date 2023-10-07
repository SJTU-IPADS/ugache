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
#include <hps/database_backend.hpp>
#include <hps/embedding_cache_base.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/inference_utils.hpp>
#include <hps/memory_pool.hpp>
#include <hps/message.hpp>
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
#include "hps/modelloader.hpp"

namespace HugeCTR {

template <typename TypeHashKey>
class HierParameterServer : public HierParameterServerBase {
  using DataType = coll_cache_lib::common::DataType;
  using Tensor = coll_cache_lib::common::Tensor;
  using TensorPtr = coll_cache_lib::common::TensorPtr;
  using RunConfig = coll_cache_lib::common::RunConfig;

 public:
  virtual ~HierParameterServer();
  HierParameterServer(const parameter_server_config& ps_config,
                      std::vector<InferenceParams>& inference_params_array);
  HierParameterServer(HierParameterServer const&) = delete;
  HierParameterServer& operator=(HierParameterServer const&) = delete;

  virtual void update_database_per_model(const InferenceParams& inference_params);
  virtual void create_embedding_cache_per_model(InferenceParams& inference_params);
  virtual void init_ec(InferenceParams& inference_params,
                       std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map);
  virtual void destory_embedding_cache_per_model(const std::string& model_name);
  virtual std::shared_ptr<EmbeddingCacheBase> get_embedding_cache(const std::string& model_name,
                                                                  int device_id);

  virtual void erase_model_from_hps(const std::string& model_name);

  virtual void* apply_buffer(const std::string& model_name, int device_id,
                             CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER);
  virtual void free_buffer(void* p);
  virtual void lookup(const void* h_keys, size_t length, float* h_vectors,
                      const std::string& model_name, size_t table_id);
  virtual void refresh_embedding_cache(const std::string& model_name, int device_id);
  virtual void insert_embedding_cache(size_t table_id,
                                      std::shared_ptr<EmbeddingCacheBase> embedding_cache,
                                      EmbeddingCacheWorkspace& workspace_handler,
                                      cudaStream_t stream);
  virtual void parse_hps_configuraion(const std::string& hps_json_config_file);
  virtual const parameter_server_config& ref_ps_config();
  virtual std::map<std::string, InferenceParams> get_hps_model_configuration_map();
  virtual double report_cache_intersect();
  virtual std::vector<double> report_access_overlap();

 private:
  const std::string HPSCacheKeyShmName =
      std::string("hps_cache_key_") + std::string(std::getenv("USER"));
  const std::string HPSCacheAccessCountShmName =
      std::string("hps_cache_access_cnt_") + std::string(std::getenv("USER"));
  // Parameter server configuration
  parameter_server_config ps_config_;
  // Database layers for multi-tier cache/lookup.
  std::unique_ptr<VolatileBackend<TypeHashKey>> volatile_db_;
  double volatile_db_cache_rate_;
  bool volatile_db_cache_missed_embeddings_;
  std::unique_ptr<PersistentBackend<TypeHashKey>> persistent_db_;
  // Realtime data ingestion.
  std::unique_ptr<MessageSource<TypeHashKey>> volatile_db_source_;
  std::unique_ptr<MessageSource<TypeHashKey>> persistent_db_source_;
  // Buffer pool that manages workspace and refreshspace of embedding caches
  std::shared_ptr<ManagerPool> buffer_pool_;
  // Configurations for memory pool
  inference_memory_pool_size_config memory_pool_config_;
  // Embedding caches of all models deployed on all devices, e.g., {"dcn": {0: dcn_embedding_cache0,
  // 1: dcnembedding_cache1}}
  std::map<std::string, std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>> model_cache_map_;
  // model configuration of all models deployed on HPS, e.g., {"dcn": dcn_inferenceParamesStruct}
  std::map<std::string, InferenceParams> inference_params_map_;
};

// template <typename TypeHashKey>
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
#ifdef DEAD_CODE
  void init_per_replica(int global_replica_id, IdType* ranking_nodes_list_ptr,
                        IdType* ranking_nodes_freq_list_ptr,
                        std::function<MemHandle(size_t)> gpu_mem_allocator, cudaStream_t stream);
#endif
  void init_per_replica(int global_replica_id, coll_cache_lib::common::ContFreqBuf* freq_rank,
                        std::function<MemHandle(size_t)> gpu_mem_allocator, cudaStream_t stream);

  // virtual void update_database_per_model(const InferenceParams& inference_params);
  // virtual void create_embedding_cache_per_model(InferenceParams& inference_params);
  // virtual void init_ec(InferenceParams& inference_params,
  //                      std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>
  //                      embedding_cache_map);
  // virtual void destory_embedding_cache_per_model(const std::string& model_name);
  // virtual std::shared_ptr<EmbeddingCacheBase> get_embedding_cache(const std::string& model_name,
  //                                                                 int device_id);

  // virtual void erase_model_from_hps(const std::string& model_name);

  // virtual void* apply_buffer(const std::string& model_name, int device_id,
  //                            CACHE_SPACE_TYPE cache_type = CACHE_SPACE_TYPE::WORKER);
  // virtual void free_buffer(void* p);
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
#ifdef DEAD_CODE
  inline void refresh(int global_replica_id, IdType* ranking_nodes_list_ptr,
                      IdType* ranking_nodes_freq_list_ptr, cudaStream_t cu_stream = nullptr,
                      bool foreground = false) {
    auto stream = static_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
    this->coll_cache_ptr_->refresh(global_replica_id, ranking_nodes_list_ptr,
                                   ranking_nodes_freq_list_ptr, stream, foreground);
  }
#endif
  inline void refresh(int global_replica_id, coll_cache_lib::common::ContFreqBuf* freq_rank, cudaStream_t cu_stream = nullptr,
                      bool foreground = false) {
    auto stream = static_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
    this->coll_cache_ptr_->refresh(global_replica_id, freq_rank, stream, foreground);
  }

  inline parameter_server_config& ref_ps_config() { return ps_config_; }

  void report_avg();
  static void barrier();
  // virtual void refresh_embedding_cache(const std::string& model_name, int device_id);
  // virtual void insert_embedding_cache(size_t table_id,
  //                                     std::shared_ptr<EmbeddingCacheBase> embedding_cache,
  //                                     EmbeddingCacheWorkspace& workspace_handler,
  //                                     cudaStream_t stream);
  // virtual void parse_hps_configuraion(const std::string& hps_json_config_file);
  // virtual std::map<std::string, InferenceParams> get_hps_model_configuration_map();

 private:
  std::shared_ptr<coll_cache_lib::CollCache> coll_cache_ptr_;
  std::shared_ptr<IModelLoader> raw_data_holder;
  coll_cache_lib::common::DataType dtype = coll_cache_lib::common::kF32;

  // Parameter server configuration
  parameter_server_config ps_config_;
  // // Database layers for multi-tier cache/lookup.
  // std::unique_ptr<VolatileBackend<coll_cache_lib::common::IdType>> volatile_db_;
  // double volatile_db_cache_rate_;
  // bool volatile_db_cache_missed_embeddings_;
  // std::unique_ptr<PersistentBackend<coll_cache_lib::common::IdType>> persistent_db_;
  // // Realtime data ingestion.
  // std::unique_ptr<MessageSource<coll_cache_lib::common::IdType>> volatile_db_source_;
  // std::unique_ptr<MessageSource<coll_cache_lib::common::IdType>> persistent_db_source_;
  // Buffer pool that manages workspace and refreshspace of embedding caches
  // std::shared_ptr<ManagerPool> buffer_pool_;
  // Configurations for memory pool
  // inference_memory_pool_size_config memory_pool_config_;
  // Embedding caches of all models deployed on all devices, e.g., {"dcn": {0: dcn_embedding_cache0,
  // 1: dcnembedding_cache1}}
  // std::map<std::string, std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>>> model_cache_map_;
  // model configuration of all models deployed on HPS, e.g., {"dcn": dcn_inferenceParamesStruct}
  // std::map<std::string, InferenceParams> inference_params_map_;
};

}  // namespace HugeCTR
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
#include <hps/direct_map_backend.hpp>
#include <hps/hash_map_backend.hpp>
#include <hps/hier_parameter_server.hpp>
#include <hps/inference_utils.hpp>
#include <hps/kafka_message.hpp>
#include <hps/modelloader.hpp>
#include <hps/redis_backend.hpp>
#include <hps/rocksdb_backend.hpp>
#include <regex>
#include <vector>

#include "base/debug/logger.hpp"
#include "coll_cache_lib/atomic_barrier.h"
#include "coll_cache_lib/common.h"
#include "coll_cache_lib/device.h"
#include "coll_cache_lib/facade.h"
#include "coll_cache_lib/run_config.h"

namespace HugeCTR {

using namespace coll_cache_lib::common;

std::string HierParameterServerBase::make_tag_name(const std::string& model_name,
                                                   const std::string& embedding_table_name,
                                                   const bool check_arguments) {
  static const std::regex syntax{"[a-zA-Z0-9_\\-]{1,120}"};
  if (check_arguments) {
    HCTR_CHECK_HINT(std::regex_match(model_name, syntax), "The provided 'model_name' is invalid!");
    HCTR_CHECK_HINT(std::regex_match(embedding_table_name, syntax),
                    "The provided 'embedding_table_name' is invalid!");
  }

  std::ostringstream os;
  os << PS_EMBEDDING_TABLE_TAG_PREFIX << '.';
  os << model_name << '.' << embedding_table_name;
  return os.str();
}

std::shared_ptr<HierParameterServerBase> HierParameterServerBase::create(
    const parameter_server_config& ps_config,
    std::vector<InferenceParams>& inference_params_array) {
  HCTR_CHECK_HINT(inference_params_array.size() > 0, "inference_params_array should not be empty");
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if (inference_params_array[i].i64_input_key != inference_params_array[0].i64_input_key) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Inconsistent key types for different models. Parameter server does not "
                     "support hybrid key types.");
    }
  }
  if (inference_params_array[0].i64_input_key) {
    return std::make_shared<HierParameterServer<long long>>(ps_config, inference_params_array);
  } else {
    return std::make_shared<HierParameterServer<unsigned int>>(ps_config, inference_params_array);
  }
}

std::shared_ptr<HierParameterServerBase> HierParameterServerBase::create(
    const std::string& hps_json_config_file) {
  parameter_server_config ps_config{hps_json_config_file};
  return HierParameterServerBase::create(ps_config, ps_config.inference_params_array);
}

HierParameterServerBase::~HierParameterServerBase() = default;

template <typename TypeHashKey>
const parameter_server_config& HierParameterServer<TypeHashKey>::ref_ps_config() {
  return this->ps_config_;
}

template <typename TypeHashKey>
HierParameterServer<TypeHashKey>::HierParameterServer(
    const parameter_server_config& ps_config, std::vector<InferenceParams>& inference_params_array)
    : HierParameterServerBase(), ps_config_(ps_config) {
  HCTR_PRINT(INFO,
             "====================================================HPS "
             "Create====================================================\n");
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if (inference_params_array[i].volatile_db != inference_params_array[0].volatile_db ||
        inference_params_array[i].persistent_db != inference_params_array[0].persistent_db) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Inconsistent database setup. HugeCTR paramter server does currently not support hybrid "
          "database deployment.");
    }
  }
  if (ps_config_.embedding_vec_size_.size() != inference_params_array.size() ||
      ps_config_.default_emb_vec_value_.size() != inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The size of parameter server parameters are not correct.");
  }

  // Connect to volatile database.
  {
    const auto& conf = inference_params_array[0].volatile_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No volatile database.

      case DatabaseType_t::DirectMap:
        HCTR_CHECK_HINT(inference_params_array.size() == 1,
                        "direct map backend supports only 1 model");
        HCTR_LOG_S(INFO, WORLD) << "Creating DirectMap CPU database backend..." << std::endl;
        volatile_db_ = std::make_unique<DirectMapBackend<TypeHashKey>>(
            conf.max_get_batch_size, conf.max_set_batch_size, conf.overflow_margin,
            conf.overflow_policy, conf.overflow_resolution_target);
        break;

      case DatabaseType_t::HashMap:
      case DatabaseType_t::ParallelHashMap:
        HCTR_LOG_S(INFO, WORLD) << "Creating HashMap CPU database backend..." << std::endl;
        volatile_db_ = std::make_unique<HashMapBackend<TypeHashKey>>(
            conf.num_partitions, conf.allocation_rate, conf.max_get_batch_size,
            conf.max_set_batch_size, conf.overflow_margin, conf.overflow_policy,
            conf.overflow_resolution_target);
        break;

      case DatabaseType_t::RedisCluster:
        HCTR_LOG_S(INFO, WORLD) << "Creating RedisCluster backend..." << std::endl;
        volatile_db_ = std::make_unique<RedisClusterBackend<TypeHashKey>>(
            conf.address, conf.user_name, conf.password, conf.num_partitions,
            conf.max_get_batch_size, conf.max_set_batch_size, conf.refresh_time_after_fetch,
            conf.overflow_margin, conf.overflow_policy, conf.overflow_resolution_target);
        break;

      default:
        HCTR_DIE("Selected backend (volatile_db.type = %d) is not supported!", conf.type);
        break;
    }
    volatile_db_cache_rate_ = conf.initial_cache_rate;
    volatile_db_cache_missed_embeddings_ = conf.cache_missed_embeddings;
    HCTR_LOG_S(INFO, WORLD) << "Volatile DB: initial cache rate = " << volatile_db_cache_rate_
                            << std::endl;
    HCTR_LOG_S(INFO, WORLD) << "Volatile DB: cache missed embeddings = "
                            << volatile_db_cache_missed_embeddings_ << std::endl;
  }

  // Connect to persistent database.
  {
    const auto& conf = inference_params_array[0].persistent_db;
    switch (conf.type) {
      case DatabaseType_t::Disabled:
        break;  // No persistent database.
      case DatabaseType_t::RocksDB:
        HCTR_LOG(INFO, WORLD, "Creating RocksDB backend...\n");
        persistent_db_ = std::make_unique<RocksDBBackend<TypeHashKey>>(
            conf.path, conf.num_threads, conf.read_only, conf.max_get_batch_size,
            conf.max_set_batch_size);
        break;
      default:
        HCTR_DIE("Selected backend (persistent_db.type = %d) is not supported!", conf.type);
        break;
    }
  }

  // Load embeddings for each embedding table from each model
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    update_database_per_model(inference_params_array[i]);
  }

  std::vector<std::string> model_config_path(inference_params_array.size());
  // Initilize embedding cache for each embedding table of each model
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    create_embedding_cache_per_model(inference_params_array[i]);
    inference_params_map_.emplace(inference_params_array[i].model_name, inference_params_array[i]);
  }
  buffer_pool_.reset(new ManagerPool(model_cache_map_, memory_pool_config_));

  // Insert emeddings to embedding cache for each embedding table of each mode
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if (inference_params_array[i].use_gpu_embedding_cache &&
        inference_params_array[i].cache_refresh_percentage_per_iteration > 0) {
      init_ec(inference_params_array[i], model_cache_map_[inference_params_array[i].model_name]);
    }
  }
}

template <typename TypeHashKey>
HierParameterServer<TypeHashKey>::~HierParameterServer() {
  for (auto it = model_cache_map_.begin(); it != model_cache_map_.end(); it++) {
    for (auto& v : it->second) {
      v.second->finalize();
    }
  }
  buffer_pool_->DestoryManagerPool();
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::update_database_per_model(
    const InferenceParams& inference_params) {
  if (ps_config_.use_coll_cache) {
    HCTR_CHECK_HINT(IModelLoader::preserved_model_loader == nullptr, "Only single model supported");
    HCTR_CHECK_HINT(inference_params.sparse_model_files.size() == 1,
                    "Only single sparse file in model is supported");
  }
  IModelLoader* rawreader = ModelLoader<TypeHashKey, float>::CreateLoader(DBTableDumpFormat_t::Raw);
  IModelLoader::preserved_model_loader = std::shared_ptr<IModelLoader>(rawreader);
  // Create input file stream to read the embedding file
  for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
    if (ps_config_.embedding_vec_size_[inference_params.model_name].size() !=
        inference_params.sparse_model_files.size()) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Wrong input: The number of embedding tables in network json file for model " +
                         inference_params.model_name +
                         " doesn't match the size of 'sparse_model_files' in configuration.");
    }
    // Get raw format model loader
    rawreader->load(inference_params.embedding_table_names[j],
                    inference_params.sparse_model_files[j]);
    if (inference_params.use_multi_worker) {
      CollCacheParameterServer::barrier();
    }
    const std::string tag_name = make_tag_name(
        inference_params.model_name, ps_config_.emb_table_name_[inference_params.model_name][j]);
    size_t num_key = rawreader->getkeycount();
    const size_t embedding_size = ps_config_.embedding_vec_size_[inference_params.model_name][j];
    // Populate volatile database(s).
    if (volatile_db_) {
      const size_t volatile_capacity = volatile_db_->capacity(tag_name);
      const size_t volatile_cache_amount =
          (num_key <= volatile_capacity)
              ? num_key
              : static_cast<size_t>(
                    volatile_db_cache_rate_ * static_cast<double>(volatile_capacity) + 0.5);

      if (dynamic_cast<DirectMapBackend<TypeHashKey>*>(volatile_db_.get()) &&
          rawreader->is_mock == false) {
        auto keys_ptr = (const TypeHashKey*)(rawreader->getkeys());
#pragma omp parallel for
        for (TypeHashKey i = 0; i < volatile_cache_amount; i++) {
          HCTR_CHECK_HINT(keys_ptr[i] == i, "direct backend requries continious source");
        }
      }
      HCTR_CHECK(volatile_db_->insert(tag_name, volatile_cache_amount,
                                      reinterpret_cast<const TypeHashKey*>(rawreader->getkeys()),
                                      reinterpret_cast<const char*>(rawreader->getvectors()),
                                      embedding_size * sizeof(float)));
      volatile_db_->synchronize();
      HCTR_LOG_S(INFO, WORLD) << "Table: " << tag_name << "; cached " << volatile_cache_amount
                              << " / " << num_key << " embeddings in volatile database ("
                              << volatile_db_->get_name()
                              << "); load: " << volatile_db_->size(tag_name) << " / "
                              << volatile_capacity << " (" << std::fixed << std::setprecision(2)
                              << (static_cast<double>(volatile_db_->size(tag_name)) * 100.0 /
                                  static_cast<double>(volatile_capacity))
                              << "%)." << std::endl;
    }

    // Persistent database - by definition - always gets all keys.
    if (persistent_db_) {
      HCTR_CHECK(persistent_db_->insert(
          tag_name, num_key, reinterpret_cast<const TypeHashKey*>(rawreader->getkeys()),
          reinterpret_cast<const char*>(rawreader->getvectors()), embedding_size * sizeof(float)));
      HCTR_LOG_S(INFO, WORLD) << "Table: " << tag_name << "; cached " << num_key
                              << " embeddings in persistent database ("
                              << persistent_db_->get_name() << ")." << std::endl;
    }
  }
  if (!ps_config_.use_coll_cache &&
      inference_params.volatile_db.type != DatabaseType_t::DirectMap) {
    rawreader->delete_table();
  }

  // Connect to online update service (if configured).
  // TODO: Maybe need to change the location where this is initialized.
  const char kafka_group_prefix[] = "hps.";

  auto kafka_prepare_filter = [](const std::string& s) -> std::string {
    std::ostringstream os;
    os << '^' << PS_EMBEDDING_TABLE_TAG_PREFIX << "\\." << s << "\\..+$";
    return os.str();
  };

  char host_name[HOST_NAME_MAX + 1];
  HCTR_CHECK_HINT(!gethostname(host_name, sizeof(host_name)), "Unable to determine hostname.\n");

  switch (inference_params.update_source.type) {
    case UpdateSourceType_t::Null:
      break;  // Disabled

    case UpdateSourceType_t::KafkaMessageQueue:
      // Volatile database updates.
      if (volatile_db_ && !inference_params.volatile_db.update_filters.empty()) {
        std::ostringstream consumer_group;
        consumer_group << kafka_group_prefix << "volatile";
        if (!volatile_db_->is_shared()) {
          consumer_group << '.' << host_name;
        }

        std::vector<std::string> tag_filters;
        std::transform(inference_params.volatile_db.update_filters.begin(),
                       inference_params.volatile_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);

        volatile_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, consumer_group.str(), tag_filters,
            inference_params.update_source.metadata_refresh_interval_ms,
            inference_params.update_source.receive_buffer_size,
            inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms,
            inference_params.update_source.max_commit_interval);
      }
      // Persistent database updates.
      if (persistent_db_ && !inference_params.persistent_db.update_filters.empty()) {
        std::ostringstream consumer_group;
        consumer_group << kafka_group_prefix << "persistent";
        if (!persistent_db_->is_shared()) {
          consumer_group << '.' << host_name;
        }

        std::vector<std::string> tag_filters;
        std::transform(inference_params.persistent_db.update_filters.begin(),
                       inference_params.persistent_db.update_filters.end(),
                       std::back_inserter(tag_filters), kafka_prepare_filter);

        persistent_db_source_ = std::make_unique<KafkaMessageSource<TypeHashKey>>(
            inference_params.update_source.brokers, consumer_group.str(), tag_filters,
            inference_params.update_source.metadata_refresh_interval_ms,
            inference_params.update_source.receive_buffer_size,
            inference_params.update_source.poll_timeout_ms,
            inference_params.update_source.max_batch_size,
            inference_params.update_source.failure_backoff_ms,
            inference_params.update_source.max_commit_interval);
      }
      break;

    default:
      HCTR_DIE("Unsupported update source!\n");
      break;
  }

  HCTR_LOG(DEBUG, WORLD, "Real-time subscribers created!\n");

  auto insert_fn = [&](DatabaseBackend<TypeHashKey>* const db, const std::string& tag,
                       const size_t num_pairs, const TypeHashKey* keys, const char* values,
                       const size_t value_size) -> bool {
    HCTR_LOG(DEBUG, WORLD,
             "Database \"%s\" update for tag: \"%s\", num_pairs: %d, value_size: %d bytes\n",
             db->get_name(), tag.c_str(), num_pairs, value_size);
    return db->insert(tag, num_pairs, keys, values, value_size);
  };

  // TODO: Update embedding cache!

  // Turn on background updates.
  if (volatile_db_source_) {
    volatile_db_source_->engage([&](const std::string& tag, const size_t num_pairs,
                                    const TypeHashKey* keys, const char* values,
                                    const size_t value_size) -> bool {
      // Try a search. If we can find the value, override it. If not, do nothing.
      return insert_fn(volatile_db_.get(), tag, num_pairs, keys, values, value_size);
    });
  }

  if (persistent_db_source_) {
    persistent_db_source_->engage([&](const std::string& tag, const size_t num_pairs,
                                      const TypeHashKey* keys, const char* values,
                                      const size_t value_size) -> bool {
      // For persistent, we always insert.
      return insert_fn(persistent_db_.get(), tag, num_pairs, keys, values, value_size);
    });
  }
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::init_ec(
    InferenceParams& inference_params,
    std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map) {
  IModelLoader* rawreader = ModelLoader<TypeHashKey, float>::CreateLoader(DBTableDumpFormat_t::Raw);

  for (size_t j = 0; j < inference_params.sparse_model_files.size(); j++) {
    rawreader->load(inference_params.embedding_table_names[j],
                    inference_params.sparse_model_files[j]);
    HCTR_LOG(INFO, ROOT, "EC initialization for model: \"%s\", num_tables: %d\n",
             inference_params.model_name.c_str(), inference_params.sparse_model_files.size());
    for (auto device_id : inference_params.deployed_devices) {
      HCTR_LOG(INFO, ROOT, "EC initialization on device: %d\n", device_id);
      cudaStream_t stream = embedding_cache_map[device_id]->get_refresh_streams()[j];
      embedding_cache_config cache_config = embedding_cache_map[device_id]->get_cache_config();
      // apply the memory block for embedding cache refresh workspace
      MemoryBlock* memory_block = nullptr;
      while (memory_block == nullptr) {
        memory_block = reinterpret_cast<MemoryBlock*>(this->apply_buffer(
            inference_params.model_name, device_id, CACHE_SPACE_TYPE::REFRESHER));
      }
      EmbeddingCacheRefreshspace refreshspace_handler = memory_block->refresh_buffer;
      // initilize the embedding cache for each table
      const size_t stride_set = floor(cache_config.num_set_in_cache_[j] *
                                      cache_config.cache_refresh_percentage_per_iteration);
      size_t length = SLAB_SIZE * SET_ASSOCIATIVITY * stride_set;
      size_t num_iteration = 0;
      for (size_t idx_set = 0; idx_set + stride_set < cache_config.num_set_in_cache_[j];
           idx_set += stride_set) {
        refreshspace_handler.h_length_ = &length;
        // copy the embedding keys from reader to refresh space
        HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.d_refresh_embeddingcolumns_,
                                       reinterpret_cast<const TypeHashKey*>(rawreader->getkeys()) +
                                           (*refreshspace_handler.h_length_ * num_iteration),
                                       *refreshspace_handler.h_length_ * sizeof(TypeHashKey),
                                       cudaMemcpyHostToDevice, stream));
        // copy the embedding vectors from reader to refresh space
        HCTR_LIB_THROW(cudaMemcpyAsync(
            refreshspace_handler.d_refresh_emb_vec_,
            reinterpret_cast<const float*>(rawreader->getvectors()) +
                (*refreshspace_handler.h_length_ * num_iteration *
                 cache_config.embedding_vec_size_[j]),
            *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[j] * sizeof(float),
            cudaMemcpyHostToDevice, stream));

        embedding_cache_map[device_id]->init(j, refreshspace_handler, stream);
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));
        num_iteration++;
      }
      this->free_buffer(memory_block);
    }
  }
  rawreader->delete_table();
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::create_embedding_cache_per_model(
    InferenceParams& inference_params) {
  if (inference_params.deployed_devices.empty()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The list of deployed devices is empty.");
  }
  if (std::find(inference_params.deployed_devices.begin(), inference_params.deployed_devices.end(),
                inference_params.device_id) == inference_params.deployed_devices.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The device id is not in the list of deployed devices.");
  }
  std::map<int64_t, std::shared_ptr<EmbeddingCacheBase>> embedding_cache_map;
  for (auto device_id : inference_params.deployed_devices) {
    HCTR_LOG(INFO, WORLD, "Creating embedding cache in device %d.\n", device_id);
    inference_params.device_id = device_id;
    embedding_cache_map[device_id] = EmbeddingCacheBase::create(inference_params, ps_config_, this);
  }
  model_cache_map_[inference_params.model_name] = embedding_cache_map;
  memory_pool_config_.num_woker_buffer_size_per_model[inference_params.model_name] =
      inference_params.number_of_worker_buffers_in_pool;
  memory_pool_config_.num_refresh_buffer_size_per_model[inference_params.model_name] =
      inference_params.number_of_refresh_buffers_in_pool;
  if (buffer_pool_ != nullptr) {
    buffer_pool_->_create_memory_pool_per_model(inference_params.model_name,
                                                inference_params.number_of_worker_buffers_in_pool,
                                                embedding_cache_map, CACHE_SPACE_TYPE::WORKER);
    buffer_pool_->_create_memory_pool_per_model(inference_params.model_name,
                                                inference_params.number_of_refresh_buffers_in_pool,
                                                embedding_cache_map, CACHE_SPACE_TYPE::REFRESHER);
  }
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::destory_embedding_cache_per_model(
    const std::string& model_name) {
  if (model_cache_map_.find(model_name) != model_cache_map_.end()) {
    for (auto& f : model_cache_map_[model_name]) {
      f.second->finalize();
    }
    model_cache_map_.erase(model_name);
  }
  buffer_pool_->DestoryManagerPool(model_name);
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::erase_model_from_hps(const std::string& model_name) {
  if (volatile_db_) {
    const std::vector<std::string>& table_names = volatile_db_->find_tables(model_name);
    volatile_db_->evict(table_names);
  }
  if (persistent_db_) {
    const std::vector<std::string>& table_names = persistent_db_->find_tables(model_name);
    persistent_db_->evict(table_names);
  }
}

template <typename TypeHashKey>
std::shared_ptr<EmbeddingCacheBase> HierParameterServer<TypeHashKey>::get_embedding_cache(
    const std::string& model_name, const int device_id) {
  const auto it = model_cache_map_.find(model_name);
  if (it == model_cache_map_.end()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "No embedding cache for model " + model_name);
  }
  if (it->second.find(device_id) == it->second.end()) {
    std::ostringstream os;
    os << "No embedding cache on device " << device_id << " for model " << model_name;
    HCTR_OWN_THROW(Error_t::WrongInput, os.str());
  }
  return model_cache_map_[model_name][device_id];
}

template <typename TypeHashKey>
std::map<std::string, InferenceParams>
HierParameterServer<TypeHashKey>::get_hps_model_configuration_map() {
  return inference_params_map_;
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::parse_hps_configuraion(
    const std::string& hps_json_config_file) {
  parameter_server_config ps_config{hps_json_config_file};

  for (auto infer_param : ps_config.inference_params_array) {
    inference_params_map_.emplace(infer_param.model_name, infer_param);
  }
}

template <typename TypeHashKey>
void* HierParameterServer<TypeHashKey>::apply_buffer(const std::string& model_name, int device_id,
                                                     CACHE_SPACE_TYPE cache_type) {
  return buffer_pool_->AllocBuffer(model_name, device_id, cache_type);
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::free_buffer(void* p) {
  buffer_pool_->FreeBuffer(p);
  return;
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::lookup(const void* const h_keys, const size_t length,
                                              float* const h_vectors, const std::string& model_name,
                                              const size_t table_id) {
  if (!length) {
    return;
  }
  const auto start_time = std::chrono::high_resolution_clock::now();
  const auto time_budget = std::chrono::nanoseconds::max();

  const auto& model_id = ps_config_.find_model_id(model_name);
  HCTR_CHECK_HINT(
      static_cast<bool>(model_id),
      "Error: parameter server unknown model name. Note that this error will also come out with "
      "using Triton LOAD/UNLOAD APIs which haven't been supported in HPS backend.\n");

  const size_t embedding_size = ps_config_.embedding_vec_size_[model_name][table_id];
  const size_t expected_value_size = embedding_size * sizeof(float);
  const std::string& embedding_table_name = ps_config_.emb_table_name_[model_name][table_id];
  const std::string& tag_name = make_tag_name(model_name, embedding_table_name);
  const float default_vec_value = ps_config_.default_emb_vec_value_[*model_id][table_id];
#ifdef ENABLE_INFERENCE
  HCTR_LOG_S(TRACE, WORLD) << "Looking up " << length << " embeddings (each with " << embedding_size
                           << " values)..." << std::endl;
#endif
  size_t hit_count = 0;

  DatabaseHitCallback check_and_copy = [&](const size_t index, const char* const value,
                                           const size_t value_size) {
    HCTR_CHECK_HINT(value_size == expected_value_size,
                    "Table: %s; Batch[%d]: Value size mismatch! (%d <> %d)!", tag_name.c_str(),
                    index, value_size, expected_value_size);
    memcpy(&h_vectors[index * embedding_size], value, value_size);
  };

  DatabaseMissCallback fill_default = [&](const size_t index) {
    std::fill_n(&h_vectors[index * embedding_size], embedding_size, default_vec_value);
  };

  // If have volatile and persistant database.
  if (volatile_db_ && persistent_db_) {
    std::mutex resource_guard;

    // Do a sequential lookup in the volatile DB, and remember the missing keys.
    std::vector<size_t> missing;
    DatabaseMissCallback record_missing = [&](const size_t index) {
      std::lock_guard<std::mutex> lock(resource_guard);
      missing.push_back(index);
    };

    hit_count += volatile_db_->fetch(tag_name, length, reinterpret_cast<const TypeHashKey*>(h_keys),
                                     check_and_copy, record_missing, time_budget);

    HCTR_LOG_S(TRACE, WORLD) << volatile_db_->get_name() << ": " << hit_count << " hits, "
                             << missing.size() << " missing!" << std::endl;

    // If the layer 0 cache should be optimized as we go, elevate missed keys.
    std::shared_ptr<std::vector<TypeHashKey>> keys_to_elevate;
    std::shared_ptr<std::vector<char>> values_to_elevate;

    if (volatile_db_cache_missed_embeddings_) {
      keys_to_elevate = std::make_shared<std::vector<TypeHashKey>>();
      values_to_elevate = std::make_shared<std::vector<char>>();

      check_and_copy = [&](const size_t index, const char* const value, const uint32_t value_size) {
        HCTR_CHECK_HINT(value_size == expected_value_size,
                        "Table: %s; Batch[%d]: Value size mismatch! (%d <> %d)!", tag_name.c_str(),
                        index, value_size, expected_value_size);
        memcpy(&h_vectors[index * embedding_size], value, value_size);

        std::lock_guard<std::mutex> lock(resource_guard);
        keys_to_elevate->emplace_back(reinterpret_cast<const TypeHashKey*>(h_keys)[index]);
        values_to_elevate->insert(values_to_elevate->end(), value, &value[value_size]);
      };
    }

    // Do a sparse lookup in the persisent DB, to fill gaps and set others to default.
    hit_count += persistent_db_->fetch(tag_name, missing.size(), missing.data(),
                                       reinterpret_cast<const TypeHashKey*>(h_keys), check_and_copy,
                                       fill_default, time_budget);

    HCTR_LOG_S(TRACE, WORLD) << persistent_db_->get_name() << ": " << hit_count << " hits, "
                             << (length - hit_count) << " missing!" << std::endl;

    // Elevate keys if desired and possible.
    if (keys_to_elevate && !keys_to_elevate->empty()) {
      HCTR_LOG_S(DEBUG, WORLD) << "Attempting to migrate " << keys_to_elevate->size()
                               << " embeddings from " << persistent_db_->get_name() << " to "
                               << volatile_db_->get_name() << '.' << std::endl;
      volatile_db_->insert_async(tag_name, keys_to_elevate, values_to_elevate, expected_value_size);
    }
  } else {
    // If any database.
    DatabaseBackend<TypeHashKey>* const db =
        volatile_db_ ? static_cast<DatabaseBackend<TypeHashKey>*>(volatile_db_.get())
                     : static_cast<DatabaseBackend<TypeHashKey>*>(persistent_db_.get());
    if (db) {
      // Do a sequential lookup in the volatile DB, but fill gaps with a default value.
      hit_count += db->fetch(tag_name, length, reinterpret_cast<const TypeHashKey*>(h_keys),
                             check_and_copy, fill_default, time_budget);

      HCTR_LOG_S(TRACE, WORLD) << db->get_name() << ": " << hit_count << " hits, "
                               << (length - hit_count) << " missing!" << std::endl;
    } else {
      // Without a database, set everything to default.
      std::fill_n(h_vectors, length * embedding_size, default_vec_value);
      HCTR_LOG_S(WARNING, WORLD) << "No database. All embeddings set to default." << std::endl;
    }
  }

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
#ifdef ENABLE_INFERENCE
  HCTR_LOG_S(TRACE, WORLD) << "Parameter server lookup of " << hit_count << " / " << length
                           << " embeddings took " << duration.count() << " us." << std::endl;
#endif
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::refresh_embedding_cache(const std::string& model_name,
                                                               const int device_id) {
  HCTR_LOG(INFO, WORLD, "*****Refresh embedding cache of model %s on device %d*****\n",
           model_name.c_str(), device_id);
  HugeCTR::Timer timer_refresh;

  std::shared_ptr<EmbeddingCacheBase> embedding_cache = get_embedding_cache(model_name, device_id);
  if (!embedding_cache->use_gpu_embedding_cache()) {
    HCTR_LOG(WARNING, WORLD, "GPU embedding cache is not enabled and cannot be refreshed!\n");
    return;
  }

  embedding_cache_config cache_config = embedding_cache->get_cache_config();
  if (cache_config.cache_refresh_percentage_per_iteration <= 0) {
    HCTR_LOG(WARNING, WORLD,
             "The configuration of cache refresh percentage per iteration must be greater than 0 "
             "to refresh the GPU embedding cache!\n");
    return;
  }
  timer_refresh.start();
  std::vector<cudaStream_t> streams = embedding_cache->get_refresh_streams();
  // apply the memory block for embedding cache refresh workspace
  MemoryBlock* memory_block = nullptr;
  while (memory_block == nullptr) {
    memory_block = reinterpret_cast<MemoryBlock*>(
        this->apply_buffer(model_name, device_id, CACHE_SPACE_TYPE::REFRESHER));
  }
  EmbeddingCacheRefreshspace refreshspace_handler = memory_block->refresh_buffer;
  // Refresh the embedding cache for each table
  const size_t stride_set = cache_config.num_set_in_refresh_workspace_;
  HugeCTR::Timer timer;
  for (size_t i = 0; i < cache_config.num_emb_table_; i++) {
    for (size_t idx_set = 0; idx_set < cache_config.num_set_in_cache_[i]; idx_set += stride_set) {
      const size_t end_idx = (idx_set + stride_set > cache_config.num_set_in_cache_[i])
                                 ? cache_config.num_set_in_cache_[i]
                                 : idx_set + stride_set;
      timer.start();
      embedding_cache->dump(i, refreshspace_handler.d_refresh_embeddingcolumns_,
                            refreshspace_handler.d_length_, idx_set, end_idx, streams[i]);

      HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.h_length_, refreshspace_handler.d_length_,
                                     sizeof(size_t), cudaMemcpyDeviceToHost, streams[i]));
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
      HCTR_LIB_THROW(cudaMemcpyAsync(refreshspace_handler.h_refresh_embeddingcolumns_,
                                     refreshspace_handler.d_refresh_embeddingcolumns_,
                                     *refreshspace_handler.h_length_ * sizeof(TypeHashKey),
                                     cudaMemcpyDeviceToHost, streams[i]));
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
      timer.stop();
      HCTR_LOG_S(INFO, ROOT) << "Embedding Cache dumping the number of " << stride_set
                             << " sets takes: " << timer.elapsedSeconds() << "s" << std::endl;
      timer.start();
      this->lookup(
          reinterpret_cast<const TypeHashKey*>(refreshspace_handler.h_refresh_embeddingcolumns_),
          *refreshspace_handler.h_length_, refreshspace_handler.h_refresh_emb_vec_, model_name, i);
      HCTR_LIB_THROW(cudaMemcpyAsync(
          refreshspace_handler.d_refresh_emb_vec_, refreshspace_handler.h_refresh_emb_vec_,
          *refreshspace_handler.h_length_ * cache_config.embedding_vec_size_[i] * sizeof(float),
          cudaMemcpyHostToDevice, streams[i]));
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
      timer.stop();
      HCTR_LOG_S(INFO, ROOT) << "Parameter Server looking up the number of "
                             << *refreshspace_handler.h_length_
                             << " keys takes: " << timer.elapsedSeconds() << "s" << std::endl;
      timer.start();
      embedding_cache->refresh(
          static_cast<int>(i), refreshspace_handler.d_refresh_embeddingcolumns_,
          refreshspace_handler.d_refresh_emb_vec_, *refreshspace_handler.h_length_, streams[i]);
      timer.stop();
      HCTR_LOG_S(INFO, ROOT) << "Embedding Cache refreshing the number of "
                             << *refreshspace_handler.h_length_
                             << " keys takes: " << timer.elapsedSeconds() << "s" << std::endl;
      HCTR_LIB_THROW(cudaStreamSynchronize(streams[i]));
    }
  }
  // apply the memory block for embedding cache refresh workspace
  this->free_buffer(memory_block);
  timer_refresh.stop();
  HCTR_LOG_S(INFO, ROOT) << "The total Time of embedding cache refresh is : "
                         << timer_refresh.elapsedSeconds() << "s" << std::endl;
}

template <typename TypeHashKey>
void HierParameterServer<TypeHashKey>::insert_embedding_cache(
    const size_t table_id, std::shared_ptr<EmbeddingCacheBase> embedding_cache,
    EmbeddingCacheWorkspace& workspace_handler, cudaStream_t stream) {
  auto cache_config = embedding_cache->get_cache_config();
#ifdef ENABLE_INFERENCE
  HCTR_LOG(TRACE, WORLD, "*****Insert embedding cache of model %s on device %d*****\n",
           cache_config.model_name_.c_str(), cache_config.cuda_dev_id_);
#endif
  // Copy the missing embeddingcolumns to host
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  HCTR_LIB_THROW(
      cudaMemcpyAsync(workspace_handler.h_missing_embeddingcolumns_[table_id],
                      workspace_handler.d_missing_embeddingcolumns_[table_id],
                      workspace_handler.h_missing_length_[table_id] * sizeof(TypeHashKey),
                      cudaMemcpyDeviceToHost, stream));

  // Query the missing embeddingcolumns from Parameter Server
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  this->lookup(workspace_handler.h_missing_embeddingcolumns_[table_id],
               workspace_handler.h_missing_length_[table_id],
               workspace_handler.h_missing_emb_vec_[table_id], cache_config.model_name_, table_id);

  // Copy missing emb_vec to device

  const size_t missing_len_in_byte = workspace_handler.h_missing_length_[table_id] *
                                     cache_config.embedding_vec_size_[table_id] * sizeof(float);
  HCTR_LIB_THROW(cudaMemcpyAsync(workspace_handler.d_missing_emb_vec_[table_id],
                                 workspace_handler.h_missing_emb_vec_[table_id],
                                 missing_len_in_byte, cudaMemcpyHostToDevice, stream));
  // Insert the vectors for missing keys into embedding cache
  embedding_cache->insert(table_id, workspace_handler, stream);
}

template <typename TypeHashKey>
double HierParameterServer<TypeHashKey>::report_cache_intersect() {
  HCTR_LOG_S(DEBUG, WORLD) << "HierParameterServer<TypeHashKey>::report_cache_intersect from "
                           << "Device " << RunConfig::worker_id << ".\n";
  HCTR_CHECK_HINT(model_cache_map_.size() == 1,
                  "There should be only one model while reporting cache intersect.");
  auto& cache_map = model_cache_map_.begin()->second;
  HCTR_CHECK_HINT(cache_map.size() == 1 || cache_map.size() == RunConfig::num_device,
                  "There should be only one device in one process, \
                  or all device in single process.");
  bool single_process = (cache_map.size() != 1);
  auto& embed_cache = cache_map[RunConfig::worker_id];
  size_t slot_num = embed_cache->get_slot_num();
  std::vector<size_t> shape = {RunConfig::num_device, slot_num};
  HCTR_CHECK_HINT((sizeof(TypeHashKey) == 4) || (sizeof(TypeHashKey) == 8),
                  "Key should be either 4 bytes or 8 bytes.");
  DataType dtype = (sizeof(TypeHashKey) == 4) ? DataType::kI32 : DataType::kI64;
  TensorPtr keys_shm_base_ptr;
  double final_ratio = 0;
  int cnt = 0;

  // main process: alloc new shared memory
  if (!single_process) {
    if (RunConfig::worker_id == 0) {
      keys_shm_base_ptr =
          Tensor::CreateShm(HPSCacheKeyShmName.c_str(), dtype, shape, HPSCacheKeyShmName.c_str());
      HCTR_LOG_S(INFO, WORLD) << "Device " << RunConfig::worker_id << " create shared memory \""
                              << HPSCacheKeyShmName.c_str() << "\" with nbytes "
                              << keys_shm_base_ptr->NumBytes() << ".\n";
    }
    CollCacheParameterServer::barrier();
    if (RunConfig::worker_id != 0) {
      keys_shm_base_ptr =
          Tensor::OpenShm(HPSCacheKeyShmName.c_str(), dtype, shape, HPSCacheKeyShmName.c_str());
      HCTR_LOG_S(DEBUG, WORLD) << "Device " << RunConfig::worker_id << " open shared memory \""
                               << HPSCacheKeyShmName.c_str() << "\" with nbytes "
                               << keys_shm_base_ptr->NumBytes() << ".\n";
    }
  } else {
    keys_shm_base_ptr =
        Tensor::CreateShm(HPSCacheKeyShmName.c_str(), dtype, shape, HPSCacheKeyShmName.c_str());
    HCTR_LOG_S(INFO, WORLD) << "Single process create shared memory \"" << keys_shm_base_ptr->Name()
                            << "\" with nbytes " << keys_shm_base_ptr->NumBytes() << ".\n";
  }

  // copy embedding keys from each GPU to CPU memory
  TypeHashKey* keys_shm_global_base = (TypeHashKey*)keys_shm_base_ptr->MutableData();
  if (!single_process) {
    TypeHashKey* keys_shm_local_base = keys_shm_global_base + slot_num * RunConfig::worker_id;
    void* keys_ptr_local = (void*)keys_shm_local_base;
    embed_cache->get_keys(keys_ptr_local, slot_num);
    CollCacheParameterServer::barrier();
  } else {
    for (size_t i = 0; i < RunConfig::num_device; i++) {
      int device_id = cache_map[i]->get_device_id();
      cudaSetDevice(device_id);
      embed_cache->get_keys(keys_shm_global_base + slot_num * i, slot_num);
    }
  }

  // util func to calculate cache intersection of two sorted arrays
  auto get_intersect_num = [](TypeHashKey* a, TypeHashKey* b, size_t total_cnt) -> size_t {
    size_t intersect_cnt = 0;
    for (uint64_t a_ptr = 0, b_ptr = 0; a_ptr < total_cnt && b_ptr < total_cnt;) {
      if (a[a_ptr] == std::numeric_limits<TypeHashKey>::max() ||
          b[b_ptr] == std::numeric_limits<TypeHashKey>::max())
        break;
      if (a[a_ptr] == b[b_ptr]) {
        a_ptr++;
        b_ptr++;
        intersect_cnt++;
      } else if (a[a_ptr] > b[b_ptr]) {
        b_ptr++;
      } else {
        a_ptr++;
      }
    }
    return intersect_cnt;
  };

  // calculate the intersect ratial of their keys one by one
  if (RunConfig::worker_id == 0) {
    HCTR_LOG_S(DEBUG, WORLD)
        << "[HierParameterServer::report_cache_intersect] gpu_key_nums on each device: \n";
    for (uint64_t i = 0; i < RunConfig::num_device; i++) {
      TypeHashKey* cur_keys_ptr = keys_shm_global_base + slot_num * i;

      // print debug info
      uint64_t non_empty_key_cnt = 0;
      for (uint64_t j = 0; j < slot_num; j++)
        if (cur_keys_ptr[j] != std::numeric_limits<TypeHashKey>::max()) non_empty_key_cnt++;
      // HCTR_LOG_S(INFO, WORLD) << "device " << i <<  " (" << non_empty_key_cnt << "|" << slot_num
      // << "): "; for (uint64_t j = 0; j < 5; j++) std::cout << cur_keys_ptr[j] << " "; std::cout
      // << "\n";

      // calculate the intersect ratial of their keys
      for (uint64_t j = i + 1; j < RunConfig::num_device; j++) {
        TypeHashKey* cur_keys_ptr_j = keys_shm_global_base + slot_num * j;
        size_t intersect_num = get_intersect_num(cur_keys_ptr, cur_keys_ptr_j, slot_num);
        double intersect_ratio = (float)intersect_num / non_empty_key_cnt;
        HCTR_LOG_S(DEBUG, WORLD) << "intersect ratio of device [" << i << ", " << j
                                 << "]: " << intersect_ratio << "%\n";
        final_ratio += intersect_ratio;
        cnt++;
      }
    }

    return (final_ratio / cnt);
  }

  return 0;
}

template <typename TypeHashKey>
std::vector<double> HierParameterServer<TypeHashKey>::report_access_overlap() {
  HCTR_LOG_S(INFO, WORLD) << "HierParameterServer<TypeHashKey>::report_access_overlap from "
                          << "Device " << RunConfig::worker_id << ".\n";
  HCTR_CHECK_HINT(model_cache_map_.size() == 1,
                  "There should be only one model while reporting access overlap.");
  auto& cache_map = model_cache_map_.begin()->second;
  HCTR_CHECK_HINT(cache_map.size() == 1 || cache_map.size() == RunConfig::num_device,
                  "There should be only one device in one process, \
                  or all device in single process.");
  bool single_process = (cache_map.size() != 1);
  auto& embed_cache = cache_map[RunConfig::worker_id];
  size_t total_key_num = embed_cache->emb_key_num;
  size_t total_lookups = embed_cache->total_lookups;
  cudaStream_t stream = embed_cache->get_refresh_streams()[0];
  std::vector<size_t> shape = {RunConfig::num_device, total_key_num * 2};
  DataType dtype = DataType::kI32;
  TensorPtr access_shm_base_ptr;
  std::vector<double> final_ratios(3, 0);
  int cnt = 0;
  Context ctx = GPU();

  // main process: alloc new shared memory
  if (!single_process) {
    if (RunConfig::worker_id == 0) {
      access_shm_base_ptr = Tensor::CreateShm(HPSCacheAccessCountShmName.c_str(), dtype, shape,
                                              HPSCacheAccessCountShmName.c_str());
      HCTR_LOG_S(INFO, WORLD) << "Device " << RunConfig::worker_id << " create shared memory \""
                              << HPSCacheAccessCountShmName.c_str() << "\" with nbytes "
                              << access_shm_base_ptr->NumBytes() << ".\n";
    }
    CollCacheParameterServer::barrier();
    if (RunConfig::worker_id != 0) {
      access_shm_base_ptr = Tensor::OpenShm(HPSCacheAccessCountShmName.c_str(), dtype, shape,
                                            HPSCacheAccessCountShmName.c_str());
      HCTR_LOG_S(INFO, WORLD) << "Device " << RunConfig::worker_id << " open shared memory \""
                              << HPSCacheAccessCountShmName.c_str() << "\" with nbytes "
                              << access_shm_base_ptr->NumBytes() << ".\n";
    }
  } else {
    access_shm_base_ptr = Tensor::CreateShm(HPSCacheAccessCountShmName.c_str(), dtype, shape,
                                            HPSCacheAccessCountShmName.c_str());
    HCTR_LOG_S(INFO, WORLD) << "Single process create shared memory \""
                            << HPSCacheAccessCountShmName.c_str() << "\" with nbytes "
                            << access_shm_base_ptr->NumBytes() << ".\n";
  }

  // copy embedding keys from each GPU to CPU memory
  uint32_t* access_shm_global_base = access_shm_base_ptr->Ptr<uint32_t>();
  auto get_shm_ptr = [&](size_t device_id, bool hit) -> void* {
    return reinterpret_cast<void*>(access_shm_global_base +
                                   total_key_num * (2 * device_id + (!hit)));
  };
  if (!single_process) {
    HCTR_LIB_THROW(cudaMemcpy(get_shm_ptr(RunConfig::worker_id, true),
                              reinterpret_cast<void*>(embed_cache->local_hit_key_counters),
                              sizeof(uint32_t) * total_key_num, cudaMemcpyDeviceToHost));
    HCTR_LIB_THROW(cudaMemcpy(get_shm_ptr(RunConfig::worker_id, false),
                              reinterpret_cast<void*>(embed_cache->local_miss_key_counters),
                              sizeof(uint32_t) * total_key_num, cudaMemcpyDeviceToHost));
    CollCacheParameterServer::barrier();
  } else {
    for (size_t i = 0; i < RunConfig::num_device; i++) {
      int device_id = cache_map[i]->get_device_id();
      cudaSetDevice(device_id);
      HCTR_LIB_THROW(
          cudaMemcpy(get_shm_ptr(device_id, true),
                     reinterpret_cast<void*>(cache_map[device_id]->local_hit_key_counters),
                     sizeof(uint32_t) * total_key_num, cudaMemcpyDeviceToHost));
      HCTR_LIB_THROW(
          cudaMemcpy(get_shm_ptr(device_id, false),
                     reinterpret_cast<void*>(cache_map[device_id]->local_miss_key_counters),
                     sizeof(uint32_t) * total_key_num, cudaMemcpyDeviceToHost));
    }
  }

  // calculate the intersect ratial of their keys one by one
  if (RunConfig::worker_id == 0) {
    uint64_t hit_cnt, miss_cnt, hit_overlap_cnt, miss_overlap_cnt;
    TensorPtr d_access_hit_local_ptr = Tensor::Empty(DataType::kI32, {total_key_num}, ctx, "");
    TensorPtr d_access_miss_local_ptr = Tensor::Empty(DataType::kI32, {total_key_num}, ctx, "");
    TensorPtr d_cur_access_hit_local_ptr = Tensor::Empty(DataType::kI32, {total_key_num}, ctx, "");
    TensorPtr d_cur_access_miss_local_ptr = Tensor::Empty(DataType::kI32, {total_key_num}, ctx, "");
    TensorPtr d_middle_result = Tensor::Empty(DataType::kI32, {total_key_num}, ctx, "");
    TensorPtr d_result = Tensor::Empty(DataType::kI64, {1}, ctx, "");
    auto get_result = [&]() -> uint64_t {
      uint64_t h_result;
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
      Device::Get(ctx)->CopyDataFromTo(d_result->Ptr<uint64_t>(), 0, &h_result, 0, sizeof(uint64_t),
                                       ctx, CPU());
      return h_result;
    };

    HCTR_LOG_S(INFO, WORLD) << "[HierParameterServer::report_cache_intersect] Access overlap: \n";
    for (uint64_t i = 0; i < RunConfig::num_device; i++) {
      Device::Get(ctx)->CopyDataFromTo(get_shm_ptr(i, true), 0,
                                       d_access_hit_local_ptr->MutableData(), 0,
                                       sizeof(uint32_t) * total_key_num, CPU(), ctx, stream);
      Device::Get(ctx)->CopyDataFromTo(get_shm_ptr(i, false), 0,
                                       d_access_miss_local_ptr->MutableData(), 0,
                                       sizeof(uint32_t) * total_key_num, CPU(), ctx, stream);
      MathUtil<uint32_t>::CubReduceSum(d_access_hit_local_ptr->Ptr<uint32_t>(),
                                       d_result->Ptr<uint64_t>(), total_key_num, stream);
      hit_cnt = get_result();
      MathUtil<uint32_t>::CubReduceSum(d_access_miss_local_ptr->Ptr<uint32_t>(),
                                       d_result->Ptr<uint64_t>(), total_key_num, stream);
      miss_cnt = get_result();

      // check miss count and hit count
      HCTR_CHECK_HINT(
          hit_cnt + miss_cnt == total_lookups,
          "cache hit cnt(%lu) and miss cnt(%lu) should add up to exactly total lookup cnts(%lu)",
          hit_cnt, miss_cnt, total_lookups);
      final_ratios[0] += ((float)hit_cnt / total_lookups);  // hit ratio

      // calculate the intersect ratial of their keys
      for (uint64_t j = i + 1; j < RunConfig::num_device; j++) {
        Device::Get(ctx)->CopyDataFromTo(get_shm_ptr(j, true), 0,
                                         d_cur_access_hit_local_ptr->MutableData(), 0,
                                         sizeof(uint32_t) * total_key_num, CPU(), ctx, stream);
        Device::Get(ctx)->CopyDataFromTo(get_shm_ptr(j, false), 0,
                                         d_cur_access_miss_local_ptr->MutableData(), 0,
                                         sizeof(uint32_t) * total_key_num, CPU(), ctx, stream);

        MathUtil<TypeHashKey>::Min(d_access_hit_local_ptr->Ptr<uint32_t>(),
                                   d_cur_access_hit_local_ptr->Ptr<uint32_t>(),
                                   d_middle_result->Ptr<uint32_t>(), total_key_num, stream);
        MathUtil<uint32_t>::CubReduceSum(d_middle_result->Ptr<uint32_t>(),
                                         d_result->Ptr<uint64_t>(), total_key_num, stream);
        hit_overlap_cnt = get_result();

        MathUtil<TypeHashKey>::Min(d_access_miss_local_ptr->Ptr<uint32_t>(),
                                   d_cur_access_miss_local_ptr->Ptr<uint32_t>(),
                                   d_middle_result->Ptr<uint32_t>(), total_key_num, stream);
        MathUtil<uint32_t>::CubReduceSum(d_middle_result->Ptr<uint32_t>(),
                                         d_result->Ptr<uint64_t>(), total_key_num, stream);
        miss_overlap_cnt = get_result();

        double hit_overlap_ratio = (float)hit_overlap_cnt / hit_cnt;
        double miss_overlap_ratio = (float)miss_overlap_cnt / miss_cnt;
        HCTR_LOG_S(INFO, WORLD) << "Device [" << i << ", " << j << "]:"
                                << " hit_cnt " << hit_cnt << " miss_cnt " << miss_cnt
                                << " hit_overlap_cnt " << hit_overlap_cnt << " miss_overlap_cnt "
                                << miss_overlap_cnt << " ,Overlap(hit|miss) ratio "
                                << hit_overlap_ratio << "|" << miss_overlap_ratio << "%\n";
        final_ratios[1] += hit_overlap_ratio;
        final_ratios[2] += miss_overlap_ratio;
        cnt++;
      }
    }

    final_ratios[0] /= RunConfig::num_device;
    final_ratios[1] /= cnt;
    final_ratios[2] /= cnt;
  }

  return final_ratios;
}

template class HierParameterServer<long long>;
template class HierParameterServer<unsigned int>;

CollCacheParameterServer::CollCacheParameterServer(const parameter_server_config& ps_config)
    : ps_config_(ps_config) {
  HCTR_PRINT(INFO,
             "====================================================HPS Coll "
             "Create====================================================\n");
  const std::vector<InferenceParams>& inference_params_array = ps_config_.inference_params_array;
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    if (inference_params_array[i].volatile_db != inference_params_array[0].volatile_db ||
        inference_params_array[i].persistent_db != inference_params_array[0].persistent_db) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Inconsistent database setup. HugeCTR paramter server does currently not support hybrid "
          "database deployment.");
    }
  }
  if (ps_config_.embedding_vec_size_.size() != inference_params_array.size() ||
      ps_config_.default_emb_vec_value_.size() != inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The size of parameter server parameters are not correct.");
  }

  if (inference_params_array.size() != 1) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Coll cache only support single model for now");
  }
  auto& inference_params = inference_params_array[0];
  if (inference_params_array[0].sparse_model_files.size() != 1) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Coll cache only support single sparse file for now");
  }

  // Connect to volatile database.
  // Create input file stream to read the embedding file
  if (ps_config_.embedding_vec_size_[inference_params.model_name].size() !=
      inference_params.sparse_model_files.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: The number of embedding tables in network json file for model " +
                       inference_params.model_name +
                       " doesn't match the size of 'sparse_model_files' in configuration.");
  }
  // hps assumes disk key file is long-long
  raw_data_holder = IModelLoader::preserved_model_loader;
  // Get raw format model loader
  size_t num_key = raw_data_holder->getkeycount();
  // const size_t embedding_size = ps_config_.embedding_vec_size_[inference_params.model_name][0];
  // Populate volatile database(s).
  // auto val_ptr = reinterpret_cast<const char*>(raw_data_holder->getvectors());

  coll_cache_lib::common::RunConfig::cache_percentage = inference_params.cache_size_percentage;
  // coll_cache_lib::common::RunConfig::cache_policy = coll_cache_lib::common::kRepCache;
  // coll_cache_lib::common::RunConfig::cache_policy = coll_cache_lib::common::kCliquePart;
  coll_cache_lib::common::RunConfig::cache_policy =
      (coll_cache_lib::common::CachePolicy)ps_config_.coll_cache_policy;
  coll_cache_lib::common::RunConfig::cross_process = false;
  coll_cache_lib::common::RunConfig::device_id_list =
      inference_params.cross_worker_deployed_devices;
  coll_cache_lib::common::RunConfig::num_device =
      inference_params.cross_worker_deployed_devices.size();
  coll_cache_lib::common::RunConfig::cross_process = ps_config_.use_multi_worker;
  coll_cache_lib::common::RunConfig::num_global_step_per_epoch =
      ps_config.iteration_per_epoch * coll_cache_lib::common::RunConfig::num_device;
  coll_cache_lib::common::RunConfig::num_epoch = ps_config.epoch;
  coll_cache_lib::common::RunConfig::num_total_item = num_key;

  HCTR_LOG_S(ERROR, WORLD)
      << "coll ps creation, with "
      << ps_config.inference_params_array[0].cross_worker_deployed_devices.size()
      << " devices, using policy " << coll_cache_lib::common::RunConfig::cache_policy << "\n";
  this->coll_cache_ptr_ = std::make_shared<coll_cache_lib::CollCache>(
      nullptr, coll_cache_lib::common::AnonymousBarrier::_global_instance);
  HCTR_LOG(ERROR, WORLD, "coll ps creation done\n");
}

#ifdef DEAD_CODE
void CollCacheParameterServer::init_per_replica(int global_replica_id,
                                                IdType* ranking_nodes_list_ptr,
                                                IdType* ranking_nodes_freq_list_ptr,
                                                std::function<MemHandle(size_t)> gpu_mem_allocator,
                                                cudaStream_t cu_stream) {
  void* cpu_data = raw_data_holder->getvectors();
  double cache_percentage = ps_config_.inference_params_array[0].cache_size_percentage;
  size_t dim = ps_config_.inference_params_array[0].embedding_vecsize_per_table[0];
  // hps may be used in hugectr or tensorflow, so we don't know how to allocate memory;
  size_t num_key = raw_data_holder->getkeycount();
  HCTR_CHECK_HINT(num_key == ps_config_.inference_params_array[0].max_vocabulary_size[0],
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
  double cache_percentage = ps_config_.inference_params_array[0].cache_size_percentage;
  size_t dim = ps_config_.inference_params_array[0].embedding_vecsize_per_table[0];
  // hps may be used in hugectr or tensorflow, so we don't know how to allocate memory;
  size_t num_key = raw_data_holder->getkeycount();
  HCTR_CHECK_HINT(num_key == ps_config_.inference_params_array[0].max_vocabulary_size[0],
                  "num key from file must equal with max vocabulary: %d", num_key);
  auto stream = reinterpret_cast<coll_cache_lib::common::StreamHandle>(cu_stream);
  HCTR_LOG(ERROR, WORLD, "Calling build_v2\n");

  {
    int value;
    cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem,
                           coll_cache_lib::common::RunConfig::device_id_list[global_replica_id]);
    HCTR_LOG_S(ERROR, WORLD) << "cudaDevAttrCanUseHostPointerForRegisteredMem is " << value << "\n";
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
}  // namespace HugeCTR
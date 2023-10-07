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

#include <algorithm>
#include <embedding.hpp>
#include <iterator>

#include "HugeCTR/include/embedding_training_cache/parameter_server_manager.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"

namespace HugeCTR {

class EmbeddingTrainingCacheImplBase {
 public:
  virtual void dump() = 0;
  virtual void update(std::vector<std::string>&) = 0;
  virtual void update(std::string&) = 0;
  virtual void update_sparse_model_file() = 0;
  virtual std::vector<std::pair<std::vector<long long>, std::vector<float>>> get_incremental_model(
      const std::vector<long long>&) = 0;
  virtual ~EmbeddingTrainingCacheImplBase() = default;
};

template <typename TypeKey>
class EmbeddingTrainingCacheImpl : public EmbeddingTrainingCacheImplBase {
  std::vector<std::shared_ptr<IEmbedding>> embeddings_;
  ParameterServerManager<TypeKey> ps_manager_;

  size_t get_max_embedding_size_() {
    size_t max_embedding_size = 0;
    for (auto& one_embedding : embeddings_) {
      size_t embedding_size = one_embedding->get_max_vocabulary_size();
      max_embedding_size =
          (embedding_size > max_embedding_size) ? embedding_size : max_embedding_size;
    }
    return max_embedding_size;
  }

  /**
   * @brief Load the embedding table according to keys stored in
   *        keyset_file_list from sparse_model_entity_ to device memory.
   */
  void load_(std::vector<std::string>& keyset_file_list);

 public:
  EmbeddingTrainingCacheImpl(std::vector<TrainPSType_t>& ps_types,
                             std::vector<std::shared_ptr<IEmbedding>>& embeddings,
                             std::vector<SparseEmbeddingHashParams>& embedding_params,
                             std::vector<std::string>& sparse_embedding_files,
                             std::shared_ptr<ResourceManager> resource_manager,
                             std::vector<std::string>& local_paths,
                             std::vector<HMemCacheConfig>& hmem_cache_configs);

  EmbeddingTrainingCacheImpl(const EmbeddingTrainingCacheImpl&) = delete;
  EmbeddingTrainingCacheImpl& operator=(const EmbeddingTrainingCacheImpl&) = delete;

  ~EmbeddingTrainingCacheImpl() = default;

  /**
   * @brief Dump the downloaded embeddings from GPUs to sparse_model_entity_.
   */
  void dump() override;

  /**
   * @brief Updates the sparse_model_entity_ using embeddings from devices,
   *        then load embeddings to device memory according to the new keyset.
   * @param keyset_file_list The file list storing keyset files.
   */
  void update(std::vector<std::string>& keyset_file_list) override;

  /**
   * @brief Updates the sparse_model_entity_ using embeddings from devices,
   *        then load embeddings to device memory according to the new keyset.
   * @param keyset_file A single file storing keysets for all embeddings.
   */
  void update(std::string& keyset_file) override;

  std::vector<std::pair<std::vector<long long>, std::vector<float>>> get_incremental_model(
      const std::vector<long long>& keys_to_load) override;

  void update_sparse_model_file() override { ps_manager_.update_sparse_model_file(); }
};

}  // namespace HugeCTR

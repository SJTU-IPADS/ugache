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
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace coll_cache_lib {

template <typename TypeHashKey, typename TypeHashValue>
struct UnifiedEmbeddingTable {
  void* vectors_ptr;
  size_t umap_len = 0;
  std::vector<TypeHashValue> meta;
  size_t key_count = 0;
};

class IModelLoader {
 public:
  bool is_mock = false;
  static std::shared_ptr<IModelLoader> preserved_model_loader;
  ~IModelLoader() = default;
  virtual void master_load(const std::string& path) = 0;
  virtual void slave_load(const std::string& path) = 0;
  virtual void delete_table() = 0;
  virtual void* getvectors() = 0;
  virtual size_t getkeycount() = 0;
  IModelLoader() = default;
};

template <typename TKey, typename TValue>
class RawModelLoader : public IModelLoader {
 private:
  UnifiedEmbeddingTable<TKey, TValue>* embedding_table_;

 public:
  RawModelLoader();
  virtual void master_load(const std::string& path);
  virtual void slave_load(const std::string& path);
  virtual void delete_table();
  virtual void* getvectors();
  virtual size_t getkeycount();
  ~RawModelLoader() { delete_table(); }
};

template <typename TKey, typename TValue>
class ModelLoader {
 public:
  static IModelLoader* CreateLoader() {
    return new RawModelLoader<TKey, TValue>();
  }
};

}  // namespace coll_cache_lib
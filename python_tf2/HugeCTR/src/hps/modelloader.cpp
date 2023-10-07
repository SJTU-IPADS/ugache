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

#include <fcntl.h>
#include <sys/mman.h>

#include <common.hpp>
#include <cstddef>
#include <hps/inference_utils.hpp>
#include <hps/modelloader.hpp>
#include <parser.hpp>
#include <string>
#include <unordered_set>
#include <utils.hpp>

namespace HugeCTR {

std::shared_ptr<IModelLoader> IModelLoader::preserved_model_loader = nullptr;

template <typename TKey, typename TValue>
RawModelLoader<TKey, TValue>::RawModelLoader() : IModelLoader() {
  HCTR_LOG_S(DEBUG, WORLD) << "Created raw model loader in local memory!" << std::endl;
  embedding_table_ = new UnifiedEmbeddingTable<TKey, TValue>();
}

namespace {

std::string GetEnv(std::string key) {
  const char* env_var_val = getenv(key.c_str());
  if (env_var_val != nullptr) {
    return std::string(env_var_val);
  } else {
    return "";
  }
}

}  // namespace

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load(const std::string& table_name, const std::string& path) {
  const std::string emb_file_prefix = path + "/";
  const std::string key_file = emb_file_prefix + "key";
  const std::string vec_file = emb_file_prefix + "emb_vector";

  if (path.find("mock_") == 0) {
    this->is_mock = true;
    size_t num_key_offset = path.find('_') + 1, dim_offset = path.find_last_of('_') + 1;
    size_t num_key = std::stoull(path.substr(num_key_offset)),
           dim = std::stoull(path.substr(dim_offset));
    HCTR_LOG_S(ERROR, WORLD) << "using mock embedding with " << num_key << " * " << dim
                             << " elements\n";
    embedding_table_->key_count = num_key;
    embedding_table_->keys.resize(num_key);

    size_t vec_file_size_in_byte = sizeof(float) * num_key * dim;
    if (GetEnv("SAMGRAPH_EMPTY_FEAT") != "") {
      size_t empty_feat_num_key = 1 << std::stoull(GetEnv("SAMGRAPH_EMPTY_FEAT"));
      vec_file_size_in_byte = sizeof(float) * empty_feat_num_key * dim;
    }
    // std::string shm_name = std::string("HPS_VEC_FILE_SHM_") + getenv("USER");
    std::string shm_name = "SAMG_FEAT_SHM";
    HCTR_CHECK_HINT(getenv("HPS_WORKER_ID") != nullptr,
                    "Env HPS_WORKER_ID must be set before loading hps lib\n");
    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    HCTR_CHECK_HINT(fd != -1, "shm open vec file shm failed\n");
    size_t padded_size = (vec_file_size_in_byte + 0x01fffff) & ~0x01fffff;
    {
      struct stat st;
      fstat(fd, &st);
      if (st.st_size < padded_size) {
        int ret = ftruncate(fd, padded_size);
        HCTR_CHECK_HINT(ret != -1, "ftruncate vec file shm failed");
      }
    }
    embedding_table_->vectors_ptr = mmap(nullptr, padded_size,
                                         PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    HCTR_CHECK_HINT(embedding_table_->vectors_ptr != nullptr, "mmap vec file shm failed\n");
    embedding_table_->umap_len = vec_file_size_in_byte;

    return;
  }

  std::ifstream key_stream(key_file);
  std::ifstream vec_stream(vec_file);
  int vec_file_fd = open(vec_file.c_str(), O_RDONLY);
  if (!key_stream.is_open() || !vec_stream.is_open()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: embeddings file not open for reading");
  }

  const size_t key_file_size_in_byte = std::filesystem::file_size(key_file);
  const size_t vec_file_size_in_byte = std::filesystem::file_size(vec_file);

  const size_t key_size_in_byte = sizeof(long long);
  const size_t num_key = key_file_size_in_byte / key_size_in_byte;
  embedding_table_->key_count = num_key;

  const size_t num_float_val_in_vec_file = vec_file_size_in_byte / sizeof(float);

  // The temp embedding table
  embedding_table_->keys.resize(num_key);
  if (std::is_same<TKey, long long>::value) {
    key_stream.read(reinterpret_cast<char*>(embedding_table_->keys.data()), key_file_size_in_byte);
  } else {
    std::vector<long long> i64_key_vec(num_key, 0);
    key_stream.read(reinterpret_cast<char*>(i64_key_vec.data()), key_file_size_in_byte);
    /** Impl 1*/
    // #pragma omp parallel for
    // for (size_t i = 0; i < num_key; i++) {
    //   embedding_table_->keys[i] = i64_key_vec[i];
    // }
    /** Impl 2*/
    std::transform(i64_key_vec.begin(), i64_key_vec.end(), embedding_table_->keys.begin(),
                   [](long long key) { return static_cast<unsigned>(key); });
  }

  /** Impl 1*/
  // std::string shm_name = std::string("HPS_VEC_FILE_SHM_") + getenv("USER");
  std::string shm_name = "SAMG_FEAT_SHM";
  HCTR_CHECK_HINT(getenv("HPS_WORKER_ID") != nullptr,
                  "Env HPS_WORKER_ID must be set before loading hps lib\n");
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  HCTR_CHECK_HINT(fd != -1, "shm open vec file shm failed\n");
  size_t padded_size = (vec_file_size_in_byte + 0x01fffff) & ~0x01fffff;
  {
    struct stat st;
    fstat(fd, &st);
    if (st.st_size < padded_size) {
      int ret = ftruncate(fd, padded_size);
      HCTR_CHECK_HINT(ret != -1, "ftruncate vec file shm failed");
    }
  }
  embedding_table_->vectors_ptr = mmap(nullptr, padded_size,
                                       PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
  HCTR_CHECK_HINT(embedding_table_->vectors_ptr != nullptr, "mmap vec file shm failed\n");
  embedding_table_->umap_len = vec_file_size_in_byte;
  if (std::stoi(getenv("HPS_WORKER_ID")) == 0) {
    HCTR_LOG_S(ERROR, WORLD) << "I'm worker 0, I should read the data " << vec_file_size_in_byte
                             << "\n";
    vec_stream.read(reinterpret_cast<char*>(embedding_table_->vectors_ptr), vec_file_size_in_byte);
  }

  /** Impl 2*/
  // embedding_table_->vectors.resize((num_float_val_in_vec_file + 0x0fffff) & (~0x0fffff));
  // vec_stream.read(reinterpret_cast<char*>(embedding_table_->vectors.data()),
  // vec_file_size_in_byte);
  HCTR_LOG_S(ERROR, WORLD) << "raw read done\n";
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::delete_table() {
  std::vector<TKey>().swap(embedding_table_->keys);
  /** Impl 1*/
  munmap(embedding_table_->vectors_ptr, embedding_table_->umap_len);
  /** Impl 2*/
  // std::vector<TValue>().swap(embedding_table_->vectors);
  std::vector<TValue>().swap(embedding_table_->meta);
  delete embedding_table_;
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getkeys() {
  return embedding_table_->keys.data();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getvectors() {
  /** Impl 1*/
  return embedding_table_->vectors_ptr;
  /** Impl 2*/
  // return embedding_table_->vectors.data();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getmetas() {
  return embedding_table_->meta.data();
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::getkeycount() {
  return embedding_table_->key_count;
}

template class RawModelLoader<long long, float>;
template class RawModelLoader<unsigned int, float>;

}  // namespace HugeCTR

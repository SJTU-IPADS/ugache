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
#include <sys/stat.h>
#include <sys/mman.h>

#include <cstddef>
#include <inference_utils.hpp>
#include <modelloader.hpp>
#include <parser.hpp>
#include <string>
#include <unordered_set>

namespace coll_cache_lib {

std::shared_ptr<IModelLoader> IModelLoader::preserved_model_loader = nullptr;

template <typename TKey, typename TValue>
RawModelLoader<TKey, TValue>::RawModelLoader() : IModelLoader() {
  COLL_LOG(DEBUG) << "Created raw model loader in local memory!" << std::endl;
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
void RawModelLoader<TKey, TValue>::master_load(const std::string& path) {
  const std::string emb_file_prefix = path + "/";
  const std::string key_file = emb_file_prefix + "key";
  const std::string vec_file = emb_file_prefix + "emb_vector";

  size_t vec_file_size_in_byte = 0;
  size_t padded_size = 0;
  size_t num_key = 0;
  if (path.find("mock_") == 0) {
    this->is_mock = true;
    size_t num_key_offset = path.find('_') + 1,
           dim_offset     = path.find_last_of('_') + 1;
    size_t dim            = std::stoull(path.substr(dim_offset));
           num_key        = std::stoull(path.substr(num_key_offset));
    COLL_LOG(ERROR) << "using mock embedding with " << num_key << " * " << dim
                             << " elements\n";

    vec_file_size_in_byte = sizeof(float) * num_key * dim;
    if (GetEnv("SAMGRAPH_EMPTY_FEAT") != "") {
      size_t empty_feat_num_key = 1 << std::stoull(GetEnv("SAMGRAPH_EMPTY_FEAT"));
      vec_file_size_in_byte = sizeof(float) * empty_feat_num_key * dim;
    }
  } else {
    const size_t key_file_size_in_byte = std::filesystem::file_size(key_file);
    vec_file_size_in_byte = std::filesystem::file_size(vec_file);

    const size_t key_size_in_byte = sizeof(long long);
    num_key = key_file_size_in_byte / key_size_in_byte;
  }

  embedding_table_->key_count = num_key;
  std::string shm_name = "SAMG_FEAT_SHM";
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  COLL_CHECK(fd != -1) << "shm open vec file shm failed\n";
  padded_size = (vec_file_size_in_byte + 0x01fffff) & ~0x01fffff;
  {
    struct stat st;
    fstat(fd, &st);
    if (st.st_size < padded_size) {
      int ret = ftruncate(fd, padded_size);
      COLL_CHECK(ret != -1) << "ftruncate vec file shm failed";
    }
  }
  embedding_table_->vectors_ptr = mmap(nullptr, padded_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
  COLL_CHECK(embedding_table_->vectors_ptr != nullptr) << "mmap vec file shm failed\n";
  embedding_table_->umap_len = padded_size;

  if (path.find("mock_") == 0) {
  } else {
    COLL_LOG(ERROR) << "I'm worker 0, I should read the data " << vec_file_size_in_byte << "\n";
    std::ifstream vec_stream(vec_file);
    COLL_CHECK(vec_stream.is_open()) << "Error: embeddings file not open for reading";
    vec_stream.read(reinterpret_cast<char*>(embedding_table_->vectors_ptr), vec_file_size_in_byte);
  }
  COLL_LOG(ERROR) << "raw read done\n";
}
template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::slave_load(const std::string& path) {
  const std::string emb_file_prefix = path + "/";
  const std::string key_file = emb_file_prefix + "key";
  const std::string vec_file = emb_file_prefix + "emb_vector";

  size_t padded_size = 0;
  size_t vec_file_size_in_byte;
  size_t num_key;
  if (path.find("mock_") == 0) {
    this->is_mock = true;
    size_t num_key_offset = path.find('_') + 1,
           dim_offset     = path.find_last_of('_') + 1;
    size_t dim            = std::stoull(path.substr(dim_offset));
           num_key        = std::stoull(path.substr(num_key_offset));
    COLL_LOG(ERROR) << "using mock embedding with " << num_key << " * " << dim
                             << " elements\n";

    vec_file_size_in_byte = sizeof(float) * num_key * dim;
    if (GetEnv("SAMGRAPH_EMPTY_FEAT") != "") {
      size_t empty_feat_num_key = 1 << std::stoull(GetEnv("SAMGRAPH_EMPTY_FEAT"));
      vec_file_size_in_byte = sizeof(float) * empty_feat_num_key * dim;
    }
  } else {
    const size_t key_file_size_in_byte = std::filesystem::file_size(key_file);
    vec_file_size_in_byte = std::filesystem::file_size(vec_file);

    const size_t key_size_in_byte = sizeof(long long);
    num_key = key_file_size_in_byte / key_size_in_byte;
  }

  embedding_table_->key_count = num_key;
  std::string shm_name = "SAMG_FEAT_SHM";
  int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
  COLL_CHECK(fd != -1) << "shm open vec file shm failed\n";
  padded_size = (vec_file_size_in_byte + 0x01fffff) & ~0x01fffff;
  {
    struct stat st;
    fstat(fd, &st);
    if (st.st_size < padded_size) {
      int ret = ftruncate(fd, padded_size);
      COLL_CHECK(ret != -1) << "ftruncate vec file shm failed";
    }
  }
  embedding_table_->vectors_ptr = mmap(nullptr, padded_size, PROT_READ, MAP_SHARED, fd, 0);
  COLL_CHECK(embedding_table_->vectors_ptr != nullptr) << "mmap vec file shm failed\n";
  embedding_table_->umap_len = padded_size;

  COLL_LOG(ERROR) << "raw read done\n";
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::delete_table() {
  munmap(embedding_table_->vectors_ptr, embedding_table_->umap_len);
  std::vector<TValue>().swap(embedding_table_->meta);
  delete embedding_table_;
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getvectors() {
  return embedding_table_->vectors_ptr;
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::getkeycount() {
  return embedding_table_->key_count;
}

template class RawModelLoader<long long, float>;
template class RawModelLoader<unsigned int, float>;

}  // namespace coll_cache_lib

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

#include <algorithm>
#include <atomic>
#include <base/debug/logger.hpp>
#include <cstddef>
#include <cstring>
#include <execution>
#include <hps/direct_map_backend.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <random>

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

template <typename TKey>
DirectMapBackend<TKey>::DirectMapBackend(const size_t max_get_batch_size,
                                         const size_t max_set_batch_size,
                                         const size_t overflow_margin,
                                         const DatabaseOverflowPolicy_t overflow_policy,
                                         const double overflow_resolution_target)
    : TBase(max_get_batch_size, max_set_batch_size, overflow_margin, overflow_policy,
            overflow_resolution_target) {
  HCTR_LOG_S(DEBUG, WORLD) << "Created blank database backend in local memory!" << std::endl;
}

template <typename TPartition>
size_t DirectMapBackend<TPartition>::size(const std::string& table_name) const {
  HCTR_CHECK_HINT(table_name == single_table_name,
                  "this direct backend has only table %s, not %s\n", single_table_name.c_str(),
                  table_name.c_str());
  return num_keys_;
}

template <typename TKey>
size_t DirectMapBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys,
                                        const std::chrono::nanoseconds& time_budget) const {
  const auto begin = std::chrono::high_resolution_clock::now();
  HCTR_CHECK_HINT(table_name == single_table_name,
                  "this direct backend has only table %s, not %s\n", single_table_name.c_str(),
                  table_name.c_str());
  return num_keys;
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

template <typename TKey>
bool DirectMapBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                    const TKey* const keys, const char* const values,
                                    const size_t value_size) {
  bool already_inited = false;
  this->inited_.compare_exchange_strong(already_inited, true);
  if (already_inited) {
    HCTR_CHECK_HINT(false, "this direct map backend is already inited\n");
  }
  single_table_name = table_name;
  this->num_keys_ = num_pairs;
  this->val_ptr = values;
  this->val_len_nbytes = value_size;

  if (GetEnv("SAMGRAPH_EMPTY_FEAT") != "") {
    this->option_empty_feat = std::stoull(GetEnv("SAMGRAPH_EMPTY_FEAT"));
  }

  return true;
}

template <typename TKey>
size_t DirectMapBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                     const TKey* const keys, const DatabaseHitCallback& on_hit,
                                     const DatabaseMissCallback& on_miss,
                                     const std::chrono::nanoseconds& time_budget) {
  const auto begin = std::chrono::high_resolution_clock::now();
  HCTR_CHECK_HINT(table_name == single_table_name,
                  "this direct backend has only table %s, not %s\n", single_table_name.c_str(),
                  table_name.c_str());
  // std::unique_lock lock(this->read_write_guard_);

  // Spawn threads.
  std::vector<std::future<void>> tasks;
  tasks.reserve(parallel_level);

  const TKey* const keys_end = &keys[num_keys];
  for (int i = 0; i < parallel_level; i++) {
    tasks.emplace_back(ThreadPool::get().submit([&, i]() {
      // Traverse through keys, and fetch them one by one.
      size_t num_batches = 0;
      for (const TKey* k = keys; k != keys_end; num_batches++) {
        // Check time budget.
        const auto elapsed = std::chrono::high_resolution_clock::now() - begin;
        // Perform a bunch of queries.

        size_t batch_size = 0;
        for (; k != keys_end; k++) {
          if ((ulong)(*k) % parallel_level == i) {
            size_t src_off = (size_t)(*k);
            if (this->option_empty_feat != 0) {
              src_off = src_off % (1 << this->option_empty_feat);
            }
            on_hit((ulong)(k - keys), this->val_ptr + src_off * val_len_nbytes, val_len_nbytes);
            if (++batch_size >= this->max_get_batch_size_) {
              break;
            }
          }
        }

        HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ", partition "
                                 << i << ", batch " << num_batches << ": " << batch_size
                                 << " hits. Time: " << elapsed.count() << " / "
                                 << time_budget.count() << " ns." << std::endl;
      }
    }));
  }
  ThreadPool::await(tasks.begin(), tasks.end());

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": " << (num_keys)
                           << '.' << std::endl;
  return num_keys;
}

template <typename TKey>
size_t DirectMapBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                     const size_t* const indices, const TKey* const keys,
                                     const DatabaseHitCallback& on_hit,
                                     const DatabaseMissCallback& on_miss,
                                     const std::chrono::nanoseconds& time_budget) {
  HCTR_CHECK_HINT(table_name == single_table_name,
                  "this direct backend has only table %s, not %s\n", single_table_name.c_str(),
                  table_name.c_str());
  const auto begin = std::chrono::high_resolution_clock::now();
  // std::unique_lock lock(this->read_write_guard_);

  // Locate the partitions.
  switch (num_indices) {
    default: {
      // Precalc constants.
      const size_t* const indices_end = &indices[num_indices];
      {
        std::atomic<size_t> joint_hit_count{0};
        std::atomic<size_t> joint_ign_count{0};

        // Process partitions.
        std::vector<std::future<void>> tasks;
        tasks.reserve(parallel_level);
        for (int partidx = 0; partidx < parallel_level; partidx++) {
          tasks.emplace_back(ThreadPool::get().submit([&]() {
            const size_t* i = indices;
            for (size_t num_batches = 0; i != indices_end; num_batches++) {
              // Check time budget.
              const auto elapsed = std::chrono::high_resolution_clock::now() - begin;

              size_t batch_size = 0;
              for (; i != indices_end; i++) {
                const TKey& k = keys[*i];
                if ((ulong)(k) % parallel_level == partidx) {
                  size_t src_off = (size_t)(k);
                  if (this->option_empty_feat != 0) {
                    src_off = src_off % (1 << this->option_empty_feat);
                  }
                  on_hit((*i), val_ptr + src_off * val_len_nbytes, val_len_nbytes);
                  if (++batch_size >= this->max_get_batch_size_) {
                    break;
                  }
                }
              }

              HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name
                                       << ", partition " << partidx << ", batch " << num_batches
                                       << ": " << batch_size << " hits. Time: " << elapsed.count()
                                       << " / " << time_budget.count() << " ns." << std::endl;
            }
          }));
        }
        ThreadPool::await(tasks.begin(), tasks.end());
      }
    } break;
  }

  HCTR_LOG_S(TRACE, WORLD) << get_name() << " backend; Table " << table_name << ": "
                           << (num_indices) << " hits" << '.' << std::endl;
  return num_indices;
}

template <typename TKey>
size_t DirectMapBackend<TKey>::evict(const std::string& table_name) {
  HCTR_CHECK_HINT(false, "Unimplemented\n");
  return 0;
}

template <typename TKey>
size_t DirectMapBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                     const TKey* const keys) {
  HCTR_CHECK_HINT(false, "Unimplemented\n");
  return 0;
}

template <typename TKey>
std::vector<std::string> DirectMapBackend<TKey>::find_tables(const std::string& model_name) {
  const std::string& tag_prefix = HierParameterServerBase::make_tag_name(model_name, "", false);
  if (single_table_name.find(tag_prefix) == 0) {
    return {single_table_name};
  }
  return {};
}

template <typename TKey>
void DirectMapBackend<TKey>::dump_bin(const std::string& table_name, std::ofstream& file) {
  HCTR_CHECK_HINT(false, "Unimplemented\n");
}

template <typename TKey>
void DirectMapBackend<TKey>::dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) {
  HCTR_CHECK_HINT(false, "Unimplemented\n");
}

template <typename TKey>
size_t DirectMapBackend<TKey>::capacity(const std::string& table_name) const {
  return this->overflow_margin_;
}

template class DirectMapBackend<unsigned int>;
template class DirectMapBackend<long long>;

}  // namespace HugeCTR

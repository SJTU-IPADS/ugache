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

#include <parallel_hashmap/phmap.h>
#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <hps/database_backend.hpp>
#include <shared_mutex>
#include <thread>
#include <thread_pool.hpp>
#include <unordered_map>
#include <vector>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * \p DatabaseBackend implementation that stores key/value pairs in the local CPU memory.
 * that takes advantage of parallel processing capabilities.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey>
class DirectMapBackend final : public VolatileBackend<TKey> {
 public:
  using TBase = VolatileBackend<TKey>;
  /**
   * Construct a new parallelized DirectMapBackend object.
   * @param num_partitions The number of parallel partitions.
   * @param allocation_rate Number of additional bytes to allocate per allocation.
   * @param overflow_margin Margin at which further inserts will trigger overflow handling.
   * @param overflow_policy Policy to use in case an overflow has been detected.
   * @param overflow_resolution_target Target margin after applying overflow handling policy.
   */
  DirectMapBackend(size_t max_get_batch_size = 10'000, size_t max_set_batch_size = 10'000,
                   size_t overflow_margin = std::numeric_limits<size_t>::max(),
                   DatabaseOverflowPolicy_t overflow_policy = DatabaseOverflowPolicy_t::EvictOldest,
                   double overflow_resolution_target = 0.8);

  bool is_shared() const override final { return false; }

  const char* get_name() const override { return "DirectMapBackend"; }

  size_t capacity(const std::string& table_name) const override;

  size_t size(const std::string& table_name) const override;

  size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys,
                  const std::chrono::nanoseconds& time_budget) const override;

  bool insert(const std::string& table_name, size_t num_pairs, const TKey* keys, const char* values,
              size_t value_size) override;

  size_t fetch(const std::string& table_name, size_t num_keys, const TKey* keys,
               const DatabaseHitCallback& on_hit, const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
               const TKey* keys, const DatabaseHitCallback& on_hit,
               const DatabaseMissCallback& on_miss,
               const std::chrono::nanoseconds& time_budget) override;

  size_t evict(const std::string& table_name) override;

  size_t evict(const std::string& table_name, size_t num_keys, const TKey* keys) override;

  std::vector<std::string> find_tables(const std::string& model_name) override;

  void dump_bin(const std::string& table_name, std::ofstream& file) override;

  void dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) override;

 protected:
  size_t val_len_nbytes;
  const char* val_ptr;
  size_t num_keys_;
  std::string single_table_name;
  std::atomic_bool inited_{false};
  ulong parallel_level = 16;
  size_t option_empty_feat = 0;
};

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR
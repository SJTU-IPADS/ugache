/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
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
 *
 */

#pragma once

#include "coll_cache_lib/common.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace coll_cache_lib {
namespace common {

void coll_cache_init(int replica_id, size_t key_space_size, std::function<MemHandle(size_t)> allocator, void *cpu_data, DataType dtype, size_t dim, double cache_percentage, StreamHandle stream);

extern "C" {

void coll_cache_config(const char **config_keys, const char **config_values,
                     const size_t num_config_items);

size_t coll_cache_num_epoch();

// size_t coll_cache_steps_per_epoch();

void coll_cache_log_step_by_key(uint64_t key, int item, double val);

void coll_cache_report_step_by_key(uint64_t key);

void coll_cache_report_step_average_by_key(uint64_t key);


void coll_cache_train_barrier();

// size_t coll_cache_num_local_step();

int coll_cache_wait_one_child();

void coll_cache_print_memory_usage();

void coll_cache_lookup(int replica_id, uint32_t* key, size_t num_keys, void* output, StreamHandle stream);

void coll_cache_record(int replica_id, uint32_t* key, size_t num_keys);
}

}  // namespace common
}  // namespace coll_cache_lib

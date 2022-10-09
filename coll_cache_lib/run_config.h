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

#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONFIG_H

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "coll_cache/asymm_link_desc.h"

namespace coll_cache_lib {
namespace common {

struct RunConfig {
  static std::unordered_map<std::string, std::string> raw_configs;

  // Configs passed from application
  // clang-format off
  static CachePolicy          cache_policy;
  static double               cache_percentage;

  // model parameters
  static size_t               hiddem_dim;
  static double               dropout;
  static double               lr;

  static bool                 is_configured;

  // For multi-gpu sampling and training
  static size_t               worker_id;
  static size_t               num_worker;
  static size_t               num_device;

  // Environment variables
  static bool                 option_profile_cuda;
  static bool                 option_log_node_access;
  static bool                 option_log_node_access_simple;
  static bool                 option_dump_trace;
  static size_t               option_empty_feat;

  static size_t               option_fake_feat_dim;

  static int                  omp_thread_num;

  // shared memory meta_data path for data communication acrossing processes
  static std::string          shared_meta_path;
  // clang-format on

  static bool                 coll_cache_concurrent_link;
  static bool                 coll_cache_no_group;
  static size_t               coll_cache_num_slot;
  static double               coll_cache_coefficient;
  static double               coll_cache_hyperparam_T_local;
  static double               coll_cache_hyperparam_T_remote;
  static double               coll_cache_hyperparam_T_cpu;

  static coll_cache::AsymmLinkDesc coll_cache_link_desc;

  static RollingPolicy        rolling;

  static inline bool UseGPUCache() {
    return cache_percentage > 0;
  }

  static inline bool UseDynamicGPUCache() {
    return cache_policy == kDynamicCache;
  }

  static void LoadConfigFromEnv();
};

}  // namespace common
}  // namespace coll_cache_lib

#endif  // SAMGRAPH_RUN_CONFIG_H

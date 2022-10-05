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

#include "run_config.h"

#include "constant.h"
#include "logging.h"
#include <unordered_set>

namespace samgraph {
namespace common {

std::unordered_map<std::string, std::string> RunConfig::raw_configs;

// clang-format off
RunArch              RunConfig::run_arch;
Context              RunConfig::sampler_ctx;
Context              RunConfig::trainer_ctx;
CachePolicy          RunConfig::cache_policy;
double               RunConfig::cache_percentage               = 0.0f;

bool                 RunConfig::is_configured                  = false;

size_t               RunConfig::num_sample_worker;
size_t               RunConfig::num_train_worker;

// For arch7
size_t               RunConfig::worker_id                      = false;
size_t               RunConfig::num_worker                     = false;

bool                 RunConfig::option_profile_cuda            = false;
bool                 RunConfig::option_log_node_access         = false;
bool                 RunConfig::option_log_node_access_simple  = false;

bool                 RunConfig::option_dump_trace              = false;
size_t               RunConfig::option_empty_feat              = 0;

size_t               RunConfig::option_fake_feat_dim = 0;

int                  RunConfig::omp_thread_num                 = 40;

std::string          RunConfig::shared_meta_path               = "/shared_meta_data";
// clang-format on

bool                 RunConfig::coll_cache_concurrent_link = false;
bool                 RunConfig::coll_cache_no_group    = false;
size_t               RunConfig::coll_cache_num_slot    = 100;
double               RunConfig::coll_cache_coefficient = 1.1;
double               RunConfig::coll_cache_hyperparam_T_local  = 1;
double               RunConfig::coll_cache_hyperparam_T_remote = 438 / (double)213;  // performance on A100
double               RunConfig::coll_cache_hyperparam_T_cpu    = 438 / (double)11.8; // performance on A100
coll_cache::AsymmLinkDesc RunConfig::coll_cache_link_desc;

RollingPolicy        RunConfig::rolling = AutoRolling;

void RunConfig::LoadConfigFromEnv() {
  std::unordered_set<std::string> ture_values = {"TRUE", "1", "ON"};
  if (IsEnvSet(Constant::kEnvProfileCuda)) {
    RunConfig::option_profile_cuda = true;
  }

  if (IsEnvSet(Constant::kEnvLogNodeAccessSimple)) {
    RunConfig::option_log_node_access_simple = true;
  }

  if (IsEnvSet(Constant::kEnvLogNodeAccess)) {
    RunConfig::option_log_node_access = true;
  }

  if (IsEnvSet(Constant::kEnvDumpTrace)) {
    RunConfig::option_dump_trace = true;
  }
  if (GetEnv(Constant::kEnvEmptyFeat) != "") {
    RunConfig::option_empty_feat = std::stoul(GetEnv(Constant::kEnvEmptyFeat));
  }

  if (GetEnv(Constant::kEnvFakeFeatDim) != "") {
    std::string env = GetEnv(Constant::kEnvFakeFeatDim);
    RunConfig::option_fake_feat_dim = std::stoi(env);
  }
  if (IsEnvSet("SAMGRAPH_COLL_CACHE_NO_GROUP")) {
    RunConfig::coll_cache_no_group = true;
  }
  if (GetEnv("SAMGRAPH_COLL_CACHE_CONCURRENT_LINK") != "") {
    RunConfig::coll_cache_concurrent_link = ture_values.find(GetEnv("SAMGRAPH_COLL_CACHE_CONCURRENT_LINK")) != ture_values.end();
  } else {
    // auto enable coll cache concurrent link
    RunConfig::coll_cache_concurrent_link = coll_cache::AutoEnableConcurrentLink();
  }
}


}  // namespace common
}  // namespace samgraph

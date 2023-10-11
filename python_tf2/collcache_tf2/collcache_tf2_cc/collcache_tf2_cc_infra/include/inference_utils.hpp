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
#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace coll_cache_lib {

struct InferenceParams {
  bool use_multi_worker = false;
  size_t coll_cache_refresh_iter;
  bool coll_cache_enable_refresh = false;
  std::string model_name;
  size_t max_batchsize;
  std::vector<std::string> sparse_model_files;
  int device_id;
  float cache_size_percentage;
  bool i64_input_key;
  std::vector<int> deployed_devices;
  std::vector<int> cross_worker_deployed_devices;
  std::vector<size_t> max_vocabulary_size;
  // HPS required parameters
  std::vector<size_t> maxnum_catfeature_query_per_table_per_sample;
  std::vector<size_t> embedding_vecsize_per_table;

  InferenceParams(const std::string& model_name, size_t max_batchsize,
                  const std::vector<std::string>& sparse_model_files, int device_id,
                  float cache_size_percentage,
                  bool i64_input_key = true,
                  const std::vector<int>& deployed_devices = {0},
                  // HPS required parameters
                  const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample = {26},
                  const std::vector<size_t>& embedding_vecsize_per_table = {128});
};

struct parameter_server_config {
  bool use_multi_worker = false;
  size_t coll_cache_refresh_iter = std::numeric_limits<size_t>::max();
  bool coll_cache_enable_refresh = false;
  size_t iteration_per_epoch = 0;
  size_t epoch = 0;
  int coll_cache_policy = 13;
  // Each vector should have size of M(# of models), where each element in the vector should be a
  // vector with size E(# of embedding tables in that model)
  std::map<std::string, std::vector<size_t>>
      embedding_vec_size_;  // The emb_vec_size per embedding table per model
  std::map<std::string, std::vector<size_t>>
      max_feature_num_per_sample_per_emb_table_;  // The max # of keys in each sample per table per
                                                  // model
  std::vector<InferenceParams>
      inference_params_array;  //// model configuration of all models deployed on HPS, e.g.,
                               ///{dcn_inferenceParamesStruct}

  // Database backend.
  parameter_server_config(
      std::map<std::string, std::vector<size_t>> embedding_vec_size,
      std::map<std::string, std::vector<size_t>> max_feature_num_per_sample_per_emb_table,
      const std::vector<InferenceParams>& inference_params_array);
  parameter_server_config(const std::vector<std::string>& model_config_path_array,
                          const std::vector<InferenceParams>& inference_params_array);
  parameter_server_config(const std::string& hps_json_config_file);
  parameter_server_config(const char* hps_json_config_file);
  void init(const std::string& hps_json_config_file);
};
}  // namespace coll_cache_lib
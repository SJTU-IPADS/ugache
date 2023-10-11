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

#include <inference_utils.hpp>
#include <parser.hpp>
#include <unordered_set>

namespace coll_cache_lib {

InferenceParams::InferenceParams(
    const std::string& model_name, const size_t max_batchsize,
    const std::vector<std::string>& sparse_model_files,
    const int device_id, const float cache_size_percentage,
    const bool i64_input_key,
    const std::vector<int>& deployed_devices,
    // Database backend.
    // HPS required
    const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample,
    const std::vector<size_t>& embedding_vecsize_per_table)
    : model_name(model_name),
      max_batchsize(max_batchsize),
      sparse_model_files(sparse_model_files),
      device_id(device_id),
      cache_size_percentage(cache_size_percentage),
      i64_input_key(i64_input_key),
      deployed_devices(deployed_devices),
      // HPS required
      maxnum_catfeature_query_per_table_per_sample(maxnum_catfeature_query_per_table_per_sample),
      embedding_vecsize_per_table(embedding_vecsize_per_table) {
  max_vocabulary_size.resize(embedding_vecsize_per_table.size());
}

parameter_server_config::parameter_server_config(const char* hps_json_config_file) {
  init(std::string(hps_json_config_file));
}

parameter_server_config::parameter_server_config(const std::string& hps_json_config_file) {
  init(hps_json_config_file);
}

void parameter_server_config::init(const std::string& hps_json_config_file) {
  COLL_LOG(INFO) << 
             "=====================================================HPS "
             "Parse====================================================\n";

  // Initialize for each model
  // Open model config file and input model json config
  nlohmann::json hps_config(read_json_file(hps_json_config_file));
  this->use_multi_worker = get_value_from_json_soft<bool>(hps_config, "use_multi_worker", false);
  if (hps_config.find("coll_cache_refresh_iter") != hps_config.end()) {
    this->coll_cache_refresh_iter =
        get_value_from_json<size_t>(hps_config, "coll_cache_refresh_iter");
  }
  this->coll_cache_enable_refresh =
      get_value_from_json_soft<bool>(hps_config, "coll_cache_enable_refresh", false);
  this->iteration_per_epoch = get_value_from_json<size_t>(hps_config, "iteration_per_epoch");
  this->epoch = get_value_from_json<size_t>(hps_config, "epoch");
  this->coll_cache_policy = get_value_from_json_soft<int>(hps_config, "coll_cache_policy", 13);

  // Search for all model configuration
  const nlohmann::json& models = get_json(hps_config, "models");
  COLL_CHECK(models.size() > 0) <<
                  "No model configurations in JSON. Is the file formatted correctly?";
  for (size_t j = 0; j < models.size(); j++) {
    const nlohmann::json& model = models[j];
    // [0] model_name -> std::string
    std::string model_name = get_value_from_json_soft<std::string>(model, "model", "");
    // [1] max_batch_size -> size_t
    size_t max_batch_size = get_value_from_json_soft<size_t>(model, "max_batch_size", 64);
    // [2] sparse_model_files -> std::vector<std::string>
    auto sparse_model_files = get_json(model, "sparse_files");
    std::vector<std::string> sparse_files;
    if (sparse_model_files.is_array()) {
      for (size_t sparse_id = 0; sparse_id < sparse_model_files.size(); ++sparse_id) {
        sparse_files.emplace_back(sparse_model_files[sparse_id].get<std::string>());
      }
    }
    // [5] cache_size_percentage -> float
    float cache_size_percentage = get_value_from_json_soft<float>(model, "gpucacheper", 0.2);

    // [7] device_id -> int
    const int device_id = 0;

    InferenceParams params(model_name, max_batch_size, sparse_files,
                           device_id, cache_size_percentage, true);
    params.use_multi_worker = this->use_multi_worker;
    params.coll_cache_refresh_iter = this->coll_cache_refresh_iter;
    params.coll_cache_enable_refresh = this->coll_cache_enable_refresh;
    params.i64_input_key = get_value_from_json_soft<bool>(model, "i64_input_key", true);

    // [11] deployed_device_list -> std::vector<int>
    auto deployed_device_list = get_json(model, "deployed_device_list");
    params.deployed_devices.clear();
    if (deployed_device_list.is_array()) {
      for (size_t device_index = 0; device_index < deployed_device_list.size(); ++device_index) {
        params.deployed_devices.emplace_back(deployed_device_list[device_index].get<int>());
      }
    }
    params.cross_worker_deployed_devices = params.deployed_devices;
    if (use_multi_worker) {
      params.deployed_devices = {params.deployed_devices[std::stoi(getenv("HPS_WORKER_ID"))]};
    }
    params.device_id = params.deployed_devices.back();
    // [13] maxnum_catfeature_query_per_table_per_sample -> std::vector<int>
    auto maxnum_catfeature_query_per_table_per_sample =
        get_json(model, "maxnum_catfeature_query_per_table_per_sample");
    params.maxnum_catfeature_query_per_table_per_sample.clear();
    if (maxnum_catfeature_query_per_table_per_sample.is_array()) {
      for (size_t cat_index = 0; cat_index < maxnum_catfeature_query_per_table_per_sample.size();
           ++cat_index) {
        params.maxnum_catfeature_query_per_table_per_sample.emplace_back(
            maxnum_catfeature_query_per_table_per_sample[cat_index].get<size_t>());
      }
    }

    // [14] embedding_vecsize_per_table -> std::vector<size_t>
    auto embedding_vecsize_per_table = get_json(model, "embedding_vecsize_per_table");
    params.embedding_vecsize_per_table.clear();
    if (embedding_vecsize_per_table.is_array()) {
      for (size_t vecsize_index = 0; vecsize_index < embedding_vecsize_per_table.size();
           ++vecsize_index) {
        params.embedding_vecsize_per_table.emplace_back(
            embedding_vecsize_per_table[vecsize_index].get<size_t>());
      }
    }
    if (model.find("max_vocabulary_size") != model.end()) {
      auto max_vocabulary_size = get_json(model, "max_vocabulary_size");
      params.max_vocabulary_size.clear();
      if (max_vocabulary_size.is_array()) {
        for (size_t name_index = 0; name_index < max_vocabulary_size.size(); ++name_index) {
          params.max_vocabulary_size.emplace_back(max_vocabulary_size[name_index].get<size_t>());
        }
      }
    } else {
      params.max_vocabulary_size.resize(sparse_model_files.size(), 0);
    }

    inference_params_array.emplace_back(params);

    // Fill the ps required parameters
    embedding_vec_size_[params.model_name] = params.embedding_vecsize_per_table;
    max_feature_num_per_sample_per_emb_table_[params.model_name] =
        params.maxnum_catfeature_query_per_table_per_sample;
  }
}

parameter_server_config::parameter_server_config(
    std::map<std::string, std::vector<size_t>> embedding_vec_size,
    std::map<std::string, std::vector<size_t>> max_feature_num_per_sample_per_emb_table,
    const std::vector<InferenceParams>& inference_params_array) {
  if (embedding_vec_size.size() != inference_params_array.size() ||
      max_feature_num_per_sample_per_emb_table.size() != inference_params_array.size()) {
    COLL_LOG(FATAL) <<
                   "Wrong input: The number of model names and inference_params_array "
                   "are not consistent.";
  }
  for (size_t i = 0; i < inference_params_array.size(); i++) {
    const auto& inference_params = inference_params_array[i];
    if (embedding_vec_size.find(inference_params.model_name) == embedding_vec_size.end() ||
        max_feature_num_per_sample_per_emb_table.find(inference_params.model_name) ==
            max_feature_num_per_sample_per_emb_table.end()) {
      COLL_LOG(FATAL) << "Wrong input: The model_name does not exist in the map.";
    }

    // Read inference config
    embedding_vec_size_[inference_params.model_name] =
        embedding_vec_size[inference_params.model_name];
    max_feature_num_per_sample_per_emb_table_[inference_params.model_name] =
        max_feature_num_per_sample_per_emb_table[inference_params.model_name];
  }  // end for
  this->inference_params_array = inference_params_array;
}

parameter_server_config::parameter_server_config(
    const std::vector<std::string>& model_config_path_array,
    const std::vector<InferenceParams>& inference_params_array) {
  if (model_config_path_array.size() != inference_params_array.size()) {
    COLL_LOG(FATAL) <<
                   "Wrong input: The size of model_config_path_array and inference_params_array "
                   "are not consistent.";
  }
  for (size_t i = 0; i < model_config_path_array.size(); i++) {
    const auto& model_config_path = model_config_path_array[i];
    const auto& inference_params = inference_params_array[i];

    // Initialize for each model
    // Open model config file and input model json config
    nlohmann::json model_config(read_json_file(model_config_path));

    // Read embedding layer config
    std::vector<size_t> embedding_vec_size;
    std::vector<size_t> max_feature_num_per_sample_per_emb_table;

    // Search for all embedding layers
    const nlohmann::json& layers = get_json(model_config, "layers");
    for (size_t j = 0; j < layers.size(); j++) {
      const nlohmann::json& layer = layers[j];
      std::string layer_type = get_value_from_json<std::string>(layer, "type");
      if (layer_type.compare("Data") == 0) {
        const nlohmann::json& sparse_inputs = get_json(layer, "sparse");
        for (size_t k = 0; k < sparse_inputs.size(); k++) {
          max_feature_num_per_sample_per_emb_table.push_back(
              get_max_feature_num_per_sample_from_nnz_per_slot(sparse_inputs[k]));
        }
      } else if (layer_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
        const nlohmann::json& embedding_hparam = get_json(layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
      } else if (layer_type.compare("LocalizedSlotSparseEmbeddingHash") == 0 ||
                 layer_type.compare("LocalizedSlotSparseEmbeddingOneHot") == 0) {
        const nlohmann::json& embedding_hparam = get_json(layer, "sparse_embedding_hparam");
        embedding_vec_size.emplace_back(
            get_value_from_json<size_t>(embedding_hparam, "embedding_vec_size"));
      } else {
        break;
      }
    }
    embedding_vec_size_[inference_params.model_name] = embedding_vec_size;
    max_feature_num_per_sample_per_emb_table_[inference_params.model_name] =
        max_feature_num_per_sample_per_emb_table;
  }  // end for
  this->inference_params_array = inference_params_array;
}

}  // namespace coll_cache_lib

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
#include "logging.hpp"
#include <fstream>
#include <functional>
#include <inference_utils.hpp>
#include <nlohmann/json.hpp>

namespace coll_cache_lib {
enum class DataReaderType_t { Norm, Raw, Parquet, RawAsync };

// inline to avoid build error: multiple definition
inline nlohmann::json read_json_file(const std::string& filename) {
  nlohmann::json config;
  std::ifstream file_stream(filename);
  if (!file_stream.is_open()) {
    COLL_LOG(FATAL) << "file_stream.is_open() failed: " + filename;
  }
  file_stream >> config;
  file_stream.close();
  return config;
}

#define HAS_KEY_(j_in, key_in)                                               \
  do {                                                                       \
    const nlohmann::json& j__ = (j_in);                                      \
    const std::string& key__ = (key_in);                                     \
    if (j__.find(key__) == j__.end())                                        \
      COLL_LOG(FATAL) << "[Parser] No Such Key: " + key__; \
  } while (0)

#define CK_SIZE_(j_in, j_size)                                             \
  do {                                                                     \
    const nlohmann::json& j__ = (j_in);                                    \
    if (j__.size() != (j_size))                                            \
      COLL_LOG(FATAL) << "[Parser] Array size is wrong"; \
  } while (0)

inline bool has_key_(const nlohmann::json& j_in, const std::string& key_in) {
  if (j_in.find(key_in) == j_in.end()) {
    return false;
  } else {
    return true;
  }
}

inline const nlohmann::json& get_json(const nlohmann::json& json, const std::string key) {
  HAS_KEY_(json, key);
  return json.find(key).value();
}

template <typename T>
inline T get_value_from_json(const nlohmann::json& json, const std::string key) {
  HAS_KEY_(json, key);
  auto value = json.find(key).value();
  CK_SIZE_(value, 1);
  return value.get<T>();
}

template <typename T>
inline T get_value_from_json_soft(const nlohmann::json& json, const std::string key,
                                  T default_value) {
  if (has_key_(json, key)) {
    auto value = json.find(key).value();
    CK_SIZE_(value, 1);
    return value.get<T>();
  } else {
    COLL_LOG(INFO) << key << " is not specified using default: " << default_value;
    return default_value;
  }
}

template <>
inline std::string get_value_from_json_soft(const nlohmann::json& json, const std::string key,
                                            const std::string default_value) {
  if (has_key_(json, key)) {
    auto value = json.find(key).value();
    CK_SIZE_(value, 1);
    return value.get<std::string>();
  } else {
    COLL_LOG(INFO) << key << " is not specified using default: " << default_value;
    return default_value;
  }
}

}  // namespace HugeCTR

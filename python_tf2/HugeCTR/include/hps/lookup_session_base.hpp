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
#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
#include <memory>

#include "coll_cache_lib/freq_recorder.h"

namespace HugeCTR {

class LookupSessionBase {
 public:
  virtual ~LookupSessionBase() = default;
  LookupSessionBase() = default;
  LookupSessionBase(LookupSessionBase const&) = delete;
  LookupSessionBase& operator=(LookupSessionBase const&) = delete;

  virtual void lookup(const void* h_keys, float* d_vectors, size_t num_keys, size_t table_id) = 0;
  virtual void lookup(const std::vector<const void*>& h_keys_per_table,
                      const std::vector<float*>& d_vectors_per_table,
                      const std::vector<size_t>& num_keys_per_table) = 0;
  virtual const InferenceParams get_inference_params() const = 0;

  static std::shared_ptr<LookupSessionBase> create(
      const InferenceParams& inference_params,
      const std::shared_ptr<EmbeddingCacheBase>& embedding_cache);
  std::shared_ptr<coll_cache_lib::common::FreqRecorder> freq_recorder_;
};

}  // namespace HugeCTR
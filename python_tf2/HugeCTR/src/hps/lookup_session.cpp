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

#include <hps/lookup_session.hpp>

#include "coll_cache_lib/freq_recorder.h"

namespace coll_cache_lib {

// LookupSessionBase::~LookupSessionBase() = default;

std::shared_ptr<LookupSessionBase> LookupSessionBase::create(
    const InferenceParams& inference_params,
    const std::shared_ptr<EmbeddingCacheBase>& embedding_cache) {
  return std::make_shared<LookupSession>(inference_params, embedding_cache);
}

LookupSession::LookupSession(const InferenceParams& inference_params,
                             const std::shared_ptr<EmbeddingCacheBase>& embedding_cache)
    : LookupSessionBase(), embedding_cache_(embedding_cache), inference_params_(inference_params) {
  try {
    if (inference_params_.use_coll_cache) {
      this->freq_recorder_ = std::make_shared<coll_cache_lib::common::FreqRecorder>(
          inference_params_.max_vocabulary_size[0], inference_params.device_id);
    }
  } catch (const std::runtime_error& rt_err) {
  }
  return;
}

LookupSession::~LookupSession() {
}

void LookupSession::lookup(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                           const size_t table_id) {
  if (freq_recorder_) {
    if (inference_params_.i64_input_key) {
      freq_recorder_->Record(reinterpret_cast<const coll_cache_lib::common::Id64Type*>(h_keys),
                             num_keys);
    } else {
      freq_recorder_->Record(reinterpret_cast<const coll_cache_lib::common::IdType*>(h_keys),
                             num_keys);
    }
  }
}

}  // namespace coll_cache_lib
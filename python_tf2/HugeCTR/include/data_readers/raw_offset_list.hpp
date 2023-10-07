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
#include <unistd.h>

#include <algorithm>
#include <random>
#include <vector>

namespace HugeCTR {

struct FileOffset {
  char* offset;
  long long samples;
};

class RawOffsetList {
 private:
  const long long num_samples_;
  const long long stride_;
  const long long batchsize_;
  const bool use_shuffle_;
  std::vector<FileOffset> offsets_;
  std::atomic<long long> counter_{0};
  const int num_workers_;
  bool repeat_;
  std::string file_name_;

 public:
  // stride: samle size in byte
  RawOffsetList(std::string file_name, long long num_samples, long long stride, long long batchsize,
                bool use_shuffle, int num_workers, bool repeat)
      : num_samples_(num_samples),
        stride_(stride),
        batchsize_(batchsize),
        use_shuffle_(use_shuffle),
        num_workers_(num_workers),
        repeat_(repeat),
        file_name_(file_name) {
    try {
      auto offset_gen = [stride](long long idx, long long samples) -> FileOffset {
        char* offset = (char*)(idx * stride);
        return {offset, samples};
      };
      for (long long sample_idx = 0; sample_idx < num_samples; sample_idx += batchsize) {
        if (sample_idx + batchsize <= num_samples) {
          offsets_.emplace_back(offset_gen(sample_idx, batchsize));
        } else {
          offsets_.emplace_back(offset_gen(sample_idx, num_samples - sample_idx));
        }
      }
      // shuffle
      if (use_shuffle) {
        std::random_device rd;
        unsigned int seed = rd();

#ifdef ENABLE_MPI
        HCTR_MPI_THROW(MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
#endif
        auto rng = std::default_random_engine{seed};
        std::shuffle(std::begin(offsets_), std::end(offsets_), rng);
      }

    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
      throw;
    }
  }

  ~RawOffsetList() {}

  FileOffset get_offset(long long round, int worker_id) {
    size_t worker_pos = round * num_workers_ + worker_id;
    if (!repeat_ && worker_pos >= offsets_.size()) {
      throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
    }
    size_t counter = (round * num_workers_ + worker_id) % offsets_.size();
    if (worker_id >= num_workers_) {
      HCTR_OWN_THROW(Error_t::WrongInput, "worker_id >= num_workers_");
    }
    if (counter == offsets_.size() - 1) {
      // HCTR_OWN_THROW(Error_t::OutOfBound, "End of File");
      HCTR_LOG_S(WARNING, WORLD) << "End of File, worker:  " << worker_id << std::endl;
    }
    return offsets_[counter];
  }

  std::string get_file_name() { return file_name_; }
  long long get_stride() { return stride_; }
  long long get_batch_size() { return batchsize_; }
};
}  // namespace HugeCTR

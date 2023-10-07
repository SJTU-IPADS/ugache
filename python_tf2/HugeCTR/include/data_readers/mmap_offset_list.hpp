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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <random>
#include <vector>

namespace HugeCTR {

struct MmapOffset {
  char* offset;
  long long samples;
};

/**
 * @brief A threads safe file list implementation.
 *
 * FileList reads file list from text file, and maintains a vector of file name. It supports
 * getting file names with multiple threads. All the threads will get the names in order.
 * Text file begins with the number of files, and then the list of file names.
 * @verbatim
 * Text file example:
 * 3
 * 1.txt
 * 2.txt
 * 3.txt
 * @endverbatim
 */
class MmapOffsetList {
 private:
  const long long length_;
  std::vector<MmapOffset> offsets_;
  std::atomic<long long> counter_{0};
  const int num_workers_;
  bool repeat_;
  char* mmapped_data_;
  int fd_;

 public:
  // stride: samle size in byte
  MmapOffsetList(std::string file_name, long long num_samples, long long stride,
                 long long batchsize, bool use_shuffle, int num_workers, bool repeat)
      : length_(num_samples * stride), num_workers_(num_workers), repeat_(repeat) {
    try {
      fd_ = open(file_name.c_str(), O_RDONLY, 0);
      if (fd_ == -1) {
        HCTR_OWN_THROW(Error_t::BrokenFile, "Error open file for read");
        return;
      }

      /* Get the size of the file. */
      mmapped_data_ = (char*)mmap(0, length_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (mmapped_data_ == MAP_FAILED) {
        close(fd_);
        HCTR_OWN_THROW(Error_t::BrokenFile, "Error mmapping the file");
        return;
      }

      auto offset_gen = [stride](char* mmapped_data, long long idx,
                                 long long samples) -> MmapOffset {
        char* offset = mmapped_data + idx * stride;
        return {offset, samples};
      };

      offsets_.reserve(num_samples);
      for (long long sample_idx = 0; sample_idx < num_samples; sample_idx += batchsize) {
        if (sample_idx + batchsize <= num_samples) {
          offsets_.emplace_back(offset_gen(mmapped_data_, sample_idx, batchsize));
        } else {
          offsets_.emplace_back(offset_gen(mmapped_data_, sample_idx, num_samples - sample_idx));
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

  ~MmapOffsetList() {
    munmap(mmapped_data_, length_);
    close(fd_);
  }

  MmapOffset get_offset(long long round, int worker_id) {
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
      HCTR_LOG_S(INFO, WORLD) << "End of File, worker:  " << worker_id << std::endl;
    }
    return offsets_[counter];
  }
};
}  // namespace HugeCTR

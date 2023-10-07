/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <nccl.h>

#include <memory>

#include "macro.hpp"

namespace core {
class Device;
class BufferBlockImpl;
class BufferImpl;

using BufferBlockPtr = std::shared_ptr<BufferBlockImpl>;
using BufferPtr = std::shared_ptr<BufferImpl>;

class IStorageImpl {
 public:
  virtual ~IStorageImpl() = default;

  virtual void *get_ptr() = 0;

  virtual size_t nbytes() const = 0;

  virtual void extend(size_t nbytes) = 0;

  virtual void allocate() = 0;
};

using Storage = std::shared_ptr<IStorageImpl>;

class GPUResourceBase {
 public:
  virtual ~GPUResourceBase() = default;
  virtual void set_stream(const std::string &name) = 0;
  virtual std::string get_current_stream_name() = 0;
  virtual cudaStream_t get_stream() = 0;  // will return current stream
};

class StreamContext {
  std::string origin_stream_name_;
  std::shared_ptr<GPUResourceBase> local_gpu_;

 public:
  StreamContext(const std::shared_ptr<GPUResourceBase> &local_gpu,
                const std::string &new_stream_name)
      : origin_stream_name_(local_gpu->get_current_stream_name()) {
    local_gpu_->set_stream(new_stream_name);
  }
  ~StreamContext() { local_gpu_->set_stream(origin_stream_name_); }
};

class CoreResourceManager {
 public:
  virtual ~CoreResourceManager() = default;

  virtual std::shared_ptr<GPUResourceBase> get_local_gpu() = 0;

  virtual const ncclComm_t &get_nccl() const = 0;

  virtual Storage CreateStorage(Device device) = 0;

  virtual int get_local_gpu_id() const = 0;

  virtual int get_global_gpu_id() const = 0;

  virtual int get_device_id() const = 0;

  virtual size_t get_local_gpu_count() const = 0;

  virtual size_t get_global_gpu_count() const = 0;

  virtual int get_gpu_global_id_from_local_id(int local_id) const = 0;

  virtual int get_gpu_local_id_from_global_id(int global_id) const = 0;
};

}  // namespace core
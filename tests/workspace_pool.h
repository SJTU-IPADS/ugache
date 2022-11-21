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

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>

#include "common.h"
#include "device.h"

namespace coll_cache_lib {
namespace common {

class WorkspacePool {
 public:
  WorkspacePool(std::function<MemHandle(size_t)> internal_allocator);
  ~WorkspacePool();

  MemHandle AllocWorkspace(size_t size, double scale = Constant::kAllocScale);

 private:

  class Pool;
  class WorkspacePoolMemHandle : public ExternelGPUMemoryHandler {
   public:
    void* data_;
    size_t nbytes_;
    // MemHandle external_handle_;
    // void* ptr() override {return external_handle_->ptr();}
    // size_t nbytes() override { return external_handle_->nbytes(); }
    void* ptr() override {return data_;}
    size_t nbytes() override { return nbytes_; }
    Pool* pool_;
    ~WorkspacePoolMemHandle();
  };
  Pool* _pool;
  std::function<MemHandle(size_t)> _internal_allocator;
  std::mutex _mutex;
};

}  // namespace common
}  // namespace samgraph

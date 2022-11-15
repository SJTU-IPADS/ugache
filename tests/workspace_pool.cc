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

#include "workspace_pool.h"

#include <memory>
#include <unordered_map>

#include "common.h"
#include "device.h"
#include "logging.h"

namespace coll_cache_lib {
namespace common {

// page size.
constexpr size_t kWorkspacePageSize = 4 << 10;

class WorkspacePool::Pool {
 public:
  std::function<MemHandle(size_t)> _internal_allocator;
  Pool() {
    // List gurad
    Entry e;
    e.data = nullptr;
    e.size = 0;

    _free_list.reserve(kListSize);
    _allocated.reserve(kListSize);

    _free_list.push_back(e);
    _allocated.push_back(e);
  }

  // allocate from pool
  void* Alloc(size_t nbytes, double scale) {
    // Allocate align to page.
    std::lock_guard<std::mutex> lock(_mutex);
    nbytes = (nbytes + (kWorkspacePageSize - 1)) / kWorkspacePageSize *
             kWorkspacePageSize;
    if (nbytes == 0) nbytes = kWorkspacePageSize;

    Entry e;
    if (_free_list.size() == 1) {
      nbytes *= scale;
      e.handle = _internal_allocator(nbytes);
      e.data = e.handle->ptr();
      e.size = nbytes;
    } else {
      if (_free_list.back().size >= nbytes) {
        // find smallest fit
        auto it = _free_list.end() - 2;
        for (; it->size >= nbytes; --it) {
        }
        e = *(it + 1);
        if (e.size > 2 * nbytes) {
          nbytes *= scale;
          e.handle = _internal_allocator(nbytes);
          e.data = e.handle->ptr();
          e.size = nbytes;
        } else {
          _free_list.erase(it + 1);
          _free_list_total_size -= e.size;
        }
      } else {
        nbytes *= scale;
        e.handle = _internal_allocator(nbytes);
        e.data = e.handle->ptr();
        e.size = nbytes;
      }
    }
    _allocated.push_back(e);
    _allocated_total_size += e.size;
    return e.data;
  }

  // free resource back to pool
  void Free(void *data) {
    std::lock_guard<std::mutex> lock(_mutex);
    Entry e;
    if (_allocated.back().data == data) {
      // quick path, last allocated.
      e = _allocated.back();
      _allocated.pop_back();
    } else {
      int index = static_cast<int>(_allocated.size()) - 2;
      for (; index > 0 && _allocated[index].data != data; --index) {
      }
      CHECK_GT(index, 0) << "trying to free things that has not been allocated";
      e = _allocated[index];
      _allocated.erase(_allocated.begin() + index);
    }
    _allocated_total_size -= e.size;

    if (_free_list.back().size < e.size) {
      _free_list.push_back(e);
    } else {
      size_t i = _free_list.size() - 1;
      _free_list.resize(_free_list.size() + 1);
      for (; e.size < _free_list[i].size; --i) {
        _free_list[i + 1] = _free_list[i];
      }
      _free_list[i + 1] = e;
    }
    _free_list_total_size += e.size;
  }

  // Release all resources
  void Release() {
    CHECK_EQ(_allocated.size(), 1);
    for (size_t i = 1; i < _free_list.size(); ++i) {
      _free_list[i].handle = nullptr;
    }
    _free_list.clear();
  }

 private:
  /*! \brief a single entry in the pool */
  struct Entry {
    void *data;
    MemHandle handle = nullptr;
    size_t size;
  };

  std::vector<Entry> _free_list;
  std::vector<Entry> _allocated;
  size_t _free_list_total_size = 0, _allocated_total_size = 0;
  std::mutex _mutex;

  constexpr static size_t kListSize = 100;

};

WorkspacePool::WorkspacePool(std::function<MemHandle(size_t)> internal_allocator){
  _internal_allocator = internal_allocator;
  _pool = new Pool;
  _pool->_internal_allocator = internal_allocator;
}

WorkspacePool::~WorkspacePool() {
  // for (size_t i = 0; i < _array.size(); ++i) {
  //   if (_array[i] != nullptr) {
  //     Context ctx;
  //     ctx.device_type = _device_type;
  //     ctx.device_id = static_cast<int>(i);
  //     _array[i]->Release(ctx, _device.get());
  //     delete _array[i];
  //   }
  // }
}

WorkspacePool::WorkspacePoolMemHandle::~WorkspacePoolMemHandle() {
  pool_->Free(this->data_);
}

MemHandle WorkspacePool::AllocWorkspace(size_t size, double scale) {
  std::shared_ptr<WorkspacePoolMemHandle> ret = std::make_shared<WorkspacePoolMemHandle>();
  ret->pool_ = _pool;
  ret->data_ = _pool->Alloc(size, scale);
  return ret;
}

// void WorkspacePool::FreeWorkspace(void *ptr) {
//   _array[ctx.device_id]->Free(ptr);
// }

}  // namespace common
}  // namespace samgraph

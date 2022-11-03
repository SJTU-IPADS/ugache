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

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <vector>

// #include "logging.h"
#include "constant.h"

namespace coll_cache_lib {
namespace common {

enum DataType {
  kF32 = 0,
  kF64 = 1,
  kF16 = 2,
  kU8 = 3,
  kI32 = 4,
  kI8 = 5,
  kI64 = 6,
};

enum DeviceType { kCPU = 0, kMMAP = 1, kGPU = 2, kGPU_UM = 3};


// cache by degree: cache the nodes with large degree
// cache by heuristic: cache the training set and the first hop neighbors first,
// then the nodes with large degree
enum CachePolicy {
  kCacheByDegree = 0,
  kCacheByHeuristic,
  kCacheByPreSample,
  kCacheByDegreeHop,
  kCacheByPreSampleStatic,
  kCacheByFakeOptimal,
  kDynamicCache,
  kCacheByRandom,
  kCollCache,
  kCollCacheIntuitive,
  kPartitionCache,
  kPartRepCache,
  kRepCache,
  kCollCacheAsymmLink,
  kCliquePart,
  kCliquePartByDegree,
};

enum RollingPolicy {
  AutoRolling = 0,
  EnableRolling,
  DisableRolling,
};



struct Context {
  DeviceType device_type;
  int device_id;


  Context() {}
  Context(DeviceType type, int id) : device_type(type), device_id(id) {}
  Context(std::string name);
  int GetCudaDeviceId() const {
    if (device_type == DeviceType::kCPU || device_type == DeviceType::kMMAP) {
      return -1;
    } else {
      return device_id;
    }
  }
  bool operator==(const Context& rhs) {
    return this->device_type == rhs.device_type &&
           this->device_id == rhs.device_id;
  }
  friend std::ostream& operator<<(std::ostream& os, const Context& ctx);
  friend Context Priority(Context c1, Context c2) {
    return (c1.device_type >= c2.device_type) ? c1 : c2;
  }
};

using StreamHandle = void*;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class ExternelGPUMemoryHandler {
 public:
  ExternelGPUMemoryHandler() {}
  virtual ~ExternelGPUMemoryHandler() {}
  virtual void* ptr() = 0;
  template<typename T> T* ptr() {return static_cast<T*>(ptr());}
};

using MemHandle = std::shared_ptr<ExternelGPUMemoryHandler>;

class ExternalBarrierHandler {
 public:
  ExternalBarrierHandler() {}
  virtual ~ExternalBarrierHandler() {}
  virtual void Wait() = 0;
};
using BarHandle = std::shared_ptr<ExternalBarrierHandler>;

size_t GetDataTypeBytes(DataType dtype);
class Tensor {
 public:
  Tensor();
  ~Tensor();

  inline std::string Name() const { return _name; }
  bool Defined() const { return _data; }
  DataType Type() const { return _dtype; }
  const std::vector<size_t>& Shape() const { return _shape; }
  const void* Data() const { return _data; }
  template<typename T> T* Ptr();
  template<typename T> const T* CPtr() const { return const_cast<Tensor*>(this)->Ptr<T>(); }
  void* MutableData() { return _data; }
  void ReplaceData(void* data);
  void Swap(TensorPtr tensor);
  size_t NumBytes() const { return _nbytes; }
  Context Ctx() const { return _ctx; }
  inline size_t NumItem() const { return std::accumulate(_shape.begin(), _shape.end(), 1ul, std::multiplies<size_t>()); }
  void ReShape(std::vector<size_t> new_shape);

  static TensorPtr Null();

  static TensorPtr CreateShm(std::string shm_path, DataType dtype,
                             std::vector<size_t> shape, std::string name);
  static TensorPtr OpenShm(std::string shm_path, DataType dtype,
                             std::vector<size_t> shape, std::string name);
  static TensorPtr Empty(DataType dtype, std::vector<size_t> shape, Context ctx,
                         std::string name);
  static TensorPtr EmptyNoScale(DataType dtype, std::vector<size_t> shape,
                                Context ctx, std::string name);
  // static TensorPtr Copy1D(TensorPtr tensor, size_t item_offset,
  //                         std::vector<size_t> shape, std::string name,
  //                         StreamHandle stream = nullptr);
  static TensorPtr FromMmap(std::string filepath, DataType dtype,
                            std::vector<size_t> shape, Context ctx,
                            std::string name, StreamHandle stream = nullptr);
  static TensorPtr FromBlob(void* data, DataType dtype,
                            std::vector<size_t> shape, Context ctx,
                            std::string name);
  // static TensorPtr CopyTo(TensorPtr source, Context ctx, StreamHandle stream = nullptr, double scale = Constant::kAllocScale);
  static TensorPtr CopyToExternal(TensorPtr source, const std::function<MemHandle(size_t)> & allocator, Context ctx, StreamHandle stream = nullptr, double scale = Constant::kAllocScale);
  static TensorPtr CopyLineToExternel(TensorPtr source, size_t line_idx, std::function<MemHandle(size_t)> & allocator, Context ctx, StreamHandle stream = nullptr, double scale = Constant::kAllocScale);
  // static TensorPtr CopyTo(TensorPtr source, Context ctx, StreamHandle stream, std::string name, double scale = Constant::kAllocScale);
  // static TensorPtr CopyLine(TensorPtr source, size_t line_idx, Context ctx, StreamHandle stream = nullptr, double scale = Constant::kAllocScale);
  // static TensorPtr UMCopyTo(TensorPtr source, std::vector<Context> ctxes, std::vector<StreamHandle> streams = {});
  // static TensorPtr UMCopyTo(TensorPtr source, std::vector<Context> ctxes, std::vector<StreamHandle> streams, std::string name);
  // static TensorPtr CopyBlob(const void * data, DataType dtype,
  //                           std::vector<size_t> shape, Context from_ctx,
  //                           Context to_ctx, std::string name, StreamHandle stream = nullptr);

 private:
  void* _data;
  DataType _dtype;
  Context _ctx;

  size_t _nbytes;
  std::vector<size_t> _shape;

  std::string _name;

  MemHandle _external_mem_hanlder;
};


constexpr static int CPU_CUDA_HOST_MALLOC_DEVICE = 0;
constexpr static int CPU_CLIB_MALLOC_DEVICE = 1;
constexpr static int CPU_FOREIGN = 2;

constexpr static int MMAP_RO_DEVICE = 0;
constexpr static int MMAP_RW_DEVICE = 1;
constexpr static int MMAP_FOREIGN = 2;

Context CPU(int device_id = CPU_CUDA_HOST_MALLOC_DEVICE);
Context CPU_CLIB(int device_id = CPU_CLIB_MALLOC_DEVICE);
Context GPU(int device_id = 0);
Context GPU_UM(int device_id = 0);
Context MMAP(int device_id = MMAP_RO_DEVICE);

DataType DataTypeParseName(std::string name);
size_t GetDataTypeBytes(DataType dtype);
size_t GetTensorBytes(DataType dtype, const std::vector<size_t> shape);
size_t GetTensorBytes(DataType dtype,
                      std::vector<size_t>::const_iterator shape_start,
                      std::vector<size_t>::const_iterator shape_end);

std::string ToReadableSize(size_t nbytes);
std::string ToPercentage(double percentage);

std::string GetEnv(std::string key);
bool IsEnvSet(std::string key);
std::string GetTimeString();
bool FileExist(const std::string& filepath);

template <typename T>
inline T RoundUpDiv(T target, T unit) {
  return (target + unit - 1) / unit;
}

template <typename T>
inline T RoundUp(T target, T unit) {
  return RoundUpDiv(target, unit) * unit;
}

template <typename T>
inline T Max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
inline T Min(T a, T b) {
  return a < b ? a : b;
}

template < typename T >
T GCD(T a, T b) {
  if(b) while((a %= b) && (b %= a));
  return a + b;
}
template < typename T >
T LCM(T a, T b) {
  return a * b / GCD(a, b);
}

std::ostream& operator<<(std::ostream&, const CachePolicy);

}  // namespace common
}  // namespace coll_cache_lib
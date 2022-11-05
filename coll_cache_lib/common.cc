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

#include "common.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cctype>
#include <chrono>  // chrono::system_clock
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>    // localtime
#include <iomanip>  // put_time
#include <numeric>
#include <sstream>  // stringstream
#include <string>   // string

#include "atomic_barrier.h"
#include "constant.h"
#include "run_config.h"
#include "device.h"
#include "cpu/mmap_cpu_device.h"
#include "logging.h"
#include "run_config.h"

namespace coll_cache_lib {
namespace common {

Context::Context(std::string name) {
  size_t delim_pos = name.find(':');
  CHECK_NE(delim_pos, std::string::npos);

  std::string device_str = name.substr(0, delim_pos);
  std::string id_str = name.substr(delim_pos + 1, std::string::npos);
  CHECK(device_str == "cpu" || device_str == "cuda" || device_str == "mmap");

  if (device_str == "cpu") {
    device_type = kCPU;
  } else if (device_str == "cuda") {
    device_type = kGPU;
  } else if (device_str == "mmap") {
    device_type = kMMAP;
  } else {
    CHECK(false);
  }

  device_id = std::stoi(id_str);
}

Tensor::Tensor() : _data(nullptr) {}

Tensor::~Tensor() {
  if (!_data) {
    return;
  }

  if (_external_mem_hanlder != nullptr) {
    _external_mem_hanlder = nullptr;
    LOG(DEBUG) << "Tensor " << _name << " has been freed";
    return;
  }

  Device::Get(_ctx)->FreeWorkspace(_ctx, _data, _nbytes);
  LOG(DEBUG) << "Tensor " << _name << " has been freed";
}

void Tensor::ReplaceData(void *data) {
  Device::Get(_ctx)->FreeWorkspace(_ctx, _data);
  _data = data;
}

void Tensor::Swap(TensorPtr tensor) {
  CHECK(this->Ctx() == tensor->Ctx());
  CHECK(this->Shape() == tensor->Shape());
  CHECK(this->Type() == tensor->Type());
  std::swap(this->_data, tensor->_data);
}

void Tensor::ReShape(std::vector<size_t> new_shape) {
  CHECK(Defined());
  CHECK(GetTensorBytes(kI8, _shape) == GetTensorBytes(kI8, new_shape));
  _shape = new_shape;
}

TensorPtr Tensor::Null() { return std::make_shared<Tensor>(); }

TensorPtr Tensor::CreateShm(std::string shm_path, DataType dtype,
                            std::vector<size_t> shape, std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());
  int fd = cpu::MmapCPUDevice::CreateShm(nbytes, shm_path);
  void* data = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, fd);

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = data;
  tensor->_ctx = MMAP(MMAP_RW_DEVICE);
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::OpenShm(std::string shm_path, DataType dtype,
                          std::vector<size_t> shape, std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());
  int fd = cpu::MmapCPUDevice::OpenShm(shm_path);

  struct stat st;
  fstat(fd, &st);
  size_t file_nbytes = st.st_size;

  if (shape.size() == 0) {
    // auto infer shape, 1-D only
    shape = {file_nbytes / GetDataTypeBytes(dtype)};
    nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());
  }

  CHECK_EQ(nbytes, file_nbytes);

  void* data = cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, fd);

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = data;
  tensor->_ctx = MMAP(MMAP_RW_DEVICE);
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::FromMmap(std::string filepath, DataType dtype,
                           std::vector<size_t> shape, Context ctx,
                           std::string name, StreamHandle stream) {
  CHECK(FileExist(filepath)) << "No file " << filepath;

  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  struct stat st;
  stat(filepath.c_str(), &st);
  size_t file_nbytes = st.st_size;
  CHECK_EQ(nbytes, file_nbytes);

  // alloc memory
  int fd = open(filepath.c_str(), O_RDONLY, 0);
  int device_id = ctx.device_id;
  ctx.device_id = MMAP_RW_DEVICE;
  void *data = Device::Get(ctx)->AllocDataSpace(ctx, nbytes, nbytes);
  ctx.device_id = device_id;
  CHECK_NE(data, (void *)-1);
  
  // read huge file
  size_t read_bytes = 0;
  while (read_bytes < nbytes)
  {
    ssize_t res = read(fd, ((uint8_t*)data) + read_bytes, nbytes - read_bytes);
    CHECK_GT(res, 0);
    read_bytes += res;
  }
  CHECK_EQ(read_bytes, nbytes) << "should read " << nbytes << ", actually read " << read_bytes;
  CHECK_EQ(mprotect(data, nbytes, PROT_READ), 0);
  close(fd);

  tensor->_dtype = dtype;
  tensor->_nbytes = nbytes;
  tensor->_shape = shape;
  tensor->_ctx = ctx;
  tensor->_name = name;

  // if the device is cuda, we have to copy the data from host memory to cuda
  // memory
  switch (ctx.device_type) {
    case kCPU:
    case kGPU:
      tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes,
                                                       Constant::kAllocNoScale);
      Device::Get(ctx)->CopyDataFromTo(data, 0, tensor->_data, 0, nbytes, CPU(),
                                       ctx, stream);
      Device::Get(ctx)->StreamSync(ctx, stream);
      Device::Get(MMAP())->FreeWorkspace(MMAP(), data, nbytes);
      break;
    case kGPU_UM: {
      LOG(FATAL) << "GPU_UM device should use `UMFromMmap`";
    }
      break;
    case kMMAP:
      tensor->_data = data;
      break;
    default:
      CHECK(0);
  }

  return tensor;
}

TensorPtr Tensor::Empty(DataType dtype, std::vector<size_t> shape, Context ctx,
                        std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  CHECK_GT(shape.size(), 0);
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = Device::Get(ctx)->AllocWorkspace(ctx, nbytes);
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}
TensorPtr Tensor::EmptyNoScale(DataType dtype, std::vector<size_t> shape,
                               Context ctx, std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  CHECK_GT(shape.size(), 0);
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = Device::Get(ctx)->
    AllocWorkspace(ctx, nbytes, Constant::kAllocNoScale);
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}


Context CPU(int device_id) { return {kCPU, device_id}; }
Context CPU_CLIB(int device_id) { return {kCPU, device_id}; }
Context GPU(int device_id) { return {kGPU, device_id}; }
Context GPU_UM(int device_id) { return {kGPU_UM, device_id}; }
Context MMAP(int device_id) { return {kMMAP, device_id}; }

TensorPtr Tensor::FromBlob(void *data, DataType dtype,
                           std::vector<size_t> shape, Context ctx,
                           std::string name) {
  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(dtype, shape.begin(), shape.end());

  tensor->_dtype = dtype;
  tensor->_shape = shape;
  tensor->_nbytes = nbytes;
  tensor->_data = data;
  tensor->_ctx = ctx;
  tensor->_name = name;

  return tensor;
}

TensorPtr Tensor::CopyToExternal(TensorPtr source, const std::function<MemHandle(size_t)> & allocator, Context ctx, StreamHandle stream, double scale) {
  CHECK(source && source->Defined());
  std::vector<size_t> shape = source->Shape();
  CHECK_GT(shape.size(), 0);

  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(source->_dtype, shape.begin(), shape.end());

  tensor->_dtype = source->_dtype;
  tensor->_shape = shape;
  tensor->_nbytes = source->_nbytes;
  tensor->_ctx = ctx;
  tensor->_external_mem_hanlder = allocator(nbytes);
  tensor->_data = tensor->_external_mem_hanlder->ptr();
  tensor->_name = source->_name;
  Context working_ctx = Priority(source->Ctx(), ctx);
  Device::Get(working_ctx)->CopyDataFromTo(source->_data, 0, tensor->_data, 0,
                                                nbytes, source->_ctx, tensor->_ctx, stream);
  Device::Get(working_ctx)->StreamSync(working_ctx, stream);
  return tensor;
}
TensorPtr Tensor::CopyLineToExternel(TensorPtr source, size_t line_idx, std::function<MemHandle(size_t)> & allocator, Context ctx, StreamHandle stream, double scale) {
  CHECK(source && source->Defined());
  const std::vector<size_t> & shape = source->_shape;
  CHECK_GT(shape.size(), 0);
  CHECK_LT(line_idx, shape[0]);

  TensorPtr tensor = std::make_shared<Tensor>();
  size_t nbytes = GetTensorBytes(source->_dtype, shape.begin() + 1, shape.end());

  tensor->_dtype = source->_dtype;
  tensor->_shape = std::vector<size_t>(shape.begin() + 1, shape.end());
  tensor->_nbytes = nbytes;
  tensor->_ctx = ctx;
  tensor->_external_mem_hanlder = allocator(nbytes);
  tensor->_data = tensor->_external_mem_hanlder->ptr();
  tensor->_name = source->_name;
  Context working_ctx = Priority(source->Ctx(), ctx);
  Device::Get(working_ctx)->CopyDataFromTo(source->_data, nbytes * line_idx, tensor->_data, 0, nbytes, source->_ctx, tensor->_ctx, stream);
  Device::Get(working_ctx)->StreamSync(working_ctx, stream);

  return tensor;
}

template<typename T> 
T* Tensor::Ptr(){ 
  CHECK(_data == nullptr || (sizeof(T) == GetDataTypeBytes(_dtype))); 
  return static_cast<T*>(_data);
}
template float* Tensor::Ptr<float>();
template double* Tensor::Ptr<double>();
template size_t* Tensor::Ptr<size_t>();
template IdType* Tensor::Ptr<IdType>();
template Id64Type* Tensor::Ptr<Id64Type>();
template char* Tensor::Ptr<char>();
template uint8_t* Tensor::Ptr<uint8_t>();

std::ostream& operator<<(std::ostream& os, const Context& ctx) {
  switch (ctx.device_type)
  {
  case DeviceType::kMMAP:
    os << "mmap:" << ctx.device_id;
    return os;
  case DeviceType::kCPU:
    os << "cpu:" << ctx.device_id;    
    return os;
  case DeviceType::kGPU:
    os << "gpu:" << ctx.device_id;
    return os;
  case DeviceType::kGPU_UM:
    os << "gpu_um:" << ctx.device_id;
    return os;
  default:
    LOG(FATAL) << "not support device type "
               << static_cast<int>(ctx.device_type) << ":" << ctx.device_id;
    // os << "not supprt:" << static_cast<int>(ctx.device_type) << ":" << ctx.device_id;
    return os;
  }
}

std::string ToReadableSize(size_t nbytes) {
  char buf[Constant::kBufferSize];
  if (nbytes > Constant::kGigabytes) {
    double new_size = (float)nbytes / Constant::kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kMegabytes) {
    double new_size = (float)nbytes / Constant::kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kKilobytes) {
    double new_size = (float)nbytes / Constant::kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}

std::string ToPercentage(double percentage) {
  char buf[Constant::kBufferSize];
  sprintf(buf, "%.2lf %%", percentage * 100);
  return std::string(buf);
}

DataType DataTypeParseName(std::string name) {
  static std::unordered_map<std::string, DataType> _map = {
    {"F32", kF32},
    {"F64", kF64},
    {"F16", kF16},
    {"U8",  kU8},
    {"I32", kI32},
    {"I8",  kI8},
    {"I64", kI64},
  };
  if (_map.find(name) == _map.end()) {
    CHECK(false) << "Unrecognized data type name: " << name;
  }
  return _map[name];
}

size_t GetDataTypeBytes(DataType dtype) {
  switch (dtype) {
    case kI8:
    case kU8:
      return 1ul;
    case kF16:
      return 2ul;
    case kF32:
    case kI32:
      return 4ul;
    case kI64:
    case kF64:
      return 8ul;
    default:
      CHECK(0) << "Unsupported data type: " << dtype;
  }
  return 4ul;
}

size_t GetTensorBytes(DataType dtype, const std::vector<size_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1ul,
                         std::multiplies<size_t>()) *
         GetDataTypeBytes(dtype);
}

size_t GetTensorBytes(DataType dtype,
                      std::vector<size_t>::const_iterator shape_start,
                      std::vector<size_t>::const_iterator shape_end) {
  return std::accumulate(shape_start, shape_end, 1ul,
                         std::multiplies<size_t>()) *
         GetDataTypeBytes(dtype);
}

std::string GetEnv(std::string key) {
  const char *env_var_val = getenv(key.c_str());
  if (env_var_val != nullptr) {
    return std::string(env_var_val);
  } else {
    return "";
  }
}

std::string GetEnvStrong(std::string key) {
  const char *env_var_val = getenv(key.c_str());
  CHECK(env_var_val != nullptr) << "Env " << key << " is required before loading coll cache lib";
  return std::string(env_var_val);
}

bool IsEnvSet(std::string key) {
  std::string val = GetEnv(key);
  if (val == "ON" || val == "1") {
    LOG(INFO) << "Environment variable " << key << " is set to " << val;
    return true;
  } else {
    return false;
  }
}

std::string GetTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%dT%H%M%S%");
  return ss.str();
}

bool FileExist(const std::string &filepath) {
  std::ifstream f(filepath);
  return f.good();
}

std::ostream& operator<<(std::ostream& os, const CachePolicy policy) {
  std::string str;
  switch (policy) {
    case kCacheByDegree:
      os << "degree";
      break;
    case kCacheByHeuristic:
      os << "heuristic";
      break;
    case kCacheByPreSample:
      os << "preSample";
      break;
    case kCacheByPreSampleStatic:
      os << "preSampleStatic";
      break;
    case kCacheByDegreeHop:
      os << "degree_hop";
      break;
    case kCacheByFakeOptimal:
      os << "fake_optimal";
      break;
    case kCacheByRandom:
      os << "random";
      break;
    case kCollCache:
      os << "coll_cache";
      break;
    case kCollCacheIntuitive:
      os << "coll_cache_naive";
      break;
    case kCollCacheAsymmLink:
      os << "coll_cache_asymm_link";
      break;
    case kCliquePart:
      os << "clique_part";
      break;
    case kCliquePartByDegree:
      os << "clique_part_by_degree";
      break;
    case kPartitionCache:
      os << "partition_cache";
      break;
    case kPartRepCache:
      os << "part_rep_cache";
      break;
    case kRepCache:
      os << "rep_cache";
      break;
    default:
      CHECK(false);
  }

  return os;
}

AnonymousBarrier::AnonymousBarrier(int worker) {
  auto mmap_ctx = MMAP(MMAP_RW_DEVICE);
  auto dev = Device::Get(mmap_ctx);
  this->_barrier_buffer = dev->AllocDataSpace(mmap_ctx, sizeof(AtomicBarrier));
  new (this->_barrier_buffer) AtomicBarrier(worker);
}
void AnonymousBarrier::Wait() { (reinterpret_cast<AtomicBarrier*>(this->_barrier_buffer))->Wait(); }
std::shared_ptr<AnonymousBarrier> AnonymousBarrier::_global_instance = std::make_shared<AnonymousBarrier>(std::stoi(GetEnvStrong("COLL_NUM_REPLICA")));
EagerGPUMemoryHandler::EagerGPUMemoryHandler() {}
EagerGPUMemoryHandler::~EagerGPUMemoryHandler() {
  CUDA_CALL(cudaSetDevice(dev_id_));
  CUDA_CALL(cudaFree(ptr_));
}
}  // namespace common
}  // namespace coll_cache_lib

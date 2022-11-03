#pragma once
#include "common.h"
// #include "cpu/cpu_device.h"
// #include "cpu/mmap_cpu_device.h"
#include "run_config.h"
// #include "logging.h"
// #include "coll_cache/ndarray.h"
// #include "coll_cache/optimal_solver_class.h"
// #include "facade.h"
// #include "timer.h"
// #include "atomic_barrier.h"
#include <cuda_runtime.h>


#define SAM_CUDA_PREPARE_1D(num_item) \
  const size_t num_tiles = RoundUpDiv((num_item), Constant::kCudaTileSize); \
  const dim3 grid(num_tiles); \
  const dim3 block(Constant::kCudaBlockSize);

#define SWITCH_TYPE(type, Type, ...)      \
  switch(type) {                     \
    case kF32: { typedef float   Type; { __VA_ARGS__ }; break; } \
    case kF64: { typedef double  Type; { __VA_ARGS__ }; break; } \
    case kF16: { typedef short   Type; { __VA_ARGS__ }; break; } \
    case kU8:  { typedef uint8_t Type; { __VA_ARGS__ }; break; } \
    case kI32: { typedef int32_t Type; { __VA_ARGS__ }; break; } \
    case kI64: { typedef int64_t Type; { __VA_ARGS__ }; break; } \
    default: CHECK(false);           \
  }

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef cuda::binary::Add<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef cuda::binary::Sub<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef cuda::binary::Mul<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef cuda::binary::Div<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_lhs") {                                \
      typedef cuda::binary::CopyLhs<DType> Op;                      \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_rhs") {                                \
      typedef cuda::binary::CopyRhs<DType> Op;                      \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

namespace coll_cache_lib {

using namespace common;
// per-gpu cache handler


using HashTableEntryLocation = int;
using HashTableEntryOffset = IdType;

struct SrcKey {
  int _location_id;
};
struct DstVal {
  IdType _src_offset;
  IdType _dst_offset;
};

struct DevicePointerExchanger {
  void* _buffer;
  // bool _cross_process = false;
  // no way to ensure barrier is globally initialized then used, so let application pass-in a barrier
  BarHandle _barrier;
  DevicePointerExchanger(BarHandle barrier,
                         std::string shm_name);
  void signin(int local_id, void* ptr_to_share);
  void* extract(int location_id);
};
class CollCache;
class CacheContext {
 private:
  BarHandle _barrier;
  using HashTableEntryLocation = int;
  using HashTableEntryOffset = IdType;

  std::shared_ptr<CollCache> _coll_cache;

  Context _trainer_ctx;

  DataType _dtype;
  size_t _dim;

  int _num_location = -1;
  int _cpu_location_id = -1;
  int _local_location_id = -1;

  size_t _cache_nbytes = 0;

  // HashTableEntry* _hash_table = nullptr;
  HashTableEntryLocation* _hash_table_location = nullptr;
  HashTableEntryOffset* _hash_table_offset = nullptr;
  MemHandle _hash_table_location_handle;
  MemHandle _hash_table_offset_handle;
  std::vector<void*> _device_cache_data;
  MemHandle _device_cache_data_local_handle;

  // std::vector<int> _remote_device_list;
  // std::vector<int> _remote_sm_list;
  std::vector<StreamHandle> _concurrent_stream_array;

  std::function<MemHandle(size_t)> _gpu_mem_allocator;
  friend class ExtractSession;

  void build_without_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
  void build_with_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
 public:
  CacheContext(BarHandle barrier) : _barrier(barrier) {}
  void build(std::function<MemHandle(size_t)> gpu_mem_allocator,
             int location_id, std::shared_ptr<CollCache> coll_cache_ptr,
             void *cpu_data, DataType dtype, size_t dim, Context gpu_ctx,
             double cache_percentage, StreamHandle stream = nullptr);
  void lookup();

 private:
};

class ExtractSession {
  std::shared_ptr<CacheContext> _cache_ctx;
  MemHandle output_src_index_handle, output_dst_index_handle;
  IdType * _group_offset = nullptr;
 public:
  ExtractSession(std::shared_ptr<CacheContext> cache_ctx);
 private:

  void SplitGroup(const SrcKey * src_index, const size_t len, IdType * & group_offset, StreamHandle stream);

  void GetMissCacheIndex(
    SrcKey* & output_src_index, DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream);

  void CombineOneGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block = 0, bool async = false);

  template<int NUM_LINK>
  void CombineConcurrent(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream);

 public:
  void ExtractFeat(const IdType* nodes, const size_t num_nodes, void* output, StreamHandle stream, uint64_t task_key);

};
}
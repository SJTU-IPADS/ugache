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
#include "cuda/cache_hashtable.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cuda_runtime.h>
#include <cuda.h>


#define SAM_CUDA_PREPARE_1D(num_item) \
  const size_t num_tiles = RoundUpDiv((num_item), Constant::kCudaTileSize); \
  const dim3 grid(num_tiles); \
  const dim3 block(Constant::kCudaBlockSize);


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
  void close(void* ptr);
};

struct HostPointerExchanger {
  void* _buffer;
  // bool _cross_process = false;
  // no way to ensure barrier is globally initialized then used, so let application pass-in a barrier
  std::string _shm_name;
  BarHandle _barrier;
  HostPointerExchanger(BarHandle barrier, std::string shm_name);
  void* signin(int local_id, size_t nbytes);
  void* extract(int location_id);
  void close(void* ptr);
};

struct ExtractionThreadCtx {
  CUcontext cu_ctx_ = nullptr;
  cudaStream_t stream_;
  std::function<void(cudaStream_t)> func_;
  std::atomic<int> todo_steps{0}, done_steps{0};
  ExtractionThreadCtx();
  void thread_func();
  void forward_one_step(std::function<void(cudaStream_t)> new_func);
  void wait_one_step();
};

class CollCache;
class CacheContext {
 private:
  std::atomic<size_t> progress{0};
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
  size_t _cache_nodes = 0;
  size_t _cache_space_capacity = 0;

  // HashTableEntry* _hash_table = nullptr;
  HashTableEntryLocation* _hash_table_location = nullptr;
  HashTableEntryOffset* _hash_table_offset = nullptr;
  MemHandle _hash_table_location_handle;
  MemHandle _hash_table_offset_handle;
  std::vector<void*> _device_cache_data;
  MemHandle _device_cache_data_local_handle;
  std::vector<BucketO2N*> _remote_hash_table;
  CacheEntryManager* _new_hash_table;
  std::vector<HashTableEntryLocation*> _remote_hash_table_location;
  std::vector<HashTableEntryOffset*> _remote_hash_table_offset;
  std::vector<CacheEntryManager*> _remote_new_hash_table;
  size_t * d_num_selected_out = nullptr;

  // MemHandle _local_node_list_handle;
  TensorPtr _local_node_list_tensor;

  // std::vector<int> _remote_device_list;
  // std::vector<int> _remote_sm_list;

  std::function<MemHandle(size_t)> _gpu_mem_allocator;
  std::function<MemHandle(size_t)> _eager_gpu_mem_allocator;
  // std::function<void()> ctx_injector_;
  friend class ExtractSession;
  friend class RefreshSession;
  inline bool IsDirectMapping() {
    return _hash_table_location == nullptr;
    // if (_hash_table_location == nullptr) {
    //   CHECK(_num_location == 1);
    //   return true;
    // }
    // return false;
  }

  void build_no_cache(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, StreamHandle stream = nullptr);
  void build_full_cache(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, size_t num_total_nodes, StreamHandle stream = nullptr);
  void build_without_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
  void build_with_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
  void build_with_advise_new_hash(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
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
  MemHandle output_src_index_alter_handle, output_dst_index_alter_handle;
  MemHandle output_sorted_nodes_handle;
  MemHandle workspace_handle;
  IdType * _group_offset = nullptr;
  std::vector<StreamHandle> _concurrent_stream_array;
  std::vector<std::shared_ptr<ExtractionThreadCtx>> _extract_ctx;
  std::vector<std::thread> _extract_threads;
 public:
  ExtractSession(std::shared_ptr<CacheContext> cache_ctx);
 private:

  void SplitGroup(const SrcKey * src_index, const size_t len, IdType * & group_offset, StreamHandle stream);

  void GetMissCacheIndexByCub(DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes,
    IdType * & group_offset,
    StreamHandle stream);

  void GetMissCacheIndex(
    SrcKey* & output_src_index, DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream);

  void SortByLocation(
    IdType* &sorted_nodes,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream);

  void CombineOneGroup(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block = 0, bool async = false);
  void CombineOneGroupRevised(const SrcKey * src_index, const DstVal * dst_index, const IdType* nodes, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block = 0, bool async = false);

  void CombineNoGroup(const IdType* nodes, const size_t num_nodes, void* output, Context ctx, DataType _dtype, IdType _dim, StreamHandle stream);
  void CombineMixGroup(const SrcKey* src_key, const DstVal* dst_val, const size_t num_nodes, void* output, Context ctx, DataType _dtype, IdType _dim, StreamHandle stream);
  template<int NUM_LINK>
  void CombineConcurrent(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream);
  template<int NUM_LINK>
  void CombineFused(const SrcKey * src_index, const DstVal * dst_index, const IdType * group_offset, void* output, StreamHandle stream);

 public:
  void ExtractFeat(const IdType* nodes, const size_t num_nodes, void* output, StreamHandle stream, uint64_t task_key);

};

class RefreshSession {
 public:
  StreamHandle stream;
  std::shared_ptr<CacheContext> _cache_ctx;
  void refresh_after_solve();
};

}
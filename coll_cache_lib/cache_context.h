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
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "atomic_barrier.h"
#include <semaphore.h>


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

struct SrcKeyDstVal {
  IdType _dst_offset__;
  EmbCacheOff _src;
  __host__ __device__ __forceinline__ IdType src_loc() const { return _src.loc(); }
  __host__ __device__ __forceinline__ IdType src_off() const { return _src.off(); }
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

class JobSyncerWorkerHandle {
 public:
  virtual void on_wait_next_job() = 0;
  virtual void on_job_done() = 0;
  virtual void on_send_job() = 0;
};
class JobSyncer{
 public:
  virtual JobSyncerWorkerHandle* get_worker_handle(int idx = 0) = 0;
  virtual JobSyncer* on_wait_job_done() = 0;
  virtual JobSyncer* on_send_job() = 0;
};

class SpinJobSync:public JobSyncer{
  std::atomic<int> todo_steps{0}, done_steps{0};
 public:
  class WorkerHandle : public JobSyncerWorkerHandle {
   public:
    WorkerHandle(SpinJobSync* base) : syncer(base) {}
    SpinJobSync* syncer;
    void on_wait_next_job() override {
      int local_done_steps = syncer->done_steps.load();
      while (syncer->todo_steps.load() == local_done_steps) {}
    };
    void on_job_done() override { syncer->done_steps.fetch_add(1); };
    void on_send_job() override { syncer->todo_steps.fetch_add(1); }
  };
  JobSyncerWorkerHandle* get_worker_handle(int idx = 0) override {
    assert(idx == 0);
    return new WorkerHandle(this);
  };
  JobSyncer* on_wait_job_done() override {
    int local_todo_steps = todo_steps.load();
    while(local_todo_steps > done_steps.load()) {}
    return this;
  };
  JobSyncer* on_send_job() override {
    // todo_steps.fetch_add(1);
    return this;
  };
};
class ParallelJobSync:public JobSyncer{
  std::vector<JobSyncer*> sync_array;
 public:
  template<typename SingleSyncer>
  static ParallelJobSync* create(int num_syncer) {
    auto ret = new ParallelJobSync;
    ret->sync_array.resize(num_syncer);
    for (auto & syncer : ret->sync_array) {
      syncer = new SingleSyncer;
    }
    return ret;
  }
  JobSyncer* on_wait_job_done() override {
    for (JobSyncer* syncer : sync_array) {
      syncer->on_wait_job_done();
    }
    return this;
  };
  JobSyncer* on_send_job() override {
    for (JobSyncer* syncer : sync_array) {
      syncer->on_send_job();
    }
    return this;
  };
  JobSyncerWorkerHandle* get_worker_handle(int idx) override {
    return sync_array[idx]->get_worker_handle(0);
  };
};

class BarJobSync : public JobSyncer{
 public:
  class WorkerHandle : public JobSyncerWorkerHandle {
   public:
    WorkerHandle(AtomicBarrier* bar) : barrier_(bar) {}
    AtomicBarrier * barrier_;
    // call by thread
    void on_wait_next_job() override {barrier_->Wait();};
    void on_job_done() override {barrier_->Wait();};
    // call by sender
    void on_send_job() override {};
  };
  AtomicBarrier * barrier_;
  BarJobSync(int num_worker) : barrier_(new AtomicBarrier(num_worker + 1)) {}
  JobSyncer* on_wait_job_done() override {barrier_->Wait(); return this; }
  JobSyncer* on_send_job() override {barrier_->Wait(); return this; }
  JobSyncerWorkerHandle* get_worker_handle(int idx = 0) override {
    return new WorkerHandle(barrier_);
  };
};

class SemJobSync : public JobSyncer{
 public:
  class WorkerHandle : public JobSyncerWorkerHandle {
    SemJobSync* base;
   public:
    WorkerHandle(SemJobSync* base) : base(base) {}
    // call by thread
    void on_wait_next_job() override {sem_wait(&base->launch);};
    void on_job_done() override {sem_post(&base->done);}
    // call by sender
    void on_send_job() override {
      sem_post(&base->launch);
    }
  };
  sem_t launch;
  sem_t done;
  SemJobSync() {
    sem_init(&launch, 0, 0);
    sem_init(&done, 0, 0);
  }
  JobSyncer* on_wait_job_done() override {
    sem_wait(&done);
    return this;
  }
  JobSyncer* on_send_job() override {
    // sem_post(&launch);
    return this;
  }
  JobSyncerWorkerHandle* get_worker_handle(int idx = 0) override {
    assert(idx == 0);
    return new WorkerHandle(this);
  };
};
class SemBarJobSync : public JobSyncer{
 public:
  class WorkerHandle : public JobSyncerWorkerHandle {
    SemBarJobSync* base;
   public:
    WorkerHandle(SemBarJobSync* base) : base(base) {}
    // call by thread
    void on_wait_next_job() override { sem_wait(&base->launch); }
    void on_job_done() override { base->bar->Wait(); }
    // call by sender
    void on_send_job() override {
      // sem_post(&base->launch);
    }
  };
  sem_t launch;
  AtomicBarrier * bar;
  int num_worker;
  SemBarJobSync(int num_worker) : num_worker(num_worker) {
    sem_init(&launch, 0, 0);
    bar = new AtomicBarrier(num_worker + 1);
  }
  JobSyncer* on_wait_job_done() override {
    bar->Wait();
    return this;
  }
  JobSyncer* on_send_job() override {
    for (int i = 0; i < num_worker; i++) {
      sem_post(&launch);
    }
    return this;
  }
  JobSyncerWorkerHandle* get_worker_handle(int idx = 0) override {
    return new WorkerHandle(this);
  };
};

struct ExtractionThreadCtx {
  CUcontext cu_ctx_ = nullptr;
  cudaStream_t stream_;
  int id = 0;
  int local_id = 0;
  std::function<void(cudaStream_t)> func_;
  ExtractionThreadCtx();
  JobSyncerWorkerHandle* syncer_;
  void v2_thread_func();
  void v2_forward_one_step(std::function<void(cudaStream_t)> new_func);
  void v2_forward_nop();
  std::thread v2_launch();
  void create_ctx(int dev_id, int num_sm, int priority = 0);
  void create_stream(int dev_id, int priority = 0);
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
#ifdef COLL_HASH_VALID_LEGACY
  HashTableEntryLocation* _hash_table_location = nullptr;
  HashTableEntryOffset* _hash_table_offset = nullptr;
  MemHandle _hash_table_location_handle;
  MemHandle _hash_table_offset_handle;
#endif
  std::vector<void*> _device_cache_data;
  std::vector<void*> _device_cache_data_clique;
  MemHandle _device_cache_data_local_handle;
  std::vector<BucketO2N*> _remote_hash_table;
  CacheEntryManager* _new_hash_table = nullptr;
#ifdef COLL_HASH_VALID_LEGACY
  std::vector<HashTableEntryLocation*> _remote_hash_table_location;
  std::vector<HashTableEntryOffset*> _remote_hash_table_offset;
#endif
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
    // return _hash_table_location == nullptr;
    return _new_hash_table == nullptr;
    // if (_hash_table_location == nullptr) {
    //   CHECK(_num_location == 1);
    //   return true;
    // }
    // return false;
  }
#ifdef COLL_HASH_VALID_LEGACY
  void compare_hashtable(StreamHandle stream);
#endif

  void build_no_cache(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, StreamHandle stream = nullptr);
  void build_full_cache(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, size_t num_total_nodes, StreamHandle stream = nullptr);

#ifdef COLL_HASH_VALID_LEGACY
  void build_without_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
  void build_with_advise(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
#endif
  void build_with_advise_new_hash(int location_id, std::shared_ptr<CollCache> coll_cache_ptr, void* cpu_data, DataType dtype, size_t dim, Context gpu_ctx, double cache_percentage, StreamHandle stream = nullptr);
 public:
  CacheContext(BarHandle barrier) : _barrier(barrier) {}
  void build(std::function<MemHandle(size_t)> gpu_mem_allocator,
             int location_id, std::shared_ptr<CollCache> coll_cache_ptr,
             void *cpu_data, DataType dtype, size_t dim, Context gpu_ctx,
             double cache_percentage, StreamHandle stream = nullptr);
  // void lookup();

 private:
};

template<bool use_empty_feat=false>
struct IdxStoreDirect : IdxStoreAPI {
  const IdType* key_list_;
  size_t empty_feat_;
  IdxStoreDirect(const IdType* key_list) : key_list_(key_list), empty_feat_(RunConfig::option_empty_feat) {}
  __forceinline__ __host__ __device__ IdType src_off(const IdType idx) const {
    if (use_empty_feat) {
      return key_list_[idx] % (1 << empty_feat_);
    } else {
      return key_list_[idx];
    }
  }
  __forceinline__ __host__ __device__ IdType dst_off(const IdType idx) const {return idx; }
};

template<bool use_empty_feat=false>
struct IdxStoreDst : IdxStoreDirect<use_empty_feat> {
  const IdType* dst_off_;
  IdxStoreDst(const IdType* key_list, const IdType* dst_off) : IdxStoreDirect<use_empty_feat>(key_list), dst_off_(dst_off) {}
  __forceinline__ __host__ __device__ IdType dst_off(const IdType idx) const {return dst_off_[idx]; }
};

struct IdxStoreLegacy : IdxStoreAPI {
  SrcKey* src_key;
  DstVal* dst_val;
  IdxStoreLegacy() {}
  IdxStoreLegacy sub_array(size_t begin) const {
    IdxStoreLegacy ret;
    ret.src_key = this->src_key + begin;
    ret.dst_val = this->dst_val + begin;
    return ret;
  }
  __forceinline__ __host__ __device__ IdType src_loc(const IdType idx) const {return src_key[idx]._location_id; }
  __forceinline__ __host__ __device__ IdType src_off(const IdType idx) const {return dst_val[idx]._src_offset; }
  __forceinline__ __host__ __device__ IdType dst_off(const IdType idx) const {return dst_val[idx]._dst_offset; }
  // __forceinline__ __host__ __device__ void set_src_loc(const IdType idx, const IdType src_loc) {src_key[idx]._location_id = src_loc; }
  // __forceinline__ __host__ __device__ void set_src_off(const IdType idx, const IdType src_off) {dst_val[idx]._src_offset  = src_off; }
  __forceinline__ __host__ __device__ void set_src(const IdType idx, const IdType src_loc, const IdType src_off) { src_key[idx]._location_id = src_loc; dst_val[idx]._src_offset  = src_off; }
  __forceinline__ __host__ __device__ void set_dst_off(const IdType idx, const IdType dst_off) {dst_val[idx]._dst_offset  = dst_off; }
  __forceinline__ __host__ __device__ size_t empty_feat() { abort(); }
  __forceinline__ __host__ __device__ IdType fallback_loc() { abort(); }
  __forceinline__ __host__ __device__ size_t required_mem(const IdType num_keys) const {
    return (sizeof(SrcKey) + sizeof(DstVal)) * num_keys;
  }
  __forceinline__ __host__ __device__ void prepare_mem(uint8_t* buf, const IdType num_keys) {
    dst_val = (DstVal*)(buf);
    src_key = (SrcKey*)(dst_val + num_keys);
  }
  __forceinline__ __host__ int*      & keys_for_sort() { return (int* &)src_key; }
  __forceinline__ __host__ uint64_t* & vals_for_sort() { return (uint64_t* &)dst_val; }
};

struct IdxStoreCompact : IdxStoreAPI {
  SrcKeyDstVal* src_dst_idx;
  IdxStoreCompact() {}
  // IdxStoreCompact(int fallback_loc) : fallback_loc(fallback_loc), empty_feat(RunConfig::option_empty_feat) {}
  IdxStoreCompact sub_array(size_t begin) const {
    IdxStoreCompact ret;
    ret.src_dst_idx = this->src_dst_idx + begin;
    return ret;
  }
  __forceinline__ __host__ __device__ IdType src_loc(const IdType idx) const {return src_dst_idx[idx].src_loc(); }
  __forceinline__ __host__ __device__ IdType src_off(const IdType idx) const {return src_dst_idx[idx].src_off(); }
  __forceinline__ __host__ __device__ IdType dst_off(const IdType idx) const {return src_dst_idx[idx]._dst_offset__; }
  // __forceinline__ __host__ __device__ void set_src_loc(const IdType idx, const IdType src_loc) { src_dst_idx[idx]._src.set_loc(src_loc);}
  // __forceinline__ __host__ __device__ void set_src_off(const IdType idx, const IdType src_off) { src_dst_idx[idx]._src.set_off(src_off); }
  __forceinline__ __host__ __device__ void set_src(const IdType idx, const IdType src_loc, const IdType src_off) { src_dst_idx[idx]._src = EmbCacheOff(src_loc, src_off); }
  __forceinline__ __host__ __device__ void set_dst_off(const IdType idx, const IdType dst_off) { src_dst_idx[idx]._dst_offset__  = dst_off; }
  __forceinline__ __host__ __device__ size_t required_mem(const IdType num_keys) const {
    return (sizeof(SrcKeyDstVal)) * num_keys;
  }
  __forceinline__ __host__ __device__ void prepare_mem(uint8_t* buf, const IdType num_keys) {
    src_dst_idx = (SrcKeyDstVal*)buf;
  }
  __forceinline__ __host__ uint64_t* & keys_for_sort() { return (uint64_t* &)src_dst_idx; }
  __forceinline__ __host__ void*  vals_for_sort() { return nullptr; }
};
using IdxStore = IdxStoreLegacy;

class ExtractSession {
  std::shared_ptr<CacheContext> _cache_ctx;
  MemHandle idx_store_handle = nullptr, idx_store_alter_handle = nullptr;
  MemHandle output_src_index_handle, output_dst_index_handle;
  MemHandle output_src_index_alter_handle, output_dst_index_alter_handle;
  MemHandle output_sorted_nodes_handle;
  MemHandle workspace_handle;
  IdType * _group_offset = nullptr;
  std::vector<StreamHandle> _concurrent_stream_array;
  std::vector<std::shared_ptr<ExtractionThreadCtx>> _extract_ctx;
  ExtractionThreadCtx* launch_thread(int src_location_id, JobSyncerWorkerHandle* syncer, std::function<void(ExtractionThreadCtx* ctx)> init_func) {
    auto ext_ctx = std::make_shared<ExtractionThreadCtx>();
    ext_ctx->id = src_location_id;
    ext_ctx->local_id = _local_location_id;
    ext_ctx->syncer_ = syncer;
    this->_extract_ctx[src_location_id] = ext_ctx;
    this->_extract_threads[src_location_id] = ext_ctx->v2_launch();
    ext_ctx->v2_forward_one_step([init_func, ext_ctx](cudaStream_t){
      init_func(ext_ctx.get());
    });
    return ext_ctx.get();
  }
  JobSyncer *_local_syncer = nullptr, *_remote_syncer = nullptr, *_cpu_syncer = nullptr;
  cudaStream_t _local_ext_stream;
  std::vector<std::thread> _extract_threads;
  double accu_cpu_time = 0;
  double accu_local_time = 0;
  double accu_remote_time = 0;
  size_t accu_step = 0;
  double accu_each_src_time[9] = {};
  double accu_each_src_nkey[9] = {};
  double phase_time_record[9] = {};

  int _local_location_id;
  int _cpu_location_id;
 public:
  ExtractSession(std::shared_ptr<CacheContext> cache_ctx);
 private:

  void LaunchWaitAllSyncer() {
    if (_remote_syncer) _remote_syncer->on_send_job();
    if (   _cpu_syncer)    _cpu_syncer->on_send_job();
    if ( _local_syncer)  _local_syncer->on_send_job();
    if (_remote_syncer) _remote_syncer->on_wait_job_done();
    if (   _cpu_syncer)    _cpu_syncer->on_wait_job_done();
    if ( _local_syncer)  _local_syncer->on_wait_job_done();
  }

  template<typename IdxStore_T = IdxStore>
  void SplitGroup(const IdxStore_T idx_store, const size_t len, IdType * & group_offset, StreamHandle stream);

#ifdef COLL_HASH_VALID_LEGACY
  void GetMissCacheIndexByCub(DstVal* & output_dst_index,
    const IdType* nodes, const size_t num_nodes,
    IdType * & group_offset,
    StreamHandle stream);
#endif

  template<typename IdxStore_T = IdxStore>
  void GetMissCacheIndex(
    IdxStore_T & idx,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream);
  template<typename IdxStore_T = IdxStore>
  void SortIndexByLoc(IdxStore_T & idx, const size_t num_nodes, StreamHandle stream);

#ifdef DEAD_CODE
  void SortByLocation(
    IdType* &sorted_nodes,
    const IdType* nodes, const size_t num_nodes, 
    StreamHandle stream);
#endif

  template<typename IdxStore_T = IdxStore>
  void CombineOneGroup(const IdxStore_T idx_store, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block = 0, bool async = false);
  template<typename IdxStore_T = IdxStore>
  void CombineOneGroupRevised(const IdxStore_T idx_store, const size_t num_node, const void* src_data, void* output, StreamHandle stream, IdType limit_block = 0, bool async = false);

#ifdef COLL_HASH_VALID_LEGACY
  void CombineNoGroup(const IdType* nodes, const size_t num_nodes, void* output, Context ctx, DataType _dtype, IdType _dim, StreamHandle stream);
#endif
  void CombineMixGroup(const SrcKey* src_key, const DstVal* dst_val, const size_t num_nodes, void* output, Context ctx, DataType _dtype, IdType _dim, StreamHandle stream);
  void CombineCliq(const IdType* keys, const size_t num_keys, void* output, Context ctx, DataType _dtype, IdType _dim, StreamHandle stream);
  template<int NUM_LINK, typename IdxStore_T = IdxStore>
  void CombineConcurrent(const IdxStore_T idx_store, const IdType * group_offset, void* output, StreamHandle stream);
  template<int NUM_LINK, typename IdxStore_T = IdxStore>
  void CombineFused(const IdxStore_T idx_store, const IdType * group_offset, void* output, StreamHandle stream);

 public:
  void ExtractFeat(const IdType* nodes, const size_t num_nodes, void* output, StreamHandle stream, uint64_t task_key);
  inline constexpr bool IsLegacy() { return false; }
};

class RefreshSession {
 public:
  StreamHandle stream;
  StreamHandle low_pri_stream;
  std::shared_ptr<CacheContext> _cache_ctx;
#ifdef COLL_HASH_VALID_LEGACY
  void refresh_after_solve(bool foreground);
  void refresh_after_solve_old(bool foreground);
#endif
  void refresh_after_solve_new(bool foreground);
  void refresh_after_solve_main(bool foreground);
  // preserved allocation for new hashtable allocation
  TensorPtr __keys_buffer       = nullptr;
  TensorPtr __offs_buffer       = nullptr;
  TensorPtr Empty1DReuse(TensorPtr& preserved_buffer, DataType dtype, std::vector<size_t> shape,
                         Context ctx);
  TensorPtr CopyTo1DReuse(TensorPtr& preserved_buffer, TensorPtr src, Context ctx,
                          StreamHandle stream);
};

}
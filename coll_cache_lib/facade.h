#pragma once
#include "common.h"
// #include "logging.h"
#include "cache_context.h"
#include "profiler.h"
#include <cuda_runtime.h>

namespace coll_cache_lib {

// using AsymmLinkDesc = coll_cache::common::coll_cache::AsymmLinkDesc;
// using AsymmLinkDesc = coll_cache::common::coll_cache::AsymmLinkDesc;
// using namespace coll_cache;
using namespace common;

// global cache manager. one per process
class CollCache : public std::enable_shared_from_this<CollCache> {
 private:
  TensorPtr _nid_to_block;
  TensorPtr _block_density;
  TensorPtr _block_placement;
  TensorPtr _block_access_advise;
  // AtomicBarrier *_process_barrier;
  BarHandle _process_barrier;
  BarHandle _replica_barrier;
  friend class CacheContext;
  friend class RefreshSession;
  std::vector<std::shared_ptr<CacheContext>> _cache_ctx_list;
  std::vector<std::shared_ptr<ExtractSession>> _session_list;
  std::vector<std::shared_ptr<RefreshSession>> _refresh_session_list;
  void solve_impl_master(IdType *ranking_nodes_list_ptr,
             IdType *ranking_nodes_freq_list_ptr, IdType num_node);
  void solve_impl_slave();
 public:
  CollCache(BarHandle process_barrier, BarHandle replica_barrier) : _process_barrier(process_barrier), _replica_barrier(replica_barrier), _profiler(std::make_shared<Profiler>()) {}
  // CollCache *instance() {
  //   static CollCache instance;
  //   return &instance;
  // }

  void solve(IdType *ranking_nodes_list_ptr,
             IdType *ranking_nodes_freq_list_ptr, IdType num_node);

  void build(std::function<std::function<MemHandle(size_t)>(int)> allocator_builder,
             void *cpu_data, DataType dtype, size_t dim,
             double cache_percentage, StreamHandle stream = nullptr);
  void lookup(int replica_id, const IdType *nodes, const size_t num_nodes,
              void *output, StreamHandle stream, uint64_t step_key);

  void build_v2(int replica_id, IdType *ranking_nodes_list_ptr,
                IdType *ranking_nodes_freq_list_ptr, IdType num_node,
                std::function<MemHandle(size_t)> gpu_mem_allocator,
                void *cpu_data, DataType dtype, size_t dim,
                double cache_percentage, StreamHandle stream = nullptr);

  void refresh(int replica_id, IdType *ranking_nodes_list_ptr,
                IdType *ranking_nodes_freq_list_ptr, StreamHandle stream = nullptr);
  void report_avg();
  void report_last_epoch(uint64_t epoch);
  void report(uint64_t key);
  std::shared_ptr<Profiler> _profiler;

  // for profile
  inline void set_step_profile_value(uint64_t epoch, uint64_t step, LogStepItem item, double val) {
    uint64_t key = RunConfig::GetBatchKey(epoch, step);
    _profiler->LogStep(key, item, val);
  }
  inline void add_epoch_profile_value(uint64_t epoch, LogEpochItem item, double val) {
    uint64_t key = RunConfig::GetBatchKey(epoch, 0);
    _profiler->LogEpochAdd(key, item, val);
  }
};

};  // namespace coll_cache_lib
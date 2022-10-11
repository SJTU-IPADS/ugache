#pragma once
#include "common.h"
#include "logging.h"
#include "cache_context.h"
#include <cuda_runtime.h>

namespace coll_cache_lib {

// using AsymmLinkDesc = coll_cache::common::coll_cache::AsymmLinkDesc;
// using AsymmLinkDesc = coll_cache::common::coll_cache::AsymmLinkDesc;
// using namespace coll_cache;
using namespace common;

// global cache manager. one per process
class CollCache : std::enable_shared_from_this<CollCache> {
 private:
  TensorPtr _nid_to_block;
  TensorPtr _block_placement;
  TensorPtr _block_access_advise;
  AtomicBarrier *_barrier;
  friend class CacheContext;
  std::vector<std::shared_ptr<CacheContext>> _cache_ctx_list;
  std::vector<std::shared_ptr<ExtractSession>> _session_list;
 public:
  CollCache* instance() {
    static CollCache instance;
    return &instance;
  }

  void solve(IdType *ranking_nodes_list_ptr,
             IdType *ranking_nodes_freq_list_ptr, IdType num_node);

  void build(std::function<std::function<MemHandle(size_t)>(int)> allocator_builder,
             void *cpu_data, DataType dtype, size_t dim,
             double cache_percentage, StreamHandle stream = nullptr);
  void lookup(int replica_id, const IdType *nodes, const size_t num_nodes,
              void *output, StreamHandle stream);
};




};
#include "common.h"
#include "cpu/cpu_device.h"
#include "cpu/mmap_cpu_device.h"
#include "run_config.h"
#include "logging.h"
#include "coll_cache/ndarray.h"
#include "coll_cache/optimal_solver_class.h"
#include "atomic_barrier.h"
#include <cuda_runtime.h>

namespace coll_cache_lib {

// using AsymmLinkDesc = coll_cache::common::coll_cache::AsymmLinkDesc;
// using AsymmLinkDesc = coll_cache::common::coll_cache::AsymmLinkDesc;
// using namespace coll_cache;
using namespace common;

// global cache manager. one per process
class CollCache {
 private:
  TensorPtr _nid_to_block;
  TensorPtr _block_placement;
  TensorPtr _block_access_advise;
  AtomicBarrier *_barrier;
  friend class CacheContext;
 public:
  CollCache* instance() {
    static CollCache instance;
    return &instance;
  }

  void solve(IdType* ranking_nodes_list_ptr, IdType* ranking_nodes_freq_list_ptr, IdType num_node) {
    // currently we only allow single stream
    if (RunConfig::worker_id == 0) {
      std::vector<int> trainer_to_stream(RunConfig::num_device, 0);
      std::vector<int> trainer_cache_percent(RunConfig::num_device, std::round(RunConfig::cache_percentage*100));
      // simply assume we use gpu 0 for now
      Context gpu_ctx = GPU();
      RunConfig::coll_cache_link_desc = coll_cache::AsymmLinkDesc::AutoBuild(gpu_ctx);

      auto ranking_nodes_list = Tensor::FromBlob(ranking_nodes_list_ptr, coll_cache::get_data_type<IdType>(), {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_list");
      auto ranking_nodes_freq_list = Tensor::FromBlob(ranking_nodes_freq_list_ptr, coll_cache::get_data_type<IdType>(), {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_freq_list");

      coll_cache::CollCacheSolver * solver = nullptr;
      switch (RunConfig::cache_policy) {
        case kRepCache: {
          solver = new coll_cache::RepSolver();
          break;
        }
        case kPartRepCache: {
          solver = new coll_cache::PartRepSolver();
          break;
        }
        case kPartitionCache: {
          solver = new coll_cache::PartitionSolver();
          break;
        }
        case kCollCacheIntuitive: {
          solver = new coll_cache::IntuitiveSolver();
          break;
        }
        case kCollCache: {
          solver = new coll_cache::OptimalSolver();
          break;
        }
        case kCollCacheAsymmLink: {
          solver = new coll_cache::OptimalAsymmLinkSolver();
          break;
        }
        case kCliquePart: {
          solver = new coll_cache::CliquePartSolver();
          break;
        }
        case kCliquePartByDegree: {
          solver = new coll_cache::CliquePartByDegreeSolver(_dataset->ranking_nodes);
          break;
        }
        default: CHECK(false);
      }
      
      LOG(ERROR) << "solver created. now build & solve";
      _nid_to_block = Tensor::CreateShm(Constant::kCollCacheNIdToBlockShmName, kI32, {num_node}, "nid_to_block");
      solver->Build(ranking_nodes_list, ranking_nodes_freq_list, trainer_to_stream, num_node, _nid_to_block);
      LOG(ERROR) << "solver built. now solve";
      solver->Solve(trainer_to_stream, trainer_cache_percent, "BIN", RunConfig::coll_cache_hyperparam_T_local, RunConfig::coll_cache_hyperparam_T_cpu);
      LOG(ERROR) << "solver solved";
      _block_placement = solver->block_placement;
      delete solver;
    }
    _barrier->Wait();
    if (RunConfig::worker_id != 0) {
      _nid_to_block = Tensor::OpenShm(Constant::kCollCacheNIdToBlockShmName, kI32, {num_node}, "nid_to_block");
      _block_placement = Tensor::OpenShm(Constant::kCollCachePlacementShmName, kU8, {}, "coll_cache_block_placement");
      if (RunConfig::cache_policy == kCollCacheAsymmLink ||
          RunConfig::cache_policy == kCliquePart ||
          RunConfig::cache_policy == kCliquePartByDegree
          ) {
        size_t num_blocks = _block_placement->Shape()[0];
        _block_access_advise = Tensor::OpenShm(Constant::kCollCacheAccessShmName, kU8, {static_cast<decltype(num_blocks)>(RunConfig::num_device), num_blocks}, "block_access_advise");
        
      }
    }
    // time_presample = tp.Passed();
  }

  void build() {}
};




};
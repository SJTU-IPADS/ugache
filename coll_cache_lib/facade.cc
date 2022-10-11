#include "facade.h"
#include "coll_cache/ndarray.h"
#include "coll_cache/optimal_solver_class.h"
#include <thread>

namespace coll_cache_lib {

void CollCache::build(std::function<std::function<MemHandle(size_t)>(int)> allocator_builder,
                      void *cpu_data, DataType dtype, size_t dim, double cache_percentage,
                      StreamHandle stream) {
  // size_t num_thread = RunConfig::cross_process ? 1 : RunConfig::num_device;
  std::vector<std::thread> builder_threads(RunConfig::num_device);
  for (size_t i = 0; i < RunConfig::num_device; i++) {
    builder_threads[i] = std::thread([&](){
      if (RunConfig::cross_process && i != RunConfig::worker_id) return;
      int device_id = RunConfig::device_id_list[i];
      auto cache_ctx = std::make_shared<CacheContext>();
      Context gpu_ctx = GPU(device_id);
      cache_ctx->build(allocator_builder(device_id), i, this->shared_from_this(), cpu_data, dtype, dim, gpu_ctx, cache_percentage);
      this->_cache_ctx_list[i] = cache_ctx;
      this->_session_list[i] = std::make_shared<ExtractSession>(cache_ctx);
    });
  }

  for (size_t i = 0; i < RunConfig::num_device; i++) {
    builder_threads[i].join();
  }
}

void CollCache::lookup(int replica_id, const IdType *nodes,
                       const size_t num_nodes, void *output,
                       StreamHandle stream) {
  _session_list[replica_id]->ExtractFeat(nodes, num_nodes, output, stream, 0);
}
void CollCache::solve(IdType *ranking_nodes_list_ptr,
                      IdType *ranking_nodes_freq_list_ptr, IdType num_node) {
  // currently we only allow single stream
  if (RunConfig::worker_id == 0) {
    std::vector<int> trainer_to_stream(RunConfig::num_device, 0);
    std::vector<int> trainer_cache_percent(
        RunConfig::num_device, std::round(RunConfig::cache_percentage * 100));
    // simply assume we use gpu 0 for now
    Context gpu_ctx = GPU();
    RunConfig::coll_cache_link_desc =
        coll_cache::AsymmLinkDesc::AutoBuild(gpu_ctx);

    auto ranking_nodes_list = Tensor::FromBlob(
        ranking_nodes_list_ptr, coll_cache::get_data_type<IdType>(),
        {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_list");
    auto ranking_nodes_freq_list = Tensor::FromBlob(
        ranking_nodes_freq_list_ptr, coll_cache::get_data_type<IdType>(),
        {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_freq_list");

    coll_cache::CollCacheSolver *solver = nullptr;
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
      CHECK(false);
      // solver = new coll_cache::CliquePartByDegreeSolver(_dataset->ranking_nodes);
      break;
    }
    default:
      CHECK(false);
    }

    LOG(ERROR) << "solver created. now build & solve";
    _nid_to_block = Tensor::CreateShm(Constant::kCollCacheNIdToBlockShmName,
                                      kI32, {num_node}, "nid_to_block");
    solver->Build(ranking_nodes_list, ranking_nodes_freq_list,
                  trainer_to_stream, num_node, _nid_to_block);
    LOG(ERROR) << "solver built. now solve";
    solver->Solve(trainer_to_stream, trainer_cache_percent, "BIN",
                  RunConfig::coll_cache_hyperparam_T_local,
                  RunConfig::coll_cache_hyperparam_T_cpu);
    LOG(ERROR) << "solver solved";
    _block_placement = solver->block_placement;
    delete solver;
  }
  _barrier->Wait();
  if (RunConfig::worker_id != 0) {
    _nid_to_block = Tensor::OpenShm(Constant::kCollCacheNIdToBlockShmName, kI32,
                                    {num_node}, "nid_to_block");
    _block_placement = Tensor::OpenShm(Constant::kCollCachePlacementShmName,
                                       kU8, {}, "coll_cache_block_placement");
    if (RunConfig::cache_policy == kCollCacheAsymmLink ||
        RunConfig::cache_policy == kCliquePart ||
        RunConfig::cache_policy == kCliquePartByDegree) {
      size_t num_blocks = _block_placement->Shape()[0];
      _block_access_advise = Tensor::OpenShm(
          Constant::kCollCacheAccessShmName, kU8,
          {static_cast<decltype(num_blocks)>(RunConfig::num_device),
           num_blocks},
          "block_access_advise");
    }
  }
  // time_presample = tp.Passed();
}
} // namespace coll_cache_lib
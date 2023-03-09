#include "facade.h"
#include "cache_context.h"
#include "coll_cache/ndarray.h"
#include "coll_cache/optimal_solver_class.h"
#include "cpu/mmap_cpu_device.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"
#include <thread>

namespace coll_cache_lib {

// fixme : build function should not create thread internal. instead, provide function and let app call it from multiple threads/processes.
void CollCache::build(std::function<std::function<MemHandle(size_t)>(int)> allocator_builder,
                      void *cpu_data, DataType dtype, size_t dim, double cache_percentage,
                      StreamHandle stream) {
  // size_t num_thread = RunConfig::cross_process ? 1 : RunConfig::num_device;
  std::vector<std::thread> builder_threads(RunConfig::num_device);
  {
    // fixme: is this a per-process call, or per-thread call?
    CUDA_CALL(cudaHostRegister(cpu_data, RoundUp<size_t>(_nid_to_block->Shape()[0] * dim * GetDataTypeBytes(dtype), 1 << 21), cudaHostRegisterDefault | cudaHostRegisterReadOnly));
    this->_cache_ctx_list.resize(RunConfig::num_device);
    this->_session_list.resize(RunConfig::num_device);
  }
  for (size_t replica_id = 0; replica_id < RunConfig::num_device; replica_id++) {
    builder_threads[replica_id] = std::thread([&, replica_id](){
      if (RunConfig::cross_process && replica_id != RunConfig::worker_id) return;
      StreamHandle build_stream = stream;
      int device_id = RunConfig::device_id_list[replica_id];
      CUDA_CALL(cudaSetDevice(device_id));
      if (!RunConfig::cross_process) {
        CUDA_CALL(cudaStreamCreate((cudaStream_t*)(&build_stream)));
      }
      LOG(ERROR) << "worker " << RunConfig::worker_id << " thread " << replica_id << " initing device " << device_id;
      auto cache_ctx = std::make_shared<CacheContext>(_replica_barrier);
      Context gpu_ctx = GPU(device_id);
      cache_ctx->build(allocator_builder(replica_id), replica_id, this->shared_from_this(), cpu_data, dtype, dim, gpu_ctx, cache_percentage, build_stream);
      Device::Get(gpu_ctx)->StreamSync(gpu_ctx, build_stream);
      Device::Get(gpu_ctx)->FreeStream(gpu_ctx, build_stream);
      this->_cache_ctx_list[replica_id] = cache_ctx;
      this->_session_list[replica_id] = std::make_shared<ExtractSession>(cache_ctx);
    });
  }

  for (size_t i = 0; i < RunConfig::num_device; i++) {
    builder_threads[i].join();
  }
}

void CollCache::lookup(int replica_id, const IdType *nodes,
                       const size_t num_nodes, void *output,
                       StreamHandle stream, uint64_t step_key) {
  Timer t;
  _session_list[replica_id]->ExtractFeat(nodes, num_nodes, output, stream, step_key);
  _profiler->LogStep(step_key, common::kLogL2CacheCopyTime, t.Passed());
}

// for master, the ptr must be valid; 
// for slave, leave them empty

void CollCache::solve_impl_master(IdType *ranking_nodes_list_ptr,
                      IdType *ranking_nodes_freq_list_ptr, IdType num_node) {
    using PerT = common::coll_cache::PerT;
    std::vector<int> trainer_to_stream(RunConfig::num_device, 0);
    // std::vector<int> trainer_cache_percent(
    //     RunConfig::num_device, std::round(RunConfig::cache_percentage * 100));
    std::vector<PerT> trainer_cache_percent(
        RunConfig::num_device, RunConfig::cache_percentage * 100);
    // replica 0 is master
    // Context gpu_ctx = GPU(RunConfig::device_id_list[0]);

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
    _block_access_advise = solver->block_access_from;
    _block_density = solver->block_density_tensor;
    delete solver;
}
void CollCache::solve_impl_slave() {
    _nid_to_block = Tensor::OpenShm(Constant::kCollCacheNIdToBlockShmName, kI32,
                                    {}, "nid_to_block");
    _block_placement = Tensor::OpenShm(Constant::kCollCachePlacementShmName,
                                       kU8, {}, "coll_cache_block_placement");
    // if (RunConfig::cache_policy == kCollCacheAsymmLink) {
    //   // for refreshment to know about sizes
    // }
    if (RunConfig::cache_policy == kCollCacheAsymmLink ||
        RunConfig::cache_policy == kRepCache ||
        RunConfig::cache_policy == kCliquePart ||
        RunConfig::cache_policy == kCliquePartByDegree) {
      size_t num_blocks = _block_placement->Shape()[0];
      _block_access_advise = Tensor::OpenShm(
          Constant::kCollCacheAccessShmName, kU8,
          {static_cast<decltype(num_blocks)>(RunConfig::num_device),
           num_blocks},
          "block_access_advise");
      _block_density = Tensor::OpenShm(Constant::kCollCachePlacementShmName + "_density",
                                        kF64, {}, "coll_cache_block_density");
    }
}

void CollCache::solve(IdType *ranking_nodes_list_ptr,
                      IdType *ranking_nodes_freq_list_ptr, IdType num_node) {
  // currently we only allow single stream
  if (RunConfig::worker_id == 0) {
    solve_impl_master(ranking_nodes_list_ptr, ranking_nodes_freq_list_ptr, num_node);
  }
  _process_barrier->Wait();
  if (RunConfig::worker_id != 0) {
    solve_impl_slave();
  }
  // time_presample = tp.Passed();
}
// CollCache::CollCache(BarHandle process_barrier, BarHandle replica_barrier) {
//   _barrier = barrier;
//   // int fd = cpu::MmapCPUDevice::CreateShm(sizeof(AtomicBarrier), Constant::kCollCacheBuilderShmName);
//   // int num_process = RunConfig::cross_process ? RunConfig::num_device : 1;
//   // _process_barrier = new (cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), sizeof(AtomicBarrier), fd))AtomicBarrier(num_process, RunConfig::worker_id == 0);
// }
void CollCache::build_v2(int replica_id, IdType *ranking_nodes_list_ptr,
                         IdType *ranking_nodes_freq_list_ptr, IdType num_node,
                         std::function<MemHandle(size_t)> gpu_mem_allocator,
                         void *cpu_data, DataType dtype, size_t dim,
                         double cache_percentage, StreamHandle stream) {
  int device_id = RunConfig::device_id_list[replica_id];
  if (RunConfig::cross_process || replica_id == 0) {
    // one-time call for each process
    RunConfig::LoadConfigFromEnv();
    RunConfig::coll_cache_link_desc = coll_cache::AsymmLinkDesc::AutoBuild(GPU(device_id));
    size_t num_node_host_mem = num_node;
    if (RunConfig::option_empty_feat != 0 && cache_percentage != 0) {
      num_node_host_mem = 1 << RunConfig::option_empty_feat;
    }
    LOG(ERROR) << "registering cpu data with " << ToReadableSize(RoundUp<size_t>(num_node_host_mem * dim * GetDataTypeBytes(dtype), 1 << 21));
    CUDA_CALL(cudaHostRegister(cpu_data, RoundUp<size_t>(num_node_host_mem * dim * GetDataTypeBytes(dtype), 1 << 21), cudaHostRegisterDefault | cudaHostRegisterReadOnly));
    LOG(ERROR) << "registering cpu data done.";
    this->_cache_ctx_list.resize(RunConfig::num_device);
    this->_session_list.resize(RunConfig::num_device);
    this->_refresh_session_list.resize(RunConfig::num_device);
  }
  bool need_solver = (cache_percentage != 0 && cache_percentage != 1);
  if (replica_id == 0 && need_solver) {
    solve_impl_master(ranking_nodes_list_ptr, ranking_nodes_freq_list_ptr, num_node);
    LOG(ERROR) << replica_id << " solved master";
  }
  this->_replica_barrier->Wait();
  if (replica_id != 0 && RunConfig::cross_process && need_solver) {
    // one-time call for none-master process
    solve_impl_slave();
  }
  LOG(ERROR) << replica_id << " solved";

  // if (RunConfig::cross_process) return;
  LOG(ERROR) << "worker " << RunConfig::worker_id << " thread " << replica_id << " initing device " << device_id;
  auto cache_ctx = std::make_shared<CacheContext>(_replica_barrier);
  Context gpu_ctx = GPU(device_id);
  cache_ctx->build(gpu_mem_allocator, replica_id, this->shared_from_this(), cpu_data, dtype, dim, gpu_ctx, cache_percentage, stream);
  Device::Get(gpu_ctx)->StreamSync(gpu_ctx, stream);
  this->_cache_ctx_list[replica_id] = cache_ctx;
  this->_session_list[replica_id] = std::make_shared<ExtractSession>(cache_ctx);
  this->_refresh_session_list[replica_id] = std::make_shared<RefreshSession>();
  this->_refresh_session_list[replica_id]->_cache_ctx = cache_ctx;
  this->_refresh_session_list[replica_id]->stream = stream;
  this->_replica_barrier->Wait();
}

void CollCache::refresh(int replica_id, IdType *ranking_nodes_list_ptr,
                         IdType *ranking_nodes_freq_list_ptr, StreamHandle stream, bool foreground) {
  int device_id = RunConfig::device_id_list[replica_id];
  if (RunConfig::cross_process || replica_id == 0) {
    // one-time call for each process
    size_t num_block = _block_density->Shape()[0];

    auto copy_shm = [replica_id](std::string path, TensorPtr old_t) -> TensorPtr {
      if (old_t == nullptr) return nullptr;
      auto ret = Tensor::CreateShm(path, old_t->Type(), old_t->Shape(), "");
      if (replica_id == 0) {
        memcpy(ret->MutableData(), old_t->Data(), old_t->NumBytes());
      }
      return ret;
    };

    this->_old_block_access_advise = copy_shm(Constant::kCollCacheAccessShmName + "_old", _block_access_advise);
    this->_old_block_density = copy_shm(Constant::kCollCachePlacementShmName + "_density" + "_old", _block_density);
    this->_old_block_placement = copy_shm(Constant::kCollCachePlacementShmName + "_old",            _block_placement);
    this->_old_nid_to_block = copy_shm(Constant::kCollCacheNIdToBlockShmName + "_old",              _nid_to_block);

    this->_block_access_advise = nullptr;
    this->_block_density = nullptr;
    this->_block_placement = nullptr;
    this->_nid_to_block = nullptr;
  }
  AnonymousBarrier::_refresh_instance->Wait();
  if (replica_id == 0) {
    solve_impl_master(ranking_nodes_list_ptr, ranking_nodes_freq_list_ptr, RunConfig::num_total_item);
    LOG(ERROR) << replica_id << " solved master";
  }
  AnonymousBarrier::_refresh_instance->Wait();
  if (replica_id != 0 && RunConfig::cross_process) {
    // one-time call for none-master process
    solve_impl_slave();
  }
  LOG(ERROR) << replica_id << " solved";
  AnonymousBarrier::_refresh_instance->Wait();

  // if (RunConfig::cross_process) return;
  LOG(ERROR) << "worker " << RunConfig::worker_id << " thread " << replica_id << " refresh device " << device_id;
  this->_refresh_session_list[replica_id]->stream = stream;
  // this->_refresh_session_list[replica_id]->refresh_after_solve(foreground);
  this->_refresh_session_list[replica_id]->refresh_after_solve_main(foreground);
  AnonymousBarrier::_refresh_instance->Wait();
}

void CollCache::report_last_epoch(uint64_t epoch) {
  _profiler->ReportStepAverageLastEpoch(epoch + 1, 0);
  // // _profiler->ReportStepMax(RunConfig::num_epoch - 1, RunConfig::num_global_step_per_epoch - 1);
  // // _profiler->ReportStepMin(RunConfig::num_epoch - 1, RunConfig::num_global_step_per_epoch - 1);
  // for (size_t epoch = 1; epoch < RunConfig::num_epoch; epoch ++) {
  //   _profiler->ReportStepAverage(epoch, RunConfig::num_global_step_per_epoch - 1);
  //   _profiler->ReportStepMax(epoch, RunConfig::num_global_step_per_epoch - 1);
  //   _profiler->ReportStepMin(epoch, RunConfig::num_global_step_per_epoch - 1);
  // }
  std::cout.flush();
}
void CollCache::report_avg() {
  _profiler->ReportStepAverage(RunConfig::num_epoch - 1, RunConfig::num_global_step_per_epoch - 1);
  _profiler->ReportStepItemPercentiles(RunConfig::num_epoch - 1, RunConfig::num_global_step_per_epoch - 1,
        kLogL2CacheCopyTime, {50, 90, 95, 99, 99.9}, "tail_logl2featcopy");
  // // _profiler->ReportStepMax(RunConfig::num_epoch - 1, RunConfig::num_global_step_per_epoch - 1);
  // // _profiler->ReportStepMin(RunConfig::num_epoch - 1, RunConfig::num_global_step_per_epoch - 1);
  // for (size_t epoch = 1; epoch < RunConfig::num_epoch; epoch ++) {
  //   _profiler->ReportStepAverage(epoch, RunConfig::num_global_step_per_epoch - 1);
  //   _profiler->ReportStepMax(epoch, RunConfig::num_global_step_per_epoch - 1);
  //   _profiler->ReportStepMin(epoch, RunConfig::num_global_step_per_epoch - 1);
  // }
  std::cout.flush();
}
void CollCache::report(uint64_t key) {
  _profiler->ReportStep(RunConfig::GetEpochFromKey(key), RunConfig::GetStepFromKey(key));
  std::cout.flush();
}
}  // namespace coll_cache_lib
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

#define COLL_SWITCH_NBYTE(nbyte_in, DTypeName, ...) \
  { \
    switch (nbyte_in) { \
      case 2:  { typedef short   DTypeName; {__VA_ARGS__}; break; }\
      case 4:  { typedef float   DTypeName; {__VA_ARGS__}; break; }\
      case 8:  { typedef double  DTypeName; {__VA_ARGS__}; break; }\
      case 16: { typedef int4    DTypeName; {__VA_ARGS__}; break; }\
      case 32: { typedef double4 DTypeName; {__VA_ARGS__}; break; }\
      \
    };\
  };

DataType nb_to_dt(size_t nbyte_in) { 
  switch (nbyte_in) { 
    case 2:  { return kF16;   break; }
    case 4:  { return kF32;   break; }
    case 8:  { return kF64;   break; }
    case 16: { return kF64_2; break; }
    case 32: { return kF64_4; break; }
    default: CHECK(false);
  };
};

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
  // if (step_key >= RunConfig::num_device && step_key < 2 * RunConfig::num_device) {
  //   auto cpu_t = Tensor::Empty(common::kI32, {num_nodes}, CPU(), "");
  //   CUDA_CALL(cudaMemcpy(cpu_t->MutableData(), nodes, cpu_t->NumBytes(), cudaMemcpyDefault));
  //   LOG(ERROR) << "replica " << replica_id << " saving step " << step_key << " with " << num_nodes << " keys";
  //   std::ofstream f("/tmp/coll.lookup." + std::to_string(replica_id));
  //   f.write((char*)cpu_t->Data(), cpu_t->NumBytes());
  //   f.close();
  // }
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
    LOG(ERROR) << "creating solver";
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
    _nid_to_block = Tensor::CreateShm(solver->_shm_name_nid_to_block,
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
void CollCache::solve_impl_master(ContFreqBuf* freq_rank, IdType num_node) {
    using PerT = common::coll_cache::PerT;
    std::vector<int> trainer_to_stream(RunConfig::num_device, 0);
    // std::vector<int> trainer_cache_percent(
    //     RunConfig::num_device, std::round(RunConfig::cache_percentage * 100));
    std::vector<PerT> trainer_cache_percent(
        RunConfig::num_device, RunConfig::cache_percentage * 100);
    // replica 0 is master
    // Context gpu_ctx = GPU(RunConfig::device_id_list[0]);

    LOG(ERROR) << "creating solver";
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
    _nid_to_block = Tensor::CreateShm(solver->_shm_name_nid_to_block,
                                      kI32, {num_node}, "nid_to_block");
    solver->BuildSingleStream(freq_rank, trainer_to_stream, num_node, _nid_to_block);
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
    _nid_to_block = Tensor::OpenShm(coll_cache::CollCacheSolver::_shm_name_nid_to_block, kI32,
                                    {}, "nid_to_block");
    _block_placement = Tensor::OpenShm(coll_cache::CollCacheSolver::_shm_name_place,
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
          coll_cache::CollCacheSolver::_shm_name_access, kU8,
          {static_cast<decltype(num_blocks)>(RunConfig::num_device),
           num_blocks},
          "block_access_advise");
      _block_density = Tensor::OpenShm(coll_cache::CollCacheSolver::_shm_name_dens,
                                        kF64, {}, "coll_cache_block_density");
    }
}

#ifdef DEAD_CODE
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
#endif
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
  }
  // if (replica_id == 0) {
  //   std::ofstream f("/tmp/coll.rank");
  //   f.write((char*)ranking_nodes_list_ptr, sizeof(IdType) * num_node);
  //   f.write((char*)ranking_nodes_freq_list_ptr, sizeof(IdType) * num_node);
  //   f.close();
  // }
  // if (GetDataTypeBytes(dtype) < 16) {
  //   LOG(ERROR) << "before scale, dtype is " << dtype << ", dim is " << dim;
  //   size_t scale = 16 / GetDataTypeBytes(dtype);
  //   if (scale <= dim) {
  //     dim /= scale;
  //     dtype = kF64_2;
  //   }
  //   LOG(ERROR) << "after scale=" << scale << ", dtype is " << dtype << ", new dim is " << dim;
  // }
  if (RunConfig::coll_cache_scale_nb != 0) {
    LOG(ERROR) << "before scale, dtype is " << dtype << ", dim is " << dim;
    size_t emb_vec_nb = GetDataTypeBytes(dtype) * dim;
    if (emb_vec_nb >= RunConfig::coll_cache_scale_nb && emb_vec_nb % RunConfig::coll_cache_scale_nb == 0) {
      dim = emb_vec_nb / RunConfig::coll_cache_scale_nb;
      dtype = nb_to_dt(RunConfig::coll_cache_scale_nb);
    }
    LOG(ERROR) << "after scale, dtype is " << dtype << ", new dim is " << dim;
  }
  if (RunConfig::cross_process || replica_id == 0) {
    // one-time call for each process
    // RunConfig::LoadConfigFromEnv();
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
    RunConfig::solver_omp_thread_num = RunConfig::omp_thread_num;
    RunConfig::solver_omp_thread_num_per_gpu = RunConfig::omp_thread_num / RunConfig::num_device;
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
  RunConfig::solver_omp_thread_num = RunConfig::refresher_omp_thread_num;
  RunConfig::solver_omp_thread_num_per_gpu = RunConfig::refresher_omp_thread_num_per_gpu;
  CUDA_CALL(cudaGetLastError());
}

void CollCache::refresh(int replica_id, IdType *ranking_nodes_list_ptr,
                         IdType *ranking_nodes_freq_list_ptr, StreamHandle stream, bool foreground) {
  AnonymousBarrier::_refresh_instance->Wait();
  int device_id = RunConfig::device_id_list[replica_id];
  if (RunConfig::cross_process || replica_id == 0) {
    if (replica_id == 0) LOG(ERROR) << "Preserving old solution";
    // one-time call for each process


    coll_cache::CollCacheSolver::_shm_name_nid_to_block.swap(coll_cache::CollCacheSolver::_shm_name_alter_nid_to_block);
    coll_cache::CollCacheSolver::_shm_name_access.swap(coll_cache::CollCacheSolver::_shm_name_alter_access);
    coll_cache::CollCacheSolver::_shm_name_place.swap(coll_cache::CollCacheSolver::_shm_name_alter_place);
    coll_cache::CollCacheSolver::_shm_name_dens.swap(coll_cache::CollCacheSolver::_shm_name_alter_dens);

    this->_old_block_access_advise = this->_block_access_advise;
    this->_old_block_density       = this->_block_density;
    this->_old_block_placement     = this->_block_placement;
    this->_old_nid_to_block        = this->_nid_to_block;

    this->_block_access_advise = nullptr;
    this->_block_density = nullptr;
    this->_block_placement = nullptr;
    this->_nid_to_block = nullptr;
  }
  AnonymousBarrier::_refresh_instance->Wait();
  if (replica_id == 0) {
    if (replica_id == 0) LOG(ERROR) << "old solution preserved, now solve";
    // RunConfig::solver_omp_thread_num = RunConfig::refresher_omp_thread_num;
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
  // LOG(ERROR) << "worker " << RunConfig::worker_id << " thread " << replica_id << " refresh device " << device_id;
  this->_refresh_session_list[replica_id]->stream = stream;
  // this->_refresh_session_list[replica_id]->refresh_after_solve(foreground);
  this->_refresh_session_list[replica_id]->refresh_after_solve_main(foreground);
  AnonymousBarrier::_refresh_instance->Wait();
}

void CollCache::build_v2(int replica_id, ContFreqBuf* freq_rank, IdType num_node,
                         std::function<MemHandle(size_t)> gpu_mem_allocator,
                         void *cpu_data, DataType dtype, size_t dim,
                         double cache_percentage, StreamHandle stream) {
  int device_id = RunConfig::device_id_list[replica_id];
  if (RunConfig::cross_process || replica_id == 0) {
    // one-time call for each process
    RunConfig::LoadConfigFromEnv();
  }
  if (RunConfig::coll_cache_scale_nb != 0) {
    LOG(ERROR) << "before scale, dtype is " << dtype << ", dim is " << dim;
    size_t emb_vec_nb = GetDataTypeBytes(dtype) * dim;
    if (emb_vec_nb >= RunConfig::coll_cache_scale_nb && emb_vec_nb % RunConfig::coll_cache_scale_nb == 0) {
      dim = emb_vec_nb / RunConfig::coll_cache_scale_nb;
      dtype = nb_to_dt(RunConfig::coll_cache_scale_nb);
    }
    LOG(ERROR) << "after scale, dtype is " << dtype << ", new dim is " << dim;
  }
  if (RunConfig::cross_process || replica_id == 0) {
    // one-time call for each process
    // RunConfig::LoadConfigFromEnv();
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
    RunConfig::solver_omp_thread_num = RunConfig::omp_thread_num;
    RunConfig::solver_omp_thread_num_per_gpu = RunConfig::omp_thread_num / RunConfig::num_device;
  }
  bool need_solver = (cache_percentage != 0 && cache_percentage != 1);
  if (replica_id == 0 && need_solver) {
    solve_impl_master(freq_rank, num_node);
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
  RunConfig::solver_omp_thread_num = RunConfig::refresher_omp_thread_num;
  RunConfig::solver_omp_thread_num_per_gpu = RunConfig::refresher_omp_thread_num_per_gpu;
  CUDA_CALL(cudaGetLastError());
}

void CollCache::refresh(int replica_id, ContFreqBuf* freq_rank, StreamHandle stream, bool foreground) {
  AnonymousBarrier::_refresh_instance->Wait();
  int device_id = RunConfig::device_id_list[replica_id];
  if (RunConfig::cross_process || replica_id == 0) {
    if (replica_id == 0) LOG(ERROR) << "Preserving old solution";
    // one-time call for each process


    coll_cache::CollCacheSolver::_shm_name_nid_to_block.swap(coll_cache::CollCacheSolver::_shm_name_alter_nid_to_block);
    coll_cache::CollCacheSolver::_shm_name_access.swap(coll_cache::CollCacheSolver::_shm_name_alter_access);
    coll_cache::CollCacheSolver::_shm_name_place.swap(coll_cache::CollCacheSolver::_shm_name_alter_place);
    coll_cache::CollCacheSolver::_shm_name_dens.swap(coll_cache::CollCacheSolver::_shm_name_alter_dens);

    this->_old_block_access_advise = this->_block_access_advise;
    this->_old_block_density       = this->_block_density;
    this->_old_block_placement     = this->_block_placement;
    this->_old_nid_to_block        = this->_nid_to_block;

    this->_block_access_advise = nullptr;
    this->_block_density = nullptr;
    this->_block_placement = nullptr;
    this->_nid_to_block = nullptr;
  }
  AnonymousBarrier::_refresh_instance->Wait();
  if (replica_id == 0) {
    if (replica_id == 0) LOG(ERROR) << "old solution preserved, now solve";
    // RunConfig::solver_omp_thread_num = RunConfig::refresher_omp_thread_num;
    solve_impl_master(freq_rank, RunConfig::num_total_item);
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
  // LOG(ERROR) << "worker " << RunConfig::worker_id << " thread " << replica_id << " refresh device " << device_id;
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
  if (RunConfig::num_bucket_step)
    _profiler->ReportSequentialAverage(RunConfig::num_bucket_step, std::cout);
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
#include "../coll_cache_lib/common.h"
#include "../coll_cache_lib/constant.h"
#include "../coll_cache_lib/logging.h"
#include "../coll_cache_lib/facade.h"
#include "../coll_cache_lib/cpu/cpu_utils.h"
#include "../coll_cache_lib/cuda/cuda_device.h"
#include "../coll_cache_lib/cpu/mmap_cpu_device.h"
#include "atomic_barrier.h"
#include "../coll_cache_lib/freq_recorder.h"
#include "profiler.h"
#include "run_config.h"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>
#include "./workspace_pool.h"
#include "timer.h"
CLI::App _app;
namespace {
using namespace coll_cache_lib;
using namespace common;
using common::IdType;

std::unordered_map<std::string, CachePolicy> cache_policy_strs = {
  {"degree"          ,  kCacheByDegree},
  {"heuristic"       ,  kCacheByHeuristic},
  {"pre_sample"      ,  kCacheByPreSample},
  {"degree_hop"      ,  kCacheByDegreeHop},
  {"presample_static",  kCacheByPreSampleStatic},
  {"fake_optimal"    ,  kCacheByFakeOptimal},
  {"dynamic_cache"   ,  kDynamicCache},
  {"random"          ,  kCacheByRandom},
  {"rep"             ,  kRepCache},
  {"clique_part"     ,  kCliquePart},
  {"coll"            ,  kCollCacheAsymmLink},
};

std::unordered_map<std::string, std::string> configs;
std::string env_profile_level = "3";
std::string env_log_level = "warn";
std::string env_empty_feat = "0";

int enabled_gpus = 0;
size_t dim = 128;
size_t num_keys = 100000000;
size_t num_step_per_epoch_per_replica = 100;

void InitOptions(std::string app_name) {
  configs = {
    {"_cache_policy",  std::to_string((int)kRepCache)},
    {"cache_policy",  "rep"},
    // {"worker_id",  "0"},
  };
  _app.add_option("--cache-policy", configs["cache_policy"])
      ->check(CLI::IsMember({
          "degree",
          "heuristic",
          "pre_sample",
          "degree_hop",
          "presample_static",
          "fake_optimal",
          "dynamic_cache",
          "random",
          "rep",
          "clique_part",
          "coll",
      }));
  _app.add_option("--cache-percentage",           RunConfig::cache_percentage);
  _app.add_option("--num-device",                 RunConfig::num_device);
  _app.add_option("--omp-thread-num",             RunConfig::omp_thread_num);
  _app.add_option("--profile-level",              env_profile_level);
  _app.add_option("--log-level",                  env_log_level);
  _app.add_option("--empty-feat",                 env_empty_feat);
  _app.add_option("--enabled-gpus",               enabled_gpus);
  _app.add_option("--dim",                        dim);
  _app.add_option("--num-keys",                   num_keys);
  _app.add_option("--num-epoch",                  RunConfig::num_epoch);
  _app.add_option("--num-step",                   num_step_per_epoch_per_replica);
}
void Parse(int argc, char** argv) {
  try {
    _app.parse(argc, argv);
  } catch(const CLI::ParseError &e) {
    _app.exit(e);
    exit(1);
  }
  configs["_cache_policy"] = std::to_string((int)cache_policy_strs[configs["cache_policy"]]);
  setenv(Constant::kEnvProfileLevel.c_str(), env_profile_level.c_str(), 1);
  setenv("SAMGRAPH_LOG_LEVEL", env_log_level.c_str(), 1);
  setenv(Constant::kEnvEmptyFeat.c_str(), env_empty_feat.c_str(), 1);

  RunConfig::cache_policy = cache_policy_strs[configs["cache_policy"]];

  // std::cout << "('cache_policy', "      << configs["cache_policy"]      << ")\n";

}

};

class DemoMemHandle : public ExternelGPUMemoryHandler {
 public:
  void* dev_ptr = nullptr;
  size_t nbytes_ = 0;
  void* ptr() override {return dev_ptr;}
  size_t nbytes() override { return nbytes_; }
  ~DemoMemHandle() { CUDA_CALL(cudaFree(dev_ptr)); }
};
// Atomic Barrier should be placed at mmap shared memory, cannot be recycled by ~shared_ptr, so we have to warp it.
class DemoBarrier : public ExternalBarrierHandler {
 public:
  AtomicBarrier* barrier;
  DemoBarrier(int worker) {
    auto mmap_ctx = MMAP(MMAP_RW_DEVICE);
    auto dev = Device::Get(mmap_ctx);
    barrier = new(dev->AllocArray<AtomicBarrier>(mmap_ctx, 1)) AtomicBarrier(worker);
  }
  void Wait() override { barrier->Wait(); }
};

int main(int argc, char** argv) {
  size_t batch_size = 8192*100;
  RunConfig::cache_percentage = 0.125;
  InitOptions("");
  Parse(argc, argv);

  if (enabled_gpus == 0) {
    enabled_gpus = RunConfig::num_device;
  }

  RunConfig::device_id_list.resize(RunConfig::num_device);
  for (int i = 0; i < RunConfig::num_device; i++) {
    RunConfig::device_id_list[i] = i;
  }
  RunConfig::cross_process = true;
  // RunConfig::num_epoch = 10;
  RunConfig::num_global_step_per_epoch = RunConfig::num_device * num_step_per_epoch_per_replica;
  RunConfig::num_total_item = num_keys;
  // RunConfig::cache_policy = coll_cache_lib::common::kCollCacheAsymmLink;
  // RunConfig::cache_policy = coll_cache_lib::common::kCliquePart;

  BarHandle _replica_barrier = std::make_shared<DemoBarrier>(RunConfig::num_device);

  auto cache_manager = std::make_shared<CollCache>(nullptr, _replica_barrier);

  auto mmap_ctx = MMAP(MMAP_RO_DEVICE);
  auto mmap_dev = Device::Get(mmap_ctx);
  void* cpu_data = mmap_dev->AllocArray<float>(mmap_ctx, num_keys * dim);

  int replica_id = 0;
  for (int i = 1; i < RunConfig::num_device; i++) {
    pid_t pid = fork();
    if (pid == 0) {
      replica_id = i;
      break;
    }
  }

  std::cout << "replica " << replica_id << " at process " << getpid() << "\n";
  // sleep(30);


  thread_local std::mt19937 gen(0x45678f1 * replica_id);

  IdType* rank_vec = nullptr;
  IdType* freq_vec = nullptr;

  if (replica_id == 0) {
    rank_vec = new IdType[num_keys];
    freq_vec = new IdType[num_keys];
    cpu::ArrangeArray(rank_vec, num_keys);
    std::exponential_distribution<double> dist(1);
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < num_keys; i++) {
      freq_vec[i] = 1;
      // freq_vec[i] = std::ceil(dist(gen));
    }
  }

  LOG(ERROR) << replica_id << " is at process " << getpid();
  std::function<MemHandle(size_t)> gpu_mem_allocator = [replica_id](size_t nbytes) {
    int dev_id = RunConfig::device_id_list[replica_id];
    // LOG(ERROR) << "replica " << replica_id << " try to allocate " << ToReadableSize(nbytes) << " on device " << dev_id;
    CUDA_CALL(cudaSetDevice(dev_id));
    void* ptr;
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
    std::shared_ptr<DemoMemHandle> handle = std::make_shared<DemoMemHandle>();
    handle->dev_ptr = ptr;
    handle->nbytes_ = nbytes;
    return handle;
  };
  WorkspacePool* wpool = new WorkspacePool(gpu_mem_allocator);

  gpu_mem_allocator = [wpool](size_t nbytes) {
    return wpool->AllocWorkspace(nbytes);
  };

  std::uniform_int_distribution<IdType> dist_int(0, num_keys - 1);

  int device_id = RunConfig::device_id_list[replica_id];
  auto stream = Device::Get(GPU(device_id))->CreateStream(GPU(device_id));

  cache_manager->build_v2(replica_id, rank_vec, freq_vec, num_keys, gpu_mem_allocator, cpu_data, kF32, dim, RunConfig::cache_percentage, stream);

  auto freq_recorder_ = std::make_shared<FreqRecorder>(RunConfig::num_total_item, replica_id);

  LOG(ERROR) << "replica " << replica_id << " build done";

  int dev_id = RunConfig::device_id_list[replica_id];
  CUDA_CALL(cudaSetDevice(dev_id));
  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(&stream);
  CUDA_CALL(cudaStreamCreate(cu_stream));
  auto key_list_cpu = Tensor::Empty(kI32, {batch_size}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
  auto output_handle = gpu_mem_allocator(batch_size * dim * GetDataTypeBytes(kF32));
  auto output = output_handle->ptr();
  // void * output = new float[batch_size * dim];
  double iter_time = 0;
  for (int iteration = 0; iteration < num_step_per_epoch_per_replica * RunConfig::num_epoch; ) {
    int next_stop = iteration + num_step_per_epoch_per_replica;
    for (; iteration < next_stop; iteration++) {
      _replica_barrier->Wait();
      // #pragma omp parallel for num_threads(RunConfig::omp_thread_num / RunConfig::num_device)
      for (int i = 0; i < batch_size; i++) {
        // IdType k = dist_int(gen);
        IdType k = (IdType)(dist(gen)) % num_keys;
        key_list_cpu->Ptr<IdType>()[i] = k;
      }
      freq_recorder_->Record(key_list_cpu->Ptr<IdType>(), batch_size);
      _replica_barrier->Wait();
      auto key_list = Tensor::CopyToExternal(key_list_cpu, gpu_mem_allocator, GPU(dev_id), stream);
      if (!(replica_id < enabled_gpus)) batch_size = 0;
      _replica_barrier->Wait();
      Timer t;
      cache_manager->lookup(replica_id, key_list->Ptr<IdType>(), batch_size, output, stream, replica_id + iteration * RunConfig::num_device);
      _replica_barrier->Wait();
      iter_time += t.Passed();
    }
    if (replica_id == 0) {
      LOG(ERROR) << "replica " << replica_id << " round " << iteration << " done, time is " << iter_time / num_step_per_epoch_per_replica;
      iter_time = 0;
    }
    if (replica_id == 0) {
      cache_manager->report_last_epoch((iteration - 1) / num_step_per_epoch_per_replica);
    }

    if (replica_id == 0) {
      auto freq_recorder = freq_recorder_;

      freq_recorder->Combine();
      freq_recorder->Sort();
      freq_recorder->GetFreq(freq_vec);
      freq_recorder->GetRankNode(rank_vec);
    }

    cache_manager->refresh(replica_id, rank_vec, freq_vec, stream);
    _replica_barrier->Wait();
  }
  for (int i = 0; i < RunConfig::num_device; i++) {
    _replica_barrier->Wait();
    if (i != replica_id) {
      continue;
    }
    if (replica_id < enabled_gpus) {
      cache_manager->report_avg();
    }
  }

  // samgraph::common::samgraph_config_from_map(configs);
  // samgraph::common::samgraph_init();
  // for (size_t i = 0; i < num_epoch; i++) {
  //   for (size_t b = 0; b < samgraph::common::samgraph_steps_per_epoch(); b++) {
  //     samgraph::common::samgraph_sample_once();
  //     samgraph::common::samgraph_get_next_batch();
  //     // samgraph::common::samgraph_report_step(i, b);
  //   }
  //   // samgraph::common::samgraph_report_epoch(i);
  // }
  // samgraph::common::samgraph_report_step_average(num_epoch-1, samgraph::common::samgraph_steps_per_epoch()-1);
  // samgraph::common::samgraph_report_epoch_average(num_epoch-1);
  // samgraph::common::samgraph_report_init();
  // samgraph::common::samgraph_report_node_access();
  // samgraph::common::samgraph_dump_trace();
  // samgraph::common::samgraph_shutdown();
}

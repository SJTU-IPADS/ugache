#include "../coll_cache_lib/common.h"
#include "../coll_cache_lib/constant.h"
#include "../coll_cache_lib/logging.h"
#include "../coll_cache_lib/facade.h"
#include "../coll_cache_lib/cpu/cpu_utils.h"
#include "../coll_cache_lib/cuda//cuda_device.h"
#include "atomic_barrier.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <random>
#include <thread>
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
};

std::unordered_map<std::string, std::string> configs;
std::string env_profile_level = "3";
std::string env_log_level = "warn";
std::string env_empty_feat = "0";

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
      }));
  _app.add_option("--cache-percentage",           RunConfig::cache_percentage);
  _app.add_option("--num-device",                 RunConfig::num_device);
  _app.add_option("--omp-thread-num",             RunConfig::omp_thread_num);
  _app.add_option("--profile-level",              env_profile_level);
  _app.add_option("--log-level",                  env_log_level);
  _app.add_option("--empty-feat",                 env_empty_feat);
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
  void* ptr() override {return dev_ptr;}
  ~DemoMemHandle() { CUDA_CALL(cudaFree(dev_ptr)); }
};
class DemoBarrier : public ExternalBarrierHandler {
 public:
  AtomicBarrier barrier;
  DemoBarrier(int worker) : barrier(worker) {}
  void Wait() override { barrier.Wait(); }
};

int main(int argc, char** argv) {
  size_t num_keys = 1000000;
  size_t dim = 128;
  size_t batch_size = 1000;
  InitOptions("");
  Parse(argc, argv);

  RunConfig::cache_percentage = 0.5;

  RunConfig::device_id_list.resize(RunConfig::num_device);
  for (int i = 0; i < RunConfig::num_device; i++) {
    RunConfig::device_id_list[i] = i;
  }

  // BarHandle _process_barrier = std::make_shared<DemoBarrier>(1);
  BarHandle _replica_barrier = std::make_shared<DemoBarrier>(RunConfig::num_device);

  auto cache_manager = std::make_shared<CollCache>(nullptr, _replica_barrier);
  IdType* ranking_nodes_list_ptr = new IdType[num_keys];
  IdType* ranking_nodes_freq_list_ptr = new IdType[num_keys];
  void* cpu_data = new float[num_keys * dim];
  cpu::ArrangeArray(ranking_nodes_list_ptr, num_keys);
  thread_local std::random_device rd;
  // thread_local std::mt19937 gen(rd());
  thread_local std::mt19937 gen(RunConfig::seed);
  std::exponential_distribution<double> dist(1);
  #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_keys; i++) {
    ranking_nodes_freq_list_ptr[i] = std::ceil(dist(gen));
  }
  // cache_manager->solve(ranking_nodes_list_ptr, ranking_nodes_freq_list_ptr, num_keys);
  std::function<std::function<MemHandle(size_t)>(int)> allocator_builder = [](int replica_id) {
    std::function<MemHandle(size_t)> lambda = [replica_id](size_t nbytes) {
      int dev_id = RunConfig::device_id_list[replica_id];
      LOG(ERROR) << "replica " << replica_id << " try to allocate " << ToReadableSize(nbytes) << " on device " << dev_id;
      CUDA_CALL(cudaSetDevice(dev_id));
      void* ptr;
      CUDA_CALL(cudaMalloc(&ptr, nbytes));
      std::shared_ptr<DemoMemHandle> handle = std::make_shared<DemoMemHandle>();
      handle->dev_ptr = ptr;
      return handle;
    };
    return lambda;
  };
  // cache_manager->build(allocator_builder, cpu_data, kF32, dim, RunConfig::cache_percentage);


  std::uniform_int_distribution<IdType> dist_int(0, num_keys - 1);
  std::vector<std::thread> workers(RunConfig::num_device);

  for (int i = 0; i < RunConfig::num_device; i++) {
    workers[i] = std::thread([&, replica_id=i]()mutable{
      // std::function<MemHandle(size_t)> gpu_mem_allocator = [replica_id](size_t nbytes) {
      //   int dev_id = RunConfig::device_id_list[replica_id];
      //   LOG(ERROR) << "replica " << replica_id << " try to allocate " << ToReadableSize(nbytes) << " on device " << dev_id;
      //   CUDA_CALL(cudaSetDevice(dev_id));
      //   void* ptr;
      //   CUDA_CALL(cudaMalloc(&ptr, nbytes));
      //   std::shared_ptr<DemoMemHandle> handle = std::make_shared<DemoMemHandle>();
      //   handle->dev_ptr = ptr;
      //   return handle;
      // };
      int device_id = RunConfig::device_id_list[replica_id];
      auto gpu_mem_allocator = allocator_builder(replica_id);
      auto stream = Device::Get(GPU(device_id))->CreateStream(GPU(device_id));
      
      cache_manager->build_v2(replica_id, ranking_nodes_list_ptr, ranking_nodes_freq_list_ptr, num_keys, gpu_mem_allocator, cpu_data, kF32, dim, RunConfig::cache_percentage, stream);
    });
  }
  for (int i = 0; i < RunConfig::num_device; i++) {
    workers[i].join();
  }


  for (int i = 0; i < RunConfig::num_device; i++) {
    workers[i] = std::thread([replica_id=i, batch_size, dist_int, cache_manager, dim, allocator_builder]()mutable{
      int dev_id = RunConfig::device_id_list[replica_id];
      CUDA_CALL(cudaSetDevice(dev_id));
      StreamHandle stream;
      cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(&stream);
      CUDA_CALL(cudaStreamCreate(cu_stream));
      auto key_list_cpu = Tensor::Empty(kI32, {batch_size}, CPU(), "");
      auto output_handle = allocator_builder(replica_id)(batch_size * dim * GetDataTypeBytes(kF32));
      auto output = output_handle->ptr();
      // void * output = new float[batch_size * dim];
      for (int iteration = 0; iteration < 10; iteration++) {
        #pragma omp parallel for num_threads(RunConfig::omp_thread_num / RunConfig::num_device)
        for (int key = 0; key < batch_size; key++) {
          key_list_cpu->Ptr<IdType>()[key] = dist_int(gen);
        }
        auto key_list = Tensor::CopyToExternal(key_list_cpu, allocator_builder(replica_id), GPU(dev_id), stream);
        cache_manager->lookup(replica_id, key_list->Ptr<IdType>(), batch_size, output, stream);
      }
    });
  }
  for (int i = 0; i < RunConfig::num_device; i++) {
    workers[i].join();
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

#include "asymm_link_desc.h"
#include "../logging.h"
#include "../run_config.h"
#include "../common.h"
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)

namespace coll_cache_lib {
namespace common {
namespace coll_cache {

void AsymmLinkDesc::BuildSwitch(int num_trainer) {
  _topo_type = kSwitch;
  link_src = vec<vec<vec<int>>>(num_trainer, vec<vec<int>>(1, vec<int>(num_trainer - 1)));

  bool useRolling = false;
  switch (RunConfig::rolling) {
    case AutoRolling:
      switch(RunConfig::cache_policy) {
        case kCollCacheAsymmLink: 
        case kCollCacheIntuitive: 
        case kCollCache: 
          useRolling = true; break;
        default: break;
      }
      break;
    case EnableRolling: useRolling = true; break;
    default: break;
  }

  for (int dst_dev = 0; dst_dev < num_trainer; dst_dev++) {
    // each device has single link, which connect to all remote device
    std::cout << dst_dev << " : link #0 : ";
    if (useRolling) {
      for (int remote_device = 0; remote_device < num_trainer - 1; remote_device++) {
        link_src[dst_dev][0][remote_device] = {(dst_dev + remote_device + 1) %  num_trainer};
        std::cout << link_src[dst_dev][0][remote_device] << ",";
      }
    } else {
      for (int remote_device = 0, i = 0; remote_device < num_trainer; remote_device++) {
        if (remote_device == dst_dev) continue;
        link_src[dst_dev][0][i] = remote_device;
        std::cout << link_src[dst_dev][0][i++] << ",";
      }
    }
    std::cout << "\n";
  }
  link_time = vec<vec<double>>(num_trainer, vec<double>(1, RunConfig::coll_cache_hyperparam_T_remote));
  compute_percent = vec<vec<double>>(num_trainer, vec<double>(1, 1.));
}
void AsymmLinkDesc::BuildSymmHardWire(int num_trainer) {
  _topo_type = kHardWiredSymm;
  int num_link = num_trainer - 1;
  link_src = vec<vec<vec<int>>>(num_trainer, vec<vec<int>>(num_link));
  for (int dst_dev = 0; dst_dev < num_trainer; dst_dev++) {
    // each device has multiple link, each link contains only one remote device
    std::cout << dst_dev << " : ";
    for (int src_link = 0; src_link < num_link; src_link++) {
      // link_src[dst_dev][src_link] = {(src_link < dst_dev) ? src_link : (src_link + 1)};
      link_src[dst_dev][src_link] = {(dst_dev + src_link + 1) % num_trainer};
      std::cout << " {link #" << src_link << " : " << link_src[dst_dev][src_link][0] << "},";
    }
    std::cout << "\n";
  }
  link_time = vec<vec<double>>(
      num_trainer,
      vec<double>(num_link, RunConfig::coll_cache_hyperparam_T_remote));
  compute_percent =
      vec<vec<double>>(num_trainer, vec<double>(num_link, 1. / num_link));
}
void AsymmLinkDesc::BuildAsymmHardWire(int num_trainer) {
  _topo_type = kHardWiredAsymm;
  double fast_link_time = RunConfig::coll_cache_hyperparam_T_remote;
  double slow_link_time = RunConfig::coll_cache_hyperparam_T_remote * 2;
  if (num_trainer == 8) {
    link_src = {
        /**
         * ┌-----------┐
         * 2 = 1 = 7 = 4
         * ║ x │   │ x ║
         * 3 = 0 = 6 = 5
         * └-----------┘
         */
        {{3}, {6}, {1}, {2}}, // 0
        {{7}, {2}, {3}, {0}}, // 1
        {{1}, {3}, {0}, {4}}, // 2
        {{2}, {0}, {5}, {1}}, // 3
        {{5}, {7}, {2}, {6}}, // 4
        {{6}, {4}, {7}, {3}}, // 5
        {{0}, {5}, {4}, {7}}, // 6
        {{4}, {1}, {6}, {5}}, // 7
    };
    link_time = vec<vec<double>>(num_trainer, 
        {{fast_link_time, fast_link_time, slow_link_time, slow_link_time},});
    compute_percent = vec<vec<double>>(
        num_trainer, vec<double>({1./3, 1./3, 1./6, 1./6}));
    return;
  }
  if (num_trainer == 6) {
    // remove 0 & 6 by reordering CUDA_VISIBLE_DEVICES
    link_src = {
        /**
         * ┌-----------┐
         * 1 = 0 = 5 = 3
         * ║ ╱       ╲ ║
         * 2           4
         * └-----------┘
         */
        // {{1}, {5}, {2}, { }}, // 0
        // {{0}, {2}, {3}, { }}, // 1
        // {{1}, { }, {0}, {4}}, // 2
        // {{4}, {5}, {1}, { }}, // 3
        // {{3}, { }, {2}, {5}}, // 4
        // {{0}, {3}, {4}, { }}, // 5
        {{1}, {5}, {2},    }, // 0
        {{0}, {2}, {3},    }, // 1
        {{1},      {0}, {4}}, // 2
        {{4}, {5}, {1},    }, // 3
        {{3},      {2}, {5}}, // 4
        {{0}, {3}, {4},    }, // 5
    };
    // link_time = vec<vec<double>>(num_trainer,{{fast_link_time, fast_link_time, slow_link_time, slow_link_time},});
    link_time = {
        {fast_link_time, fast_link_time, slow_link_time},
        {fast_link_time, fast_link_time, slow_link_time},
        {fast_link_time,                 slow_link_time, slow_link_time},
        {fast_link_time, fast_link_time, slow_link_time},
        {fast_link_time,                 slow_link_time, slow_link_time},
        {fast_link_time, fast_link_time, slow_link_time},
    };
    compute_percent = {
        // {2./5, 2./5, 1./5,    0},
        // {2./5, 2./5, 1./5,    0},
        // {1./2,    0, 1./4, 1./4},
        // {2./5, 2./5, 1./5,    0},
        // {1./2,    0, 1./4, 1./4},
        // {2./5, 2./5, 1./5,    0},
        {2./5, 2./5, 1./5,     },
        {2./5, 2./5, 1./5,     },
        {1./2,       1./4, 1./4},
        {2./5, 2./5, 1./5,     },
        {1./2,       1./4, 1./4},
        {2./5, 2./5, 1./5,     },
    };
    return;
  }
  CHECK(false) << "Unsupported configuration";
}

AsymmLinkDesc AsymmLinkDesc::AutoBuild(int num_trainer, int total_gpu,
                                       std::string gpu_model) {
  int cpu_sm_decision = 0;
  int remote_sm_decision = 0;
  AsymmLinkDesc desc;
  if (gpu_model.find("A100") != std::string::npos) {
    // 730;30;9;
    RunConfig::coll_cache_hyperparam_T_remote = 730 / (double)30;
    RunConfig::coll_cache_hyperparam_T_cpu    = 730 / (double)8;
    if (GetEnv("COLL_CPU_TIME_OVERRIDE") != "") {
      RunConfig::coll_cache_hyperparam_T_cpu    = 730 / std::stod(GetEnv("COLL_CPU_TIME_OVERRIDE"));
    }
    if (GetEnv("COLL_REMOTE_TIME_OVERRIDE") != "") {
      RunConfig::coll_cache_hyperparam_T_remote    = 730 / std::stod(GetEnv("COLL_REMOTE_TIME_OVERRIDE"));
    }
    if (RunConfig::concurrent_link_impl == kMPS || 
        RunConfig::concurrent_link_impl == kMPSPhase || 
        RunConfig::concurrent_link_impl == kMPSForLandC || 
        RunConfig::concurrent_link_impl == kMultiKernelNumBlock || 
        RunConfig::concurrent_link_impl == kMultiKernelNumBlockOld) {
      cpu_sm_decision = 10;
      remote_sm_decision = 56;
    }
  } else if (gpu_model.find("V100") != std::string::npos) {
    RunConfig::coll_cache_hyperparam_T_remote = 350 / (double)38;
    RunConfig::coll_cache_hyperparam_T_cpu    = 350 / (double)11;
    if (total_gpu > 4) {
      LOG(ERROR) << "v100x8, slow pcie";
      RunConfig::coll_cache_hyperparam_T_cpu    = 350 / (double)3.4;
    }
    if (GetEnv("COLL_CPU_TIME_OVERRIDE") != "") {
      RunConfig::coll_cache_hyperparam_T_cpu    = 350 / std::stod(GetEnv("COLL_CPU_TIME_OVERRIDE"));
    }
    if (GetEnv("COLL_REMOTE_TIME_OVERRIDE") != "") {
      RunConfig::coll_cache_hyperparam_T_remote    = 350 / std::stod(GetEnv("COLL_REMOTE_TIME_OVERRIDE"));
    }
    if (RunConfig::concurrent_link_impl == kMPS || 
        RunConfig::concurrent_link_impl == kMPSPhase || 
        RunConfig::concurrent_link_impl == kMPSForLandC || 
        RunConfig::concurrent_link_impl == kMultiKernelNumBlock || 
        RunConfig::concurrent_link_impl == kMultiKernelNumBlockOld) {
      cpu_sm_decision = 8;
      // remote_sm_decision = 60;
      remote_sm_decision = 72;
      // CHECK(false) << "not supported now";
    }
  } else {
    CHECK(false) << "No bandwidth data for " << gpu_model;
  }

  if (GetEnv("COLL_CPU_SM_NUM_OVERRIDE") != "") {
    cpu_sm_decision = std::stoi(GetEnv("COLL_CPU_SM_NUM_OVERRIDE"));
  }
  LOG(ERROR) << "assigning " << cpu_sm_decision << " to cpu";
  desc.cpu_sm.resize(num_trainer, cpu_sm_decision);
  desc.total_sm_for_remote.resize(num_trainer, remote_sm_decision);

  std::string topo_name;
  if (total_gpu <= 4) {
    topo_name = "symm";
    desc.BuildSymmHardWire(num_trainer);
  } else if (gpu_model.find("V100") != std::string::npos) {
    topo_name = "asymm";
    desc.BuildAsymmHardWire(num_trainer);
  } else if (gpu_model.find("A100") != std::string::npos) {
    topo_name = "symm";
    desc.BuildSymmHardWire(num_trainer);
  } else {
    CHECK(false) << "Unsupported configuration: " << total_gpu << " X " << gpu_model;
  }
  LOG(ERROR) << "build " << topo_name << " link desc with " << num_trainer << "X " << gpu_model << " out of " << total_gpu;
  LOG(ERROR) << "remote time is " << RunConfig::coll_cache_hyperparam_T_remote;
  LOG(ERROR) << "cpu time is " << RunConfig::coll_cache_hyperparam_T_cpu;
  return desc;
}
AsymmLinkDesc AsymmLinkDesc::AutoBuild(Context ctx) {
  int total_gpu;
  CUDA_CALL(cudaGetDeviceCount(&total_gpu));
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, ctx.device_id));
  // fixme
  auto desc = AutoBuild(RunConfig::num_device, total_gpu, prop.name);
  desc.local_sm.resize(desc.cpu_sm.size(), prop.multiProcessorCount - desc.cpu_sm[0]);
  // desc.SMPercentToNum(prop.multiProcessorCount - desc.cpu_sm[0]);
  desc.SMPercentToNum(desc.total_sm_for_remote[0]);
  for (int dst_dev = 0; dst_dev < RunConfig::num_device; dst_dev++) {
    std::cout << dst_dev << " : local " << desc.local_sm[dst_dev] << ", cpu " << desc.cpu_sm[dst_dev];
    for (int src_link = 0; src_link < desc.link_src[dst_dev].size(); src_link++) {
      std::cout << " {link #" << src_link << " : g" << desc.link_src[dst_dev][src_link][0] << " " << desc.link_sm[dst_dev][src_link] << "},";
    }
    std::cout << "\n";
  }
  return desc;
}
void AsymmLinkDesc::SMPercentToNum(int total_sm) {
  link_sm.resize(compute_percent.size());
  FOR_LOOP(dev_id, compute_percent.size()) {
    link_sm[dev_id].resize(compute_percent[dev_id].size());
    FOR_LOOP(link, compute_percent[dev_id].size()) {
      link_sm[dev_id][link] = std::round(total_sm * compute_percent[dev_id][link]);
    }
  }
}

bool AutoEnableConcurrentLink() {
  switch(RunConfig::cache_policy) {
    case kCollCacheAsymmLink:
    case kCollCacheIntuitive:
    case kCollCache:
      return true;
    // case kCacheByHeuristic:
    // case kCacheByPreSample:
    // case kCacheByPreSampleStatic:
    // case kPartitionCache:
    // case kPartRepCache:
    // case kCacheByDegree:
    // case kCacheByDegreeHop:
    // case kCacheByFakeOptimal:
    // case kCacheByRandom:
    // case kDynamicCache:
    default:
      return false;
  }
}
void AutoHandlePartImpl() {
  if (RunConfig::coll_hash_impl == kHashImplAuto) {
    if (RunConfig::cache_percentage * RunConfig::coll_cache_link_desc.CliqueSize() - 1 > -1e-6 &&
        RunConfig::coll_cache_no_group == kDirectNoGroup) {
      RunConfig::coll_skip_hash = true;
      RunConfig::coll_hash_impl = kChunk;
      LOG(ERROR) << "full partition, skip hash set to true";
    } else {
      RunConfig::coll_skip_hash = false;
      RunConfig::coll_hash_impl = kDefault;
    }
  }
  if (RunConfig::coll_skip_hash) {
    CHECK(RunConfig::coll_hash_impl == kChunk || RunConfig::coll_hash_impl == kRR);
  }
  LOG(ERROR) << "using hash table impl " << RunConfig::coll_hash_impl << ", skip hash is " << RunConfig::coll_skip_hash;
}

double AsymmLinkDesc::AggregatedRemoteTime() {
  CHECK(_topo_type != kHardWiredAsymm);
  if (RunConfig::coll_cache_concurrent_link) {
    return RunConfig::coll_cache_hyperparam_T_remote / link_time[0].size();
  } else {
    return RunConfig::coll_cache_hyperparam_T_remote;
  }
}
int AsymmLinkDesc::CliqueSize() {
  switch (_topo_type) {
    case kSwitch:
    case kHardWiredSymm: return link_src.size();
    case kHardWiredAsymm: CHECK((link_src.size() % 2) == 0); return link_src.size() / 2;
    default: CHECK(false);
  }
}
void AutoScaleDim(DataType& dtype, size_t& dim, Context ctx) {
  size_t total_nbytes = GetDataTypeBytes(dtype) * dim;
  CHECK((total_nbytes % 32) == 0) << "only support perfectly aligned dim for now.";
  cudaDeviceProp prop;
  CUDA_CALL(cudaGetDeviceProperties(&prop, ctx.device_id));
  // fixme
  std::string gpu_model = prop.name;

  auto scale_func = [&dtype, &dim](DataType target_dtype){
    CHECK((GetDataTypeBytes(target_dtype) % GetDataTypeBytes(dtype)) == 0);
    size_t scale_factor = GetDataTypeBytes(target_dtype) / GetDataTypeBytes(dtype);
    CHECK((dim % scale_factor) == 0);
    size_t old_dim = dim;
    dim /= scale_factor;
    LOG(ERROR) << "scale from " << old_dim << "*" << dtype << " to " << dim << "*" << target_dtype;
    dtype = target_dtype;
  };

  if (gpu_model.find("A100") != std::string::npos) {
    // always use kF64_4, i.e. double4
    scale_func(kF64_4);
  } else if (gpu_model.find("V100") != std::string::npos) {
    if (RunConfig::num_device <= 4) {
      // always use kF64_2, i.e. double2
      scale_func(kF64_2);
    }
    if (RunConfig::concurrent_link_impl != kMPS && total_nbytes == 1024) {
      // corner case use kF64_4
      scale_func(kF64_4);
    } else {
      scale_func(kF64_2);
    }
  }
}
}  // namespace coll_cache
}
}
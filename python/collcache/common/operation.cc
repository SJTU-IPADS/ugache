/*
 * Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "operation.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <regex>

// #include "./dist/dist_engine.h"
#include "coll_cache_lib/common.h"
#include "coll_cache_lib/constant.h"
#include "coll_cache_lib/device.h"
// #include "coll_cache_lib/engine.h"
#include "coll_cache_lib/facade.h"
#include "coll_cache_lib/logging.h"
#include "coll_cache_lib/profiler.h"
#include "coll_cache_lib/run_config.h"
#include "coll_cache_lib/timer.h"
// #include "cuda/cuda_engine.h"
#include "coll_cache_lib/run_config.h"

namespace coll_cache_lib {
namespace common {

std::shared_ptr<CollCache> _coll_cache = nullptr;
std::shared_ptr<Profiler> _profiler = nullptr;
size_t internal_feat_dim = 0;

size_t outside_feat_dim = 0;

size_t steps[9] = {0};

std::vector<std::shared_ptr<FreqRecorder>> _freq_recorder;
size_t max_num_keys = 0;

namespace {

void CriticalExec(coll_cache_lib::common::BarHandle barrier, int syncer, std::function<void()> func, int local_id) {
  for (int i = 0; i < syncer; i++) {
    if (i == local_id) { func(); }
    barrier->Wait();
  }
}

void coll_cache_config_from_map(std::unordered_map<std::string, std::string>& configs) {
  using CRC = coll_cache_lib::common::RunConfig;
  CHECK(!CRC::is_configured);

  CHECK(configs.count("cache_percentage"));
  CHECK(configs.count("_cache_policy"));
  CHECK(configs.count("num_device"));
  CHECK(configs.count("num_global_step_per_epoch"));
  CHECK(configs.count("num_epoch"));
  CHECK(configs.count("num_total_item"));
  CHECK(configs.count("omp_thread_num"));

  // CRC::raw_configs = configs;
  CRC::cache_percentage = std::stod(configs["cache_percentage"]);
  CRC::cache_policy = static_cast<coll_cache_lib::common::CachePolicy>(std::stoi(configs["_cache_policy"]));
  // fixme: do we need support single process?
  CRC::cross_process = true;
  CRC::num_device = std::stoull(configs["num_device"]);
  CRC::device_id_list.resize(CRC::num_device);
  for (int i = 0; i < CRC::num_device; i++) {CRC::device_id_list[i] = i;}
  // fixme: pass this in
  CRC::num_global_step_per_epoch = std::stoull(configs["num_global_step_per_epoch"]);
  CRC::num_epoch = std::stoull(configs["num_epoch"]);
  // fixme: pass this in
  CRC::num_total_item = std::stoull(configs["num_total_item"]);
  CRC::omp_thread_num = std::stoi(configs["omp_thread_num"]);
  // fixme: is refresh required?

  // RC::dataset_path = configs["dataset_path"];
  CRC::is_configured = true;

  _profiler = std::make_shared<Profiler>();
  _coll_cache = std::make_shared<CollCache>(nullptr, AnonymousBarrier::_global_instance);
  _coll_cache->_profiler = _profiler;
}

std::string ToReadableSizePrivate(size_t nbytes) {
  char buf[Constant::kBufferSize];
  if (nbytes > Constant::kGigabytes) {
    double new_size = (float)nbytes / Constant::kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kMegabytes) {
    double new_size = (float)nbytes / Constant::kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > Constant::kKilobytes) {
    double new_size = (float)nbytes / Constant::kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}

};

void coll_cache_init(int replica_id, size_t key_space_size, std::function<MemHandle(size_t)> allocator, void *cpu_data, DataType dtype, size_t dim, double cache_percentage, StreamHandle stream) {
  CHECK(RunConfig::is_configured);
  outside_feat_dim = dim;
  if (dtype == kF16) {
    CHECK(dim % 2 == 0);
    dtype = kF32;
    dim /= 2;
  }
  internal_feat_dim = dim;

  CriticalExec(AnonymousBarrier::_global_instance, RunConfig::num_device, [replica_id](){
    _freq_recorder[replica_id]->LocalCombineToShared();
  }, replica_id);
  ContFreqBuf* freq_rand = nullptr;
  if (replica_id == 0) {
    _freq_recorder[0]->GlobalCombine();
    _freq_recorder[0]->Sort();
    freq_rand = _freq_recorder[0]->global_cont_freq_buf;
  }
  _coll_cache->build_v2(replica_id, freq_rand, key_space_size, allocator, cpu_data, dtype, dim, cache_percentage, stream);
  LOG(INFO) << "SamGraph train has been initialized successfully";
}

extern "C" {

void coll_cache_config(const char **config_keys, const char **config_values,
                     const size_t num_config_items) {
  using RC = RunConfig;
  CHECK(!RC::is_configured);

  std::unordered_map<std::string, std::string> configs;

  for (size_t i = 0; i < num_config_items; i++) {
    std::string k(config_keys[i]);
    std::string v(config_values[i]);
    configs[k] = v;
  }
  coll_cache_config_from_map(configs);
}


// size_t coll_cache_num_epoch() {
//   return RunConfig::num_epoch;
// }

// size_t coll_cache_steps_per_epoch() {
//   CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
//   return Engine::Get()->NumStep();
// }

void coll_cache_log_step_by_key(uint64_t key, int item, double val) {
  CHECK_LT(item, kNumLogStepItems);
  _profiler->LogStep(key, static_cast<LogStepItem>(item), val);
}

void coll_cache_report_step_by_key(uint64_t key) {
  _profiler->ReportStep(RunConfig::GetEpochFromKey(key), RunConfig::GetStepFromKey(key));
  // _coll_cache->report(key);
  std::cout.flush();
}

void coll_cache_report_step_average_by_key(uint64_t key) {
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


void coll_cache_record_init(int replica_id) {
  CHECK(RunConfig::is_configured);
  _freq_recorder.resize(RunConfig::num_device);
  _freq_recorder[replica_id] = std::make_shared<FreqRecorder>(RunConfig::num_total_item, replica_id);
}

void coll_cache_train_barrier() {
  AnonymousBarrier::_global_instance->Wait();
}

// size_t coll_cache_num_local_step() {
//   return Engine::Get()->NumLocalStep();
// }

int coll_cache_wait_one_child() {
  int child_stat;
  pid_t pid = waitpid(-1, &child_stat, 0);
  if (WEXITSTATUS(child_stat) != 0) {
    LOG(ERROR) << "detect a terminated child " << pid << ", status is "
               << WEXITSTATUS(child_stat);
    return 1;
  } else if (WIFSIGNALED(child_stat)) {
    LOG(ERROR) << "detect an abnormal terminated child, signal is " << strsignal(WTERMSIG(child_stat));
    return 1;
  } else return 0;
}

void coll_cache_lookup(int replica_id, uint32_t* key, size_t num_keys, void* output, StreamHandle stream) {
  uint64_t step_key = steps[replica_id] * RunConfig::num_device + replica_id;
  steps[replica_id]++;
  _coll_cache->lookup(replica_id, key, num_keys, output, stream, step_key);
}
// void coll_cache_lookup_with_skey(int replica_id, uint32_t* key, size_t num_keys, void* output, uint64_t step_key, StreamHandle stream) {
//   _coll_cache->lookup(replica_id, key, num_keys, output, stream, step_key);
// }

void coll_cache_record(int replica_id, uint32_t* key, size_t num_keys) {
  _freq_recorder[replica_id]->Record(key, num_keys);
  max_num_keys = std::max(max_num_keys, num_keys);
}

namespace {

// size_t get_cuda_used(Context ctx) {
size_t get_cuda_used() {
  size_t free, total;
  // cudaSetDevice(ctx.device_id);
  cudaMemGetInfo(&free, &total);
  return total - free;
}

}

void coll_cache_print_memory_usage() {
  std::cout << "[CUDA] cuda" << /*ctx.device_id << */ ": usage: " << ToReadableSizePrivate(get_cuda_used()) << "\n";
  std::cout.flush();
}

}  // extern "c"

}  // namespace common
}  // namespace coll_cache_lib

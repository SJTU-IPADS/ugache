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

#include "./dist/dist_engine.h"
#include "common.h"
#include "constant.h"
#include "device.h"
#include "engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"

namespace samgraph {
namespace common {

extern "C" {

void samgraph_config(const char **config_keys, const char **config_values,
                     const size_t num_config_items) {
  using RC = RunConfig;
  CHECK(!RC::is_configured);

  std::unordered_map<std::string, std::string> configs;

  for (size_t i = 0; i < num_config_items; i++) {
    std::string k(config_keys[i]);
    std::string v(config_values[i]);
    configs[k] = v;
  }
  samgraph_config_from_map(configs);
}

void samgraph_config_from_map(std::unordered_map<std::string, std::string>& configs) {
  using RC = RunConfig;
  CHECK(!RC::is_configured);

  CHECK(configs.count("dataset_path"));
  CHECK(configs.count("_arch"));
  CHECK(configs.count("_sample_type"));
  CHECK(configs.count("batch_size"));
  CHECK(configs.count("num_epoch"));
  CHECK(configs.count("_cache_policy"));
  CHECK(configs.count("cache_percentage"));
  CHECK(configs.count("max_sampling_jobs"));
  CHECK(configs.count("max_copying_jobs"));
  CHECK(configs.count("omp_thread_num"));
  CHECK(configs.count("num_layer"));
  CHECK(configs.count("num_hidden"));
  CHECK(configs.count("lr"));
  CHECK(configs.count("dropout"));

  RC::raw_configs = configs;
  RC::run_arch = static_cast<RunArch>(std::stoi(configs["_arch"]));
  RC::cache_policy =
      static_cast<CachePolicy>(std::stoi(configs["_cache_policy"]));
  RC::cache_percentage = std::stod(configs["cache_percentage"]);

  RC::omp_thread_num = std::stoi(configs["omp_thread_num"]);

  RC::rolling = static_cast<RollingPolicy>(std::stoi(configs["rolling"]));

  switch (RC::run_arch) {
    case kArch0:
    case kArch1:
    case kArch2:
    case kArch3:
    case kArch4:
      CHECK(configs.count("sampler_ctx"));
      CHECK(configs.count("trainer_ctx"));
      RC::sampler_ctx = Context(configs["sampler_ctx"]);
      RC::trainer_ctx = Context(configs["trainer_ctx"]);
      break;
    case kArch5:
      CHECK(configs.count("num_sample_worker"));
      CHECK(configs.count("num_train_worker"));
      RC::num_sample_worker = std::stoull(configs["num_sample_worker"]);
      RC::num_train_worker = std::stoull(configs["num_train_worker"]);
      if (!configs.count("have_switcher")) {
        configs["have_switcher"] = "0";
      }
      break;
    case kArch6:
      CHECK(configs.count("num_worker"));
      RC::num_sample_worker = std::stoull(configs["num_worker"]);
      RC::num_train_worker = std::stoull(configs["num_worker"]);
      break;
    case kArch7:
      CHECK(configs.count("worker_id"));
      CHECK(configs.count("num_worker"));
      CHECK(configs.count("sampler_ctx"));
      CHECK(configs.count("trainer_ctx"));
      RC::worker_id = std::stoull(configs["worker_id"]);
      RC::num_worker = std::stoull(configs["num_worker"]);
      RC::sampler_ctx = Context(configs["sampler_ctx"]);
      RC::trainer_ctx = Context(configs["trainer_ctx"]);
      break;
    case kArch9:
      CHECK(configs.count("num_sample_worker"));
      CHECK(configs.count("num_train_worker"));
      RC::num_sample_worker = std::stoi(configs["num_sample_worker"]);
      RC::num_train_worker = std::stoi(configs["num_train_worker"]);
      CHECK(configs.count("unified_memory"));
      CHECK(configs.count("unified_memory_ctx"));
      break;
    default:
      CHECK(false);
  }

  RC::LoadConfigFromEnv();

  RC::is_configured = true;
}

void samgraph_init() {
  CHECK(RunConfig::is_configured);
  Engine::Create();
  Engine::Get()->Init();

  LOG(INFO) << "SamGraph has been initialied successfully";
}

void samgraph_start() {
  CHECK(Engine::Get()->IsInitialized() && !Engine::Get()->IsShutdown());
  if (RunConfig::option_profile_cuda) {
    CUDA_CALL(cudaProfilerStart());
  }

  Engine::Get()->Start();
  LOG(INFO) << "SamGraph has been started successfully";
}

void samgraph_shutdown() {
  Engine::Get()->Shutdown();
  if (RunConfig::option_profile_cuda) {
    CUDA_CALL(cudaProfilerStop());
  }
  LOG(INFO) << "SamGraph has been completely shutdown now";
}

void samgraph_log_step(uint64_t epoch, uint64_t step, int item, double val) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get().LogStep(key, static_cast<LogStepItem>(item), val);
}
void samgraph_log_step_by_key(uint64_t key, int item, double val) {
  CHECK_LT(item, kNumLogStepItems);
  Profiler::Get().LogStep(key, static_cast<LogStepItem>(item), val);
}

void samgraph_log_step_add(uint64_t epoch, uint64_t step, int item,
                           double val) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  Profiler::Get().LogStepAdd(key, static_cast<LogStepItem>(item), val);
}

void samgraph_log_epoch_add(uint64_t epoch, int item, double val) {
  CHECK_LT(item, kNumLogEpochItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, 0);
  Profiler::Get().LogEpochAdd(key, static_cast<LogEpochItem>(item), val);
}

double samgraph_get_log_init_value(int item) {
  CHECK_LT(item, kNumLogInitItems);
  return Profiler::Get().GetLogInitValue(static_cast<LogInitItem>(item));
}

double samgraph_get_log_step_value(uint64_t epoch, uint64_t step, int item) {
  CHECK_LT(item, kNumLogStepItems);
  uint64_t key = Engine::Get()->GetBatchKey(epoch, step);
  return Profiler::Get().GetLogStepValue(key, static_cast<LogStepItem>(item));
}
double samgraph_get_log_step_value_by_key(uint64_t key, int item) {
  CHECK_LT(item, kNumLogStepItems);
  return Profiler::Get().GetLogStepValue(key, static_cast<LogStepItem>(item));
}

double samgraph_get_log_epoch_value(uint64_t epoch, int item) {
  CHECK_LT(item, kNumLogEpochItems);
  return Profiler::Get().GetLogEpochValue(epoch,
                                          static_cast<LogEpochItem>(item));
}

void samgraph_report_init() {
  Profiler::Get().ReportInit();
  std::cout.flush();
}

void samgraph_report_step(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStep(epoch, step);
}

void samgraph_report_step_average(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStepAverage(epoch, step);
  std::cout.flush();
}
void samgraph_report_step_max(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStepMax(epoch, step);
  std::cout.flush();
}
void samgraph_report_step_min(uint64_t epoch, uint64_t step) {
  Profiler::Get().ReportStepMin(epoch, step);
  std::cout.flush();
}

void samgraph_report_epoch(uint64_t epoch) {
  Profiler::Get().ReportEpoch(epoch);
}

void samgraph_report_epoch_average(uint64_t epoch) {
  Profiler::Get().ReportEpochAverage(epoch);
  std::cout.flush();
}

void samgraph_report_node_access() {
  if (RunConfig::option_log_node_access_simple) {
    Profiler::Get().ReportNodeAccessSimple();
  }
  if (RunConfig::option_log_node_access) {
    Profiler::Get().ReportNodeAccess();
  }
  std::cout.flush();
}

void samgraph_trace_step_begin(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item), us);
}

void samgraph_trace_step_end(uint64_t key, int item, uint64_t us) {
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item), us);
}

void samgraph_trace_step_begin_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepBegin(key, static_cast<TraceItem>(item),
                                 t.TimePointMicro());
}

void samgraph_trace_step_end_now(uint64_t key, int item) {
  Timer t;
  Profiler::Get().TraceStepEnd(key, static_cast<TraceItem>(item),
                               t.TimePointMicro());
}

void samgraph_dump_trace() {
  Profiler::Get().DumpTrace(std::cerr);
  std::cerr.flush();
}

void samgraph_data_init() {
  CHECK(RunConfig::is_configured);
  Engine::Create();
  Engine::Get()->Init();

  LOG(INFO) << "SamGraph data has been initialized successfully";
}

void samgraph_sample_init(int worker_id, const char*ctx) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->SampleInit(worker_id, Context(std::string(ctx)));

  LOG(INFO) << "SamGraph sample has been initialized successfully";
}

void samgraph_train_init(int worker_id, const char*ctx) {
  CHECK(RunConfig::is_configured);
  dist::DistEngine::Get()->TrainInit(worker_id, Context(std::string(ctx)), dist::DistType::Extract);

  LOG(INFO) << "SamGraph train has been initialized successfully";
}

void samgraph_train_barrier() {
  dist::DistEngine::Get()->GetTrainerBarrier()->Wait();
}

size_t samgraph_num_local_step() {
  return Engine::Get()->NumLocalStep();
}

int samgraph_wait_one_child() {
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

namespace {

size_t get_cuda_used(Context ctx) {
  size_t free, total;
  cudaSetDevice(ctx.device_id);
  cudaMemGetInfo(&free, &total);
  return total - free;
}

}

void samgraph_print_memory_usage() {
  if (dynamic_cast<dist::DistEngine*>(Engine::Get())) {
    Context ctx;
    if (dist::DistEngine::Get()->GetDistType() == dist::DistType::Sample) {
      ctx = dist::DistEngine::Get()->GetSamplerCtx();
    } else {
      ctx = dist::DistEngine::Get()->GetTrainerCtx();
    }
    auto _target_device = Device::Get(ctx);
    std::cout << "[CUDA] cuda" << ctx.device_id << ": usage: " << ToReadableSize(get_cuda_used(ctx)) << "\n";
    std::cout << "[SAM] cuda" << ctx.device_id << " data alloc        : " << ToReadableSize(_target_device->DataSize(ctx)) << "\n";
    std::cout << "[SAM] cuda" << ctx.device_id << " workspace         : " << ToReadableSize(_target_device->WorkspaceSize(ctx)) << "\n";
    std::cout << "[SAM] cuda" << ctx.device_id << " workspace reserve : " << ToReadableSize(_target_device->FreeWorkspaceSize(ctx)) << "\n";
    std::cout << "[SAM] cuda" << ctx.device_id << " total             : " << ToReadableSize(_target_device->TotalSize(ctx)) << "\n";
  }
  std::cout.flush();
}

}  // extern "c"

}  // namespace common
}  // namespace samgraph

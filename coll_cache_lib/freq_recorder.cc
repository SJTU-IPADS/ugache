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

#include "common.h"
#include "run_config.h"
// #include "constant.h"
#include "device.h"
#include "cpu/mmap_cpu_device.h"
#include "freq_recorder.h"
#include "logging.h"
#include <cstring>
#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include "timer.h"

namespace coll_cache_lib {
namespace common {

FreqRecorder::FreqRecorder(size_t num_nodes, int local_id) {
  Timer t_init;
  _num_nodes = num_nodes;
  CHECK(num_nodes < std::numeric_limits<IdType>::max() / 2);
  size_t nbytes = _num_nodes * (std::stoi(GetEnvStrong("COLL_NUM_REPLICA")) + 1) * sizeof(Id64Type);
  int fd = cpu::MmapCPUDevice::CreateShm(nbytes, Constant::kCollCacheFreqRecorderShmName);
  global_freq_table_ptr = (Id64Type*)cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, fd);
  freq_table = global_freq_table_ptr + local_id * num_nodes;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
    *nid_ptr = i;
    *(nid_ptr + 1) = 0;
  }
  _cpu_device_holder = cpu::CPUDevice::Global();
  // Profiler::Get().LogInit(kLogInitL3PresampleInit, t_init.Passed());
}

FreqRecorder::~FreqRecorder() {
  // Device::Get(CPU())->FreeDataSpace(CPU(), freq_table);
}

void FreqRecorder::Record(const Id64Type* input, size_t num_inputs){
  auto cpu_device = Device::Get(CPU());
  Timer t2;
  #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_inputs; i++) {
    CHECK(input[i] < _num_nodes);
    auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input[i]]);
    *(freq_ptr+1) += 1;
  }
  double count_time = t2.Passed();
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleSample, sample_time);
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleCopy, copy_time);
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleCount, count_time);
  // LOG(ERROR) << "presample spend "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleSample) << " on sample, "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCopy) << " on copy, "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCount) << " on count";
}

void FreqRecorder::Record(const IdType* input, size_t num_inputs){
  auto cpu_device = Device::Get(CPU());
  Timer t2;
  #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < num_inputs; i++) {
    CHECK(input[i] < _num_nodes);
    auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input[i]]);
    *(freq_ptr+1) += 1;
  }
  double count_time = t2.Passed();
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleSample, sample_time);
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleCopy, copy_time);
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleCount, count_time);
  // LOG(ERROR) << "presample spend "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleSample) << " on sample, "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCopy) << " on copy, "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCount) << " on count";
}

void FreqRecorder::Sort() {
  Timer ts;
  Id64Type* aggregate_freq_table = global_freq_table_ptr + RunConfig::num_device * _num_nodes;
#ifdef __linux__
  __gnu_parallel::sort(aggregate_freq_table, &aggregate_freq_table[_num_nodes],
                       std::greater<Id64Type>());
#else
  std::sort(freq_table, &freq_table[_num_nodes],
            std::greater<Id64Type>());
#endif
  double sort_time = ts.Passed();
  // Profiler::Get().LogInit(kLogInitL3PresampleSort, sort_time);
  // LOG(ERROR) << "presample spend " << sort_time << " on sort freq.";
  // Profiler::Get().ResetStepEpoch();
  // Profiler::Get().LogInit(kLogInitL3PresampleReset, t_reset.Passed());

}

TensorPtr FreqRecorder::GetFreq() {
  auto ranking_freq = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  auto ranking_freq_ptr = static_cast<IdType*>(ranking_freq->MutableData());
  GetFreq(ranking_freq_ptr);
  return ranking_freq;
}

void FreqRecorder::GetFreq(IdType* ranking_freq_ptr) {
  Id64Type* aggregate_freq_table = global_freq_table_ptr + RunConfig::num_device * _num_nodes;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
    ranking_freq_ptr[i] = *(nid_ptr + 1);
  }
}
TensorPtr FreqRecorder::GetRankNode() {
  auto ranking_nodes = Tensor::Empty(DataType::kI32, {_num_nodes}, CPU(), "");
  GetRankNode(ranking_nodes);
  return ranking_nodes;
}
void FreqRecorder::GetRankNode(TensorPtr& ranking_nodes) {
  auto ranking_nodes_ptr = ranking_nodes->Ptr<IdType>();
  GetRankNode(ranking_nodes_ptr);
}

void FreqRecorder::GetRankNode(IdType* ranking_nodes_ptr) {
  Timer t_prepare_rank;
  Id64Type* aggregate_freq_table = global_freq_table_ptr + RunConfig::num_device * _num_nodes;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
    ranking_nodes_ptr[i] = *(nid_ptr);
  }
  // Profiler::Get().LogInit(kLogInitL3PresampleGetRank, t_prepare_rank.Passed());
}

// void FreqRecorder::Combine(FreqRecorder *other) {
//   #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for (size_t i = 0; i < _num_nodes; i++) {
//     auto local_nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
//     auto other_nid_ptr = reinterpret_cast<IdType*>(&other->freq_table[i]);
//     *(local_nid_ptr + 1) += *(other_nid_ptr + 1);
//   }
// }

// void FreqRecorder::Combine(int other_local_id) {
//   Id64Type* other_freq_table = global_freq_table_ptr + other_local_id * _num_nodes;
//   #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
//   for (size_t i = 0; i < _num_nodes; i++) {
//     auto local_nid_ptr = reinterpret_cast<IdType*>(&freq_table[i]);
//     auto other_nid_ptr = reinterpret_cast<IdType*>(&other_freq_table[i]);
//     *(local_nid_ptr + 1) += *(other_nid_ptr + 1);
//   }
// }
void FreqRecorder::Combine() {
  Id64Type* aggregate_freq_table = global_freq_table_ptr + RunConfig::num_device * _num_nodes;
  #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
    *nid_ptr = i;
    *(nid_ptr + 1) = 0;
  }

  for (int other_local_id = 0; other_local_id < RunConfig::num_device; other_local_id++) {
    Id64Type* other_freq_table = global_freq_table_ptr + other_local_id * _num_nodes;
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < _num_nodes; i++) {
      auto agg_nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
      auto other_nid_ptr = reinterpret_cast<IdType*>(&other_freq_table[i]);
      *(agg_nid_ptr + 1) += *(other_nid_ptr + 1);
    }
  }
}
}
}

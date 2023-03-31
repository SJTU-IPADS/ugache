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

#ifdef DEAD_CODE
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
  // auto cpu_device = Device::Get(CPU());
  // Timer t2;
  // #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  // #pragma omp parallel for num_threads(10)
  for (size_t i = 0; i < num_inputs; i++) {
    // CHECK(input[i] < _num_nodes);
    auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input[i]]);
    *(freq_ptr+1) += 1;
  }
  // double count_time = t2.Passed();
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleSample, sample_time);
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleCopy, copy_time);
  // Profiler::Get().LogInitAdd(kLogInitL3PresampleCount, count_time);
  // LOG(ERROR) << "presample spend "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleSample) << " on sample, "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCopy) << " on copy, "
  //            << Profiler::Get().GetLogInitValue(kLogInitL3PresampleCount) << " on count";
}

void FreqRecorder::Record(const IdType* input, size_t num_inputs){
  // auto cpu_device = Device::Get(CPU());
  // Timer t2;
  // #pragma omp parallel for num_threads(10)
  for (size_t i = 0; i < num_inputs; i++) {
    // CHECK(input[i] < _num_nodes) << input[i] << " greater than " << _num_nodes;
    auto freq_ptr = reinterpret_cast<IdType*>(&freq_table[input[i]]);
    *(freq_ptr+1) += 1;
  }
  // double count_time = t2.Passed();
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
                       std::greater<Id64Type>(), __gnu_parallel::default_parallel_tag(RunConfig::solver_omp_thread_num));
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
#pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
    ranking_freq_ptr[i] = *(nid_ptr + 1);
  }
  LOG(ERROR) << "top 5 freq is " 
             << ranking_freq_ptr[0] << ","
             << ranking_freq_ptr[1] << ","
             << ranking_freq_ptr[2] << ","
             << ranking_freq_ptr[3] << ","
             << ranking_freq_ptr[4];
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
#pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num)
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
  LOG(ERROR) << "Combining with " << RunConfig::solver_omp_thread_num << " threads";
  Id64Type* aggregate_freq_table = global_freq_table_ptr + RunConfig::num_device * _num_nodes;
  #pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num)
  for (size_t i = 0; i < _num_nodes; i++) {
    auto nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
    *nid_ptr = i;
    *(nid_ptr + 1) = 0;
  }

  for (int other_local_id = 0; other_local_id < RunConfig::num_device; other_local_id++) {
    Id64Type* other_freq_table = global_freq_table_ptr + other_local_id * _num_nodes;
    Id64Type total_freq = 0;
    #pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num) reduction(+ : total_freq)
    for (size_t i = 0; i < _num_nodes; i++) {
      auto agg_nid_ptr = reinterpret_cast<IdType*>(&aggregate_freq_table[i]);
      auto other_nid_ptr = reinterpret_cast<IdType*>(&other_freq_table[i]);
      *(agg_nid_ptr + 1) += *(other_nid_ptr + 1);
      total_freq += *(other_nid_ptr + 1);
    }
    LOG(ERROR) << "combined one location with " << total_freq;
  }
}
#endif

namespace {
const std::string FreqRecorderShmV2 = std::string("coll_cache_freq_recorder_shm_v2_") + GetEnvStrong("USER");
const size_t BufMaxLen = 100'000'000;
size_t GlobalBuxMaxLen = 0;
};

FreqRecorder::FreqRecorder(size_t num_nodes, int local_id)
  //  : local_freq_buf_mutex() 
{
  Timer t_init;
  _num_nodes = num_nodes;
  CHECK(num_nodes < std::numeric_limits<IdType>::max() / 2);
  _cpu_device_holder = cpu::CPUDevice::Global();
  sem_init(&local_freq_buf_sem, 0, 1);
  {
    GlobalBuxMaxLen = BufMaxLen * std::stoi(GetEnvStrong("COLL_NUM_REPLICA"));
    size_t nbytes = GlobalBuxMaxLen * sizeof(FreqEntry) + sizeof(DupFreqBuf);
    int fd = cpu::MmapCPUDevice::CreateShm(nbytes, FreqRecorderShmV2 + "_v2");
    global_dup_freq_buf = new (cpu::MmapCPUDevice::MapFd(MMAP(MMAP_RW_DEVICE), nbytes, fd)) DupFreqBuf;
    local_freq_buf = new MapFreqBuf[RunConfig::solver_omp_thread_num_per_gpu];
    local_freq_buf_alter = new MapFreqBuf[RunConfig::solver_omp_thread_num_per_gpu];
    global_cont_freq_buf = new ContFreqBuf;
  }
}

template<typename KeyT>
void FreqRecorder::Record(const KeyT* input, size_t num_inputs){
  // std::lock_guard<std::mutex> guard(this->local_freq_buf_mutex);
  sem_wait(&local_freq_buf_sem);
  if (RunConfig::solver_omp_thread_num_per_gpu == 1) {
    for (size_t i = 0; i < num_inputs; i++) {
      // CHECK(input[i] < _num_nodes) << input[i] << " greater than " << _num_nodes;
      // CHECK(local_freq_buf->mapping.size() < BufMaxLen);
      local_freq_buf->add(input[i], 1);
    }
  } else {
    #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num_per_gpu)
    {
      MapFreqBuf * thread_local_buf = local_freq_buf + omp_get_thread_num();
      for (size_t i = 0; i < num_inputs; i++) {
        if (input[i] % RunConfig::solver_omp_thread_num_per_gpu != omp_get_thread_num()) continue;
        thread_local_buf->add(input[i], 1);
      }
    }
  }
  // for (size_t i = 0; i < num_inputs; i++) {
  //   // CHECK(input[i] < _num_nodes) << input[i] << " greater than " << _num_nodes;
  //   // CHECK(local_freq_buf->mapping.size() < BufMaxLen);
  //   local_freq_buf->add(input[i], 1);
  // }
  sem_post(&local_freq_buf_sem);
}
template void FreqRecorder::Record<Id64Type>(const Id64Type* input, size_t num_inputs);
template void FreqRecorder::Record<IdType>(const IdType* input, size_t num_inputs);


void FreqRecorder::LocalCombineToShared(){
  // fixme: atomic alter freq_table
  auto local_buf = this->AlterLocalBuf();
  for (int thd_idx = 0; thd_idx < RunConfig::solver_omp_thread_num_per_gpu; thd_idx++) {
    global_dup_freq_buf->bulk_append(local_buf + thd_idx);
    local_buf[thd_idx].mapping.clear();
  }
}
void FreqRecorder::GlobalCombine(){
  LOG(ERROR) << "freq recorder global combining";
  this->global_cont_freq_buf->mapping.clear();
  global_dup_freq_buf->reduce(this->global_cont_freq_buf);
  LOG(ERROR) << "freq recorder global combine from " << global_dup_freq_buf->cur_len << " to " << global_cont_freq_buf->mapping.size();
  global_dup_freq_buf->cur_len = 0;
}

void FreqRecorder::Sort() {
  uint64_t* to_sort = (uint64_t*)global_cont_freq_buf->buf;
#ifdef __linux__
  __gnu_parallel::sort(to_sort, &to_sort[global_cont_freq_buf->mapping.size()],
                       std::greater<uint64_t>());
#else
  std::sort(to_sort, &to_sort[global_cont_freq_buf->mapping.size()],
            std::greater<uint64_t>());
#endif
  #pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num)
  for (size_t rnk = 0; rnk < global_cont_freq_buf->mapping.size(); rnk++) {
    global_cont_freq_buf->mapping[global_cont_freq_buf->buf[rnk].key] = rnk;
  }
  LOG(ERROR) << "top 5 freq is " 
             << global_cont_freq_buf->buf[0].cnt << ","
             << global_cont_freq_buf->buf[1].cnt << ","
             << global_cont_freq_buf->buf[2].cnt << ","
             << global_cont_freq_buf->buf[3].cnt << ","
             << global_cont_freq_buf->buf[4].cnt;
}
void ContFreqBuf::GetLegacyFreqRank(LegacyFreqBuf* output, size_t total_num_node) {
  output->rank_vec.resize(total_num_node);
  output->freq_vec.resize(total_num_node, 0);
  constexpr size_t local_buf_len = 10000;
  size_t cur_global_len = this->mapping.size();
  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    #pragma omp for
    for (size_t rnk = 0; rnk < this->mapping.size(); rnk++) {
      output->rank_vec[rnk] = this->buf[rnk].key;
      output->freq_vec[rnk] = this->buf[rnk].cnt;
    }
    std::vector<IdType> local_no_freq_nids;
    local_no_freq_nids.reserve(10000);
    #pragma omp for
    for (size_t nid = 0; nid < total_num_node; nid++) {
      if (this->mapping.find(nid) != this->mapping.end()) continue;
      if (local_no_freq_nids.size() == local_buf_len) {
        size_t to_global_off;
        #pragma omp critical
        {
          to_global_off = cur_global_len;
          cur_global_len += local_no_freq_nids.size();
        }
        memcpy(output->rank_vec.data() + to_global_off, local_no_freq_nids.data(),
               local_no_freq_nids.size() * sizeof(IdType));
        local_no_freq_nids.clear();
      }
      local_no_freq_nids.push_back(nid);
    }

    size_t to_global_off;
    #pragma omp critical
    {
      to_global_off = cur_global_len;
      cur_global_len += local_no_freq_nids.size();
    }
    memcpy(output->rank_vec.data() + to_global_off, local_no_freq_nids.data(),
           local_no_freq_nids.size() * sizeof(IdType));
    local_no_freq_nids.clear();
  }
  CHECK(cur_global_len == total_num_node) << cur_global_len << " != " << total_num_node;
}
void DupFreqBuf::bulk_append(const MapFreqBuf* input) {
  LOG(ERROR) << "bulk append " << input->mapping.size() << " into " << cur_len;
  CHECK(input->mapping.size() <= BufMaxLen) << input->mapping.size() << ">" << BufMaxLen;
  CHECK(cur_len + input->mapping.size() <= GlobalBuxMaxLen) << cur_len << "+" << input->mapping.size() << ">" << GlobalBuxMaxLen;
  size_t len = cur_len;
  IdType total_freq = 0;
  for (auto& iter : input->mapping) {
    buf[len++] = {iter.first, iter.second};
    total_freq += iter.second;
  }
  cur_len = len;
  LOG(ERROR) << "bulk append " << total_freq;
}
MapFreqBuf* FreqRecorder::AlterLocalBuf() {
  // std::lock_guard<std::mutex> guard(this->local_freq_buf_mutex);
  sem_wait(&local_freq_buf_sem);
  std::swap(local_freq_buf, local_freq_buf_alter);
  auto ret = local_freq_buf_alter;
  sem_post(&local_freq_buf_sem);
  return local_freq_buf_alter;
}
}
}

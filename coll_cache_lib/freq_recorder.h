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

#pragma once

#include <memory>
#include "cpu/cpu_device.h"
#include "common.h"
#include <cstring>
#include <semaphore.h>
#include "robin_hood.h"
// #include "constant.h"
namespace coll_cache_lib {
namespace common {

using CntT = IdType;
using OffT = uint32_t;
struct FreqEntry {
  IdType key;
  CntT cnt;
};

struct LegacyFreqBuf {
  std::vector<uint32_t> rank_vec, freq_vec;
};

struct ContFreqBuf {
  robin_hood::unordered_map<IdType, OffT> mapping;
  FreqEntry * buf;
  void add(const IdType & key, const CntT & cnt) {
    auto iter = mapping.find(key);
    if (iter == mapping.end()) {
      IdType offset = mapping.size();
      iter = mapping.insert({key, offset}).first;
      buf[offset].key = key;
      buf[offset].cnt = cnt;
    } else {
      IdType offset = iter->second;
      buf[offset].cnt += cnt;
    }
  }
  CntT get(const IdType & key) {
    auto iter = mapping.find(key);
    if (iter == mapping.end()) {
      return 0;
    } else {
      return buf[iter->second].cnt;
    }
  }

  void GetLegacyFreqRank(LegacyFreqBuf* output, size_t total_num_node);
};


struct MapFreqBuf {
  robin_hood::unordered_map<IdType, CntT> mapping;
  void add(const IdType & key, const CntT & cnt) {
    auto iter = mapping.find(key);
    if (iter == mapping.end()) {
      mapping.insert({key, cnt});
    } else {
      iter->second += cnt;
    }
  }
  uint32_t get(const IdType & key) {
    auto iter = mapping.find(key);
    if (iter == mapping.end()) {
      return 0;
    } else {
      return iter->second;
    }
  }
};
struct DupFreqBuf {
  volatile size_t cur_len = 0;
  FreqEntry buf[0];
  void append(const IdType & key, const CntT & cnt) {
    buf[cur_len++] = {key, cnt};
  }
  void bulk_append(const MapFreqBuf* input);
  void reduce(ContFreqBuf* output) {
    output->buf = buf;
    output->mapping.clear();
    const size_t len = cur_len;
    for (size_t i = 0; i < len; i++) {
      output->add(buf[i].key, buf[i].cnt);
    }
  }
};

#ifdef DEAD_CODE
class FreqRecorder {
 public:
  FreqRecorder(size_t num_nodes, int local_id);
  ~FreqRecorder();
  void Record(const IdType* inputs, size_t num_input);
  void Record(const Id64Type* inputs, size_t num_input);
  void Sort();
  TensorPtr GetFreq();
  void GetFreq(IdType*);
  TensorPtr GetRankNode();
  void GetRankNode(TensorPtr &);
  void GetRankNode(IdType *);
  // void Combine(FreqRecorder* other);
  // void Combine(int other_local_id);
  void Combine();
  inline size_t NumNodes() const { return _num_nodes; }
 private:
  Id64Type * freq_table;
  Id64Type * global_freq_table_ptr;
  size_t _num_nodes;
  std::shared_ptr<cpu::CPUDevice> _cpu_device_holder = nullptr;
};
#endif

class FreqRecorder {
 public:
  FreqRecorder(size_t num_nodes, int local_id);
  template<typename KeyT>
  void Record(const KeyT* inputs, size_t num_input);
  void Sort();
  // void GetLegacyFreqRank(LegacyFreqBuf*);
  inline size_t NumNodes() const { return _num_nodes; }
  void LocalCombineToShared();
  void GlobalCombine();
  MapFreqBuf* AlterLocalBuf();
  ContFreqBuf* global_cont_freq_buf;
 private:
  size_t _num_nodes;
  std::shared_ptr<cpu::CPUDevice> _cpu_device_holder = nullptr;
  MapFreqBuf *local_freq_buf, *local_freq_buf_alter;
  DupFreqBuf* global_dup_freq_buf;
  // std::mutex local_freq_buf_mutex;
  sem_t local_freq_buf_sem;
};

}
}

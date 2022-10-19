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
// #include "constant.h"
namespace coll_cache_lib {
namespace common {

class FreqRecorder {
 public:
  FreqRecorder(size_t num_nodes);
  ~FreqRecorder();
  void Record(const IdType* inputs, size_t num_input);
  void Record(const Id64Type* inputs, size_t num_input);
  void Sort();
  TensorPtr GetFreq();
  void GetFreq(IdType*);
  TensorPtr GetRankNode();
  void GetRankNode(TensorPtr &);
  void GetRankNode(IdType *);
  void Combine(FreqRecorder* other);
 private:
  Id64Type * freq_table;
  size_t _num_nodes;
  std::shared_ptr<cpu::CPUDevice> _cpu_device_holder = nullptr;
};

}
}

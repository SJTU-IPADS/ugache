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

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

#include "../common.h"
#include "../constant.h"
#include "../logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class SimpleHashTable;

using ValType = IdType;
constexpr IdType kEmptyPos = 0xffffffff;

class DeviceSimpleHashTable {
 public:
  // 1| 111...111 -> Default state, this bucket is not used yet
  //          insert may use this, search must stop at this
  // 1| xxx...xxx -> Invalid bucket,
  //          insert may use this, but search must proceed beyond this
  // 0| xxx...xxx -> bucket inuse,
  //          insert cannot use this, but search must proceed beyond this
  struct alignas(unsigned long long) BucketO2N {
    // don't change the position of version and key
    //   which used for efficient insert operation
    IdType state_key;
    ValType val;
  };

  typedef const BucketO2N *ConstIterator;

  DeviceSimpleHashTable(const DeviceSimpleHashTable &other) = default;
  DeviceSimpleHashTable &operator=(const DeviceSimpleHashTable &other) =
      default;

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while ((_o2n_table[pos].state_key & 0x7fffffff) != id) {
      if (_o2n_table[pos].state_key == Constant::kEmptyKey) {
        return kEmptyPos;
      }
      pos = HashO2N(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);
    if (_o2n_table[pos].state_key & 0x80000000) return kEmptyPos;
    return pos;
#else
    return id;
#endif
  }

  inline __device__ ConstIterator SearchO2N(const IdType id) const {
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while ((_o2n_table[pos].state_key & 0x7fffffff) != id) {
      if (_o2n_table[pos].state_key == Constant::kEmptyKey) {
        return nullptr;
      }
      pos = HashO2N(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);
    if (_o2n_table[pos].state_key & 0x80000000) return nullptr;
    return &_o2n_table[pos];
  }

 protected:
  const BucketO2N *_o2n_table;
  const size_t _o2n_size;
  IdType _version;

  explicit DeviceSimpleHashTable(const BucketO2N *const o2n_table,
                                  const size_t o2n_size,
                                  const IdType _);


  inline __device__ IdType HashO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    return id % _o2n_size;
#else
    return id;
#endif
  }

  friend class SimpleHashTable;
};

class SimpleHashTable {
 public:
  static constexpr size_t kDefaultScale = 2;

  using BucketO2N = typename DeviceSimpleHashTable::BucketO2N;

  SimpleHashTable(const size_t size, Context ctx,
                   StreamHandle stream, const size_t scale = kDefaultScale);

  ~SimpleHashTable();

  // Disable copying
  SimpleHashTable(const SimpleHashTable &other) = delete;
  SimpleHashTable &operator=(const SimpleHashTable &other) = delete;

  void Reset(StreamHandle stream);

  void FillWithUnique(const IdType *const input, 
                      const ValType *const vals,
                      const size_t num_input,
                      StreamHandle stream);
  void EvictWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);
  void LookupIfExist(const IdType* const input, const size_t num_input, IdType * pos, StreamHandle stream);
  void LookupVal(const IdType* const input, const size_t num_input, ValType * vals, StreamHandle stream);
  void CountEntries(StreamHandle stream);

  size_t NumItems() const { return _num_items; }

  DeviceSimpleHashTable DeviceHandle() const;

 private:
  Context _ctx;

  BucketO2N *_o2n_table;
  size_t _o2n_size;

  IdType _version;
  IdType _num_items;
};


void check_cuda_array(IdType* array, IdType cmp, IdType num_items, bool exp, StreamHandle stream);

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

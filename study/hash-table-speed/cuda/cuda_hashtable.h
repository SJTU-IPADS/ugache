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

#ifndef SAMGRAPH_CUDA_HASHTABLE_H
#define SAMGRAPH_CUDA_HASHTABLE_H

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

#include "../common.h"
#include "../constant.h"
#include "../logging.h"

namespace samgraph {
namespace common {
namespace cuda {

class OrderedHashTable;
enum EntryStatus {
  kUnused = 0,
  kOccupied,
  kInvalid,
};

using ValType = IdType;
constexpr IdType kEmptyPos = 0xffffffff;

class DeviceOrderedHashTable {
 public:
  struct alignas(unsigned long long) BucketO2N {
    // don't change the position of version and key
    //   which used for efficient insert operation
    // IdType version;
    IdType key;
    IdType state; // 0 for unused, 1 for inserted, 2 for inserted but evicted out
    // IdType local;
    ValType val;
    // IdType index;

  };

  typedef const BucketO2N *ConstIterator;

  DeviceOrderedHashTable(const DeviceOrderedHashTable &other) = default;
  DeviceOrderedHashTable &operator=(const DeviceOrderedHashTable &other) =
      default;

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    IdType pos = HashO2N(id);

    // linearly scan for matching entry
    IdType delta = 1;
    while (_o2n_table[pos].key != id) {
      if (_o2n_table[pos].key == Constant::kEmptyKey || _o2n_table[pos].state == kUnused) {
        return kEmptyPos;
      }
      pos = HashO2N(pos + delta);
      delta += 1;
    }
    assert(pos < _o2n_size);
    if (_o2n_table[pos].state == kInvalid) return kEmptyPos;
    return pos;
#else
    return id;
#endif
  }

  inline __device__ ConstIterator SearchO2N(const IdType id) const {
    const IdType pos = SearchForPositionO2N(id);
    return &_o2n_table[pos];
  }

 protected:
  const BucketO2N *_o2n_table;
  const size_t _o2n_size;
  IdType _version;

  explicit DeviceOrderedHashTable(const BucketO2N *const o2n_table,
                                  const size_t o2n_size,
                                  const IdType _);


  inline __device__ IdType HashO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    return id % _o2n_size;
#else
    return id;
#endif
  }

  friend class OrderedHashTable;
};

class OrderedHashTable {
 public:
  static constexpr size_t kDefaultScale = 2;

  using BucketO2N = typename DeviceOrderedHashTable::BucketO2N;

  OrderedHashTable(const size_t size, Context ctx,
                   StreamHandle stream, const size_t scale = kDefaultScale);

  ~OrderedHashTable();

  // Disable copying
  OrderedHashTable(const OrderedHashTable &other) = delete;
  OrderedHashTable &operator=(const OrderedHashTable &other) = delete;

  void Reset(StreamHandle stream);

  void FillWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);
  void EvictWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);
  void LookupIfExist(const IdType* const input, const size_t num_input, IdType * pos, StreamHandle stream);
  void LookupVal(const IdType* const input, const size_t num_input, ValType * vals, StreamHandle stream);
  void CountEntries(StreamHandle stream);

  size_t NumItems() const { return _num_items; }

  DeviceOrderedHashTable DeviceHandle() const;

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

#endif  // SAMGRAPH_CUDA_HASHTABLE_H

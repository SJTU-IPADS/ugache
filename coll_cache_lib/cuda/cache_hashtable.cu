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

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cub/cub.cuh>

#include "../common.h"
#include "../device.h"
#include "../logging.h"
#include "../timer.h"
#include "cache_hashtable.cuh"
#include "cuda_utils.h"

namespace coll_cache_lib {
namespace common {

class MutableDeviceSimpleHashTable : public DeviceSimpleHashTable {
  static inline __host__ __device__ uint32_t high32(uint64_t val) {
    constexpr uint64_t kI32Mask = 0xffffffff;
    return (val >> 32) & kI32Mask;
  }
  static inline __host__ __device__ uint32_t low32(uint64_t val) {
    constexpr uint64_t kI32Mask = 0xffffffff;
    return val & kI32Mask;
  }
  static inline __host__ __device__ uint64_t uint_to_ll(uint32_t low, uint32_t high) {
    return (((uint64_t)high) << 32) + low;
  }
 public:
  typedef typename DeviceSimpleHashTable::BucketO2N *IteratorO2N;

  explicit MutableDeviceSimpleHashTable(SimpleHashTable *const host_table)
      : DeviceSimpleHashTable(host_table->DeviceHandle()) {}

  inline __device__ IteratorO2N SearchO2N(const IdType id) {
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
    return const_cast<IteratorO2N>(_o2n_table + pos);
  }

  static constexpr IdType MAX_DELTA = 100;
  enum InsertStatus {
    kConflict = 0,
    kFirstSuccess,
    kDupSuccess,
  };
  inline __device__ InsertStatus AttemptInsertAtO2N(const IdType pos, const IdType id,
                                            const ValType val) {
    auto iter = GetMutableO2N(pos);
#ifndef SXN_NAIVE_HASHMAP
    // FIXME: only support sizeof(IdType) == 4
    static_assert(sizeof(IdType) == 4, "");

    IdType old = iter->state_key;
    IdType old_state = old & 0x80000000;
    IdType old_key = old & 0x7fffffff;
    if (old_state == 0) {
      if (old_key == id) assert(false);
      return kConflict;
    }
    IdType new_val = id;
    IdType ret_val = atomicCAS(&iter->state_key, old, new_val);
    if (ret_val == old) {
      iter->val = val;
      return kFirstSuccess;
    }
    IdType ret_key = ret_val & 0x7fffffff;
    if (ret_key == id) assert(false);
    return kConflict;
#else
    IdType old_version = iter->version;
    if (old_version == version) return kDupSuccess;
    if (atomicCAS(&(iter->version), old_version, version) == old_version) {
      iter->key = id;
      iter->index = index;
      iter->local = Constant::kEmptyKey;
      return kFirstSuccess;
    }
    return kDupSuccess;
#endif
  }

  /** Return corresponding bucket on first insertion.
   *  Duplicate attemps return nullptr
   */
  inline __device__ IteratorO2N InsertO2N(const IdType id, const ValType val) {
#ifndef SXN_NAIVE_HASHMAP
    IdType pos = HashO2N(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    InsertStatus ret;
    while ((ret = AttemptInsertAtO2N(pos, id, val)) == kConflict) {
      pos = HashO2N(pos + delta);
      delta += 1;
    }
#else
    IdType pos = id;
    ret = AttemptInsertAtO2N(pos, id, index, version);
    assert(ret != kConflict);
#endif
    return (ret == kFirstSuccess) ? GetMutableO2N(pos) : nullptr;
  }

  inline __device__ IdType IterO2NToPos(const IteratorO2N iter) {
    return iter - _o2n_table;
  }

//  private:
  inline __device__ IteratorO2N GetMutableO2N(const IdType pos) {
    assert(pos < this->_o2n_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of SimpleHashTable, making this
    // a safe cast to perform.
    return const_cast<IteratorO2N>(this->_o2n_table + pos);
  }

};

class MutableDeviceFlatHashTable : public DeviceFlatHashTable {
  static inline __host__ __device__ uint32_t high32(uint64_t val) {
    constexpr uint64_t kI32Mask = 0xffffffff;
    return (val >> 32) & kI32Mask;
  }
  static inline __host__ __device__ uint32_t low32(uint64_t val) {
    constexpr uint64_t kI32Mask = 0xffffffff;
    return val & kI32Mask;
  }
  static inline __host__ __device__ uint64_t uint_to_ll(uint32_t low, uint32_t high) {
    return (((uint64_t)high) << 32) + low;
  }
 public:
  typedef typename DeviceFlatHashTable::BucketFlat *IteratorO2N;

  explicit MutableDeviceFlatHashTable(FlatHashTable *const host_table)
      : DeviceFlatHashTable(host_table->DeviceHandle()) {}

  inline __device__ IteratorO2N SearchO2N(const IdType id) {
    IdType pos = HashO2N(id);

    assert(pos < _flat_size);
    if (_flat_table[pos].val.data == kEmptyPos) {
      return nullptr;
    }
    return const_cast<IteratorO2N>(_flat_table + pos);
  }

  enum InsertStatus {
    kConflict = 0,
    kFirstSuccess,
    kDupSuccess,
  };
  inline __device__ InsertStatus AttemptInsertAtO2N(const IdType pos, const IdType id,
                                            const ValType val) {
    auto iter = GetMutableO2N(pos);
    //fixme: replace with normal store
    if (atomicCAS(&(iter->val.data), kEmptyPos, val.data) == kEmptyPos) {
      return kFirstSuccess;
    }
    assert(false);
  }

  /** Return corresponding bucket on first insertion.
   *  Duplicate attemps return nullptr
   */
  inline __device__ IteratorO2N InsertO2N(const IdType id, const ValType val) {
    InsertStatus ret;
    IdType pos = id;
    ret = AttemptInsertAtO2N(pos, id, val);
    assert(ret == kFirstSuccess);
    return GetMutableO2N(pos);
  }

  inline __device__ IdType IterO2NToPos(const IteratorO2N iter) {
    return iter - _flat_table;
  }

//  private:
  inline __device__ IteratorO2N GetMutableO2N(const IdType pos) {
    assert(pos < this->_flat_size);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of FlatHashTable, making this
    // a safe cast to perform.
    return const_cast<IteratorO2N>(this->_flat_table + pos);
  }
};

/**
 * Calculate the number of buckets in the hashtable. To guarantee we can
 * fill the hashtable in the worst case, we must use a number of buckets which
 * is a power of two.
 * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
 */
size_t TableSize(const size_t num, const size_t scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename MutableDeviceHashTable_T>
__global__ void generate_hashmap_unique(const IdType *const items,
                                        const ValType* const vals,
                                        const size_t num_items,
                                        MutableDeviceHashTable_T table) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N = typename MutableDeviceHashTable_T::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.InsertO2N(items[index], vals[index]);
      // since we are only inserting unique items, we know their local id
      // will be equal to their index
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename ValMaker, typename MutableDeviceHashTable_T>
__global__ void insert_hashmap_unique(const IdType *const items,
                                        ValMaker val_maker,
                                        const size_t num_items,
                                        MutableDeviceHashTable_T table) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N = typename MutableDeviceHashTable_T::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.InsertO2N(items[index], val_maker(index));
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void evict_hashmap_unique(const IdType *const items,
                                    const size_t num_items,
                                    MutableDeviceSimpleHashTable table) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N = typename MutableDeviceSimpleHashTable::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.SearchO2N(items[index]);
      bucket->state_key |= 0x80000000;
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void evict_hashmap_unique(const IdType *const items,
                                    const size_t num_items,
                                    MutableDeviceFlatHashTable table) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N = typename MutableDeviceFlatHashTable::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.SearchO2N(items[index]);
      bucket->val.data = kEmptyPos;
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename MutableDeviceHashTable_T>
__global__ void lookup_hashmap_ifexist(const IdType *const items,
                             const size_t num_items,
                             IdType* pos,
                             MutableDeviceHashTable_T table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      auto rst_pos = table.SearchForPositionO2N(items[index]);
      pos[index] = rst_pos;
    }
  }
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename MutableDeviceHashTable_T>
__global__ void lookup_val_hashmap(const IdType *const items,
                             const size_t num_items,
                             ValType* vals,
                             MutableDeviceHashTable_T table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      auto iter = table.SearchO2N(items[index]);
      vals[index] = iter->val;
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename DefValMaker, typename MutableDeviceHashTable_T>
__global__ void lookup_hashmap_with_def(const IdType *const items,
                             const size_t num_items,
                             ValType* vals,
                             DefValMaker def_val_maker,
                             MutableDeviceHashTable_T table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      auto iter = table.SearchO2N(items[index]);
      if (iter == nullptr) {
        vals[index] = def_val_maker(items[index]);
      } else {
        vals[index] = iter->val;
      }
    }
  }
}


// DeviceSimpleHashTable implementation
DeviceSimpleHashTable::DeviceSimpleHashTable(const BucketO2N *const o2n_table,
                                              MurmurHash3_32<IdType> hasher,
                                               const size_t o2n_size)
    : _o2n_table(o2n_table),
      _hasher(hasher),
      _o2n_size(o2n_size) {}

DeviceSimpleHashTable SimpleHashTable::DeviceHandle() const {
  return DeviceSimpleHashTable(_o2n_table,
      _hasher,
      _o2n_size);
}

// SimpleHashTable implementation
SimpleHashTable::SimpleHashTable(const size_t size, Context ctx,
                                   StreamHandle stream, const size_t scale)
    : _o2n_table(nullptr),
      _hasher(),
      max_efficient_size(size),
#ifndef SXN_NAIVE_HASHMAP
      _o2n_size(TableSize(size, scale)),
#else
      _o2n_size(size),
#endif
      _ctx(ctx) {
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  LOG(INFO) << "SimpleHashTable allocating " << ToReadableSize(sizeof(BucketO2N) * _o2n_size);
  _o2n_table = static_cast<BucketO2N *>(
      device->AllocDataSpace(_ctx, sizeof(BucketO2N) * _o2n_size));

  CUDA_CALL(cudaMemsetAsync(_o2n_table, (int)Constant::kEmptyKey,
                       sizeof(BucketO2N) * _o2n_size, cu_stream));
  device->StreamSync(_ctx, stream);
  LOG(INFO) << "cuda hashtable init with " << _o2n_size
            << " O2N table size";
}

// SimpleHashTable implementation
SimpleHashTable::SimpleHashTable(BucketO2N* table, const size_t size, Context ctx,
                                   StreamHandle stream, const size_t scale)
    : _o2n_table(table),
      _hasher(),
      max_efficient_size(size),
#ifndef SXN_NAIVE_HASHMAP
      _o2n_size(TableSize(size, scale)),
#else
      _o2n_size(size),
#endif
      _ctx(ctx) {
  // make sure we will at least as many buckets as items.
  LOG(INFO) << "cuda hashtable init with " << _o2n_size
            << " O2N table size";
}
// SimpleHashTable implementation
SimpleHashTable::SimpleHashTable(std::function<MemHandle(size_t)> allocator, const size_t size, Context ctx,
                                   StreamHandle stream, const size_t scale)
    : _o2n_table(nullptr),
      _hasher(),
      max_efficient_size(size),
#ifndef SXN_NAIVE_HASHMAP
      _o2n_size(TableSize(size, scale)),
#else
      _o2n_size(size),
#endif
      _ctx(ctx) {
  LOG(ERROR) << "create a hashtable with " << max_efficient_size << " possible elem, scale to " << _o2n_size;
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  LOG(ERROR) << "SimpleHashTable allocating " << ToReadableSize(sizeof(BucketO2N) * _o2n_size);
  _o2n_table_handle = (allocator(sizeof(BucketO2N) * _o2n_size));
  _o2n_table = _o2n_table_handle->ptr<BucketO2N>();

  CUDA_CALL(cudaMemsetAsync(_o2n_table, (int)Constant::kEmptyKey,
                       sizeof(BucketO2N) * _o2n_size, cu_stream));
  device->StreamSync(_ctx, stream);
  LOG(INFO) << "cuda hashtable init with " << _o2n_size
            << " O2N table size";
}

SimpleHashTable::~SimpleHashTable() {
  this->_o2n_table_handle = nullptr;
}

void SimpleHashTable::FillWithUnique(const IdType *const input,
                                      const ValType *const vals,
                                      const size_t num_input,
                                      StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, vals, num_input, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

template<typename ValMaker>
void SimpleHashTable::InsertUnique(const IdType *const input, ValMaker val_maker, const size_t num_input, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  insert_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, val_maker, num_input, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}
template void SimpleHashTable::InsertUnique<HashTableInsertHelper::SingleLoc>(const IdType *const input, HashTableInsertHelper::SingleLoc val_maker, const size_t num_input, StreamHandle stream);
template void SimpleHashTable::InsertUnique<HashTableInsertHelper::SingleLocSeqOff>(const IdType *const input, HashTableInsertHelper::SingleLocSeqOff val_maker, const size_t num_input, StreamHandle stream);

void SimpleHashTable::EvictWithUnique(const IdType *const input,
                                       const size_t num_input,
                                       StreamHandle stream) {
  if (num_input == 0) return;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  evict_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

void SimpleHashTable::LookupIfExist(const IdType *const input, const size_t num_input, IdType *pos, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  lookup_hashmap_ifexist<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, pos, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

template<typename DefValMaker>
void SimpleHashTable::LookupValWithDef(const IdType* const input, const size_t num_input, ValType * vals, DefValMaker default_val_maker, StreamHandle stream){
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  lookup_hashmap_with_def<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, vals, default_val_maker, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}
template
void SimpleHashTable::LookupValWithDef<CPUFallback>(const IdType* const input, const size_t num_input, ValType * vals, CPUFallback default_val_maker, StreamHandle stream);


template void SimpleHashTable::LookupValCustom<HashTableLookupHelper::EmbedVal>(const IdType* const input, const size_t num_input, HashTableLookupHelper::EmbedVal helper, StreamHandle stream);
template void SimpleHashTable::LookupValCustom<HashTableLookupHelper::OffsetOnly>(const IdType* const input, const size_t num_input, HashTableLookupHelper::OffsetOnly helper, StreamHandle stream);
template void SimpleHashTable::LookupValCustom<HashTableLookupHelper::SepLocOffset>(const IdType* const input, const size_t num_input, HashTableLookupHelper::SepLocOffset helper, StreamHandle stream);

// DeviceFlatHashTable implementation
DeviceFlatHashTable::DeviceFlatHashTable(const BucketFlat *const o2n_table,
                                               const size_t o2n_size)
    : _flat_table(o2n_table),
      _flat_size(o2n_size) {}

DeviceFlatHashTable FlatHashTable::DeviceHandle() const {
  return DeviceFlatHashTable(_flat_table,
      _flat_size);
}

// FlatHashTable implementation
FlatHashTable::FlatHashTable(const size_t size, Context ctx,
                                   StreamHandle stream)
    : _flat_table(nullptr),
      _flat_size(size),
      _ctx(ctx) {
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  LOG(INFO) << "FlatHashTable allocating " << ToReadableSize(sizeof(BucketFlat) * _flat_size);
  _flat_table = static_cast<BucketFlat *>(
      device->AllocDataSpace(_ctx, sizeof(BucketFlat) * _flat_size));

  CUDA_CALL(cudaMemsetAsync(_flat_table, (int)kEmptyPos,
                       sizeof(BucketFlat) * _flat_size, cu_stream));
  device->StreamSync(_ctx, stream);
  LOG(INFO) << "cuda hashtable init with " << _flat_size
            << " O2N table size";
}

// FlatHashTable implementation
FlatHashTable::FlatHashTable(BucketFlat* table, const size_t size, Context ctx,
                                   StreamHandle stream)
    : _flat_table(table),
      _flat_size(size),
      _ctx(ctx) {
  // make sure we will at least as many buckets as items.
  LOG(INFO) << "cuda hashtable init with " << _flat_size
            << " O2N table size";
}
// FlatHashTable implementation
FlatHashTable::FlatHashTable(std::function<MemHandle(size_t)> allocator, const size_t size, Context ctx,
                                   StreamHandle stream)
    : _flat_table(nullptr),
      _flat_size(size),
      _ctx(ctx) {
  LOG(ERROR) << "create a hashtable with " << _flat_size;
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  LOG(ERROR) << "FlatHashTable allocating " << ToReadableSize(sizeof(BucketFlat) * _flat_size);
  _flat_table_handle = (allocator(sizeof(BucketFlat) * _flat_size));
  _flat_table = _flat_table_handle->ptr<BucketFlat>();

  CUDA_CALL(cudaMemsetAsync(_flat_table, (int)kEmptyPos,
                       sizeof(BucketFlat) * _flat_size, cu_stream));
  device->StreamSync(_ctx, stream);
  LOG(INFO) << "cuda hashtable init with " << _flat_size
            << " O2N table size";
}

FlatHashTable::~FlatHashTable() {
  this->_flat_table_handle = nullptr;
}

void FlatHashTable::FillWithUnique(const IdType *const input,
                                      const ValType *const vals,
                                      const size_t num_input,
                                      StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceFlatHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, vals, num_input, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

template<typename ValMaker>
void FlatHashTable::InsertUnique(const IdType *const input, ValMaker val_maker, const size_t num_input, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceFlatHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  insert_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, val_maker, num_input, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}
template void FlatHashTable::InsertUnique<HashTableInsertHelper::SingleLoc>(const IdType *const input, HashTableInsertHelper::SingleLoc val_maker, const size_t num_input, StreamHandle stream);
template void FlatHashTable::InsertUnique<HashTableInsertHelper::SingleLocSeqOff>(const IdType *const input, HashTableInsertHelper::SingleLocSeqOff val_maker, const size_t num_input, StreamHandle stream);

void FlatHashTable::EvictWithUnique(const IdType *const input,
                                       const size_t num_input,
                                       StreamHandle stream) {
  if (num_input == 0) return;
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceFlatHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  evict_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

void FlatHashTable::LookupIfExist(const IdType *const input, const size_t num_input, IdType *pos, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceFlatHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  lookup_hashmap_ifexist<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, pos, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

template<typename DefValMaker>
void FlatHashTable::LookupValWithDef(const IdType* const input, const size_t num_input, ValType * vals, DefValMaker default_val_maker, StreamHandle stream){
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceFlatHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  lookup_hashmap_with_def<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, vals, default_val_maker, device_table);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}
template
void FlatHashTable::LookupValWithDef<CPUFallback>(const IdType* const input, const size_t num_input, ValType * vals, CPUFallback default_val_maker, StreamHandle stream);


template void FlatHashTable::LookupValCustom<HashTableLookupHelper::EmbedVal>(const IdType* const input, const size_t num_input, HashTableLookupHelper::EmbedVal helper, StreamHandle stream);
template void FlatHashTable::LookupValCustom<HashTableLookupHelper::OffsetOnly>(const IdType* const input, const size_t num_input, HashTableLookupHelper::OffsetOnly helper, StreamHandle stream);
template void FlatHashTable::LookupValCustom<HashTableLookupHelper::SepLocOffset>(const IdType* const input, const size_t num_input, HashTableLookupHelper::SepLocOffset helper, StreamHandle stream);

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void check_cuda_array_(const IdType* array, IdType cmp, IdType num_items, bool exp) {
  assert(BLOCK_SIZE == blockDim.x);
  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      assert((array[index] == cmp) == exp);
    }
  }
}

void check_cuda_array(const IdType* array, IdType cmp, IdType num_items, bool exp, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv<size_t>(num_items, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);
  auto cu_stream = static_cast<cudaStream_t>(stream);
  check_cuda_array_<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(array, cmp, num_items, exp);
}

template <typename OffsetT = ptrdiff_t>
struct cubEntryIs {
  // Required iterator traits
  typedef cubEntryIs                       self_type;              ///< My own type
  typedef OffsetT                          difference_type;        ///< Type to express the result of subtracting one iterator from another
  typedef IdType                           value_type;             ///< The type of the element the iterator can point to
  typedef IdType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
  typedef IdType                           reference;              ///< The type of a reference to an element the iterator can point to
  typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category
  SimpleHashTable::BucketO2N* array;
  IdType cmp;
  IdType mask;
  __host__ __device__ cubEntryIs(SimpleHashTable::BucketO2N* arr, IdType c, IdType m) : array(arr),cmp(c),mask(m) {}
  template <typename Distance>
  __host__ __device__ __forceinline__ IdType operator[](const Distance d) const {
  // __host__ __device__ __forceinline__ IdType operator[](const IdType d) {
    return ((array[d].state_key & mask) == cmp) ? 1 : 0;
  }
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    return self_type(array + n, cmp, mask);
  }
};

void SimpleHashTable::CountEntries(StreamHandle stream){
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes;
  cubEntryIs<> input_iter(this->_o2n_table, 0, 0x80000000);
  IdType * d_out;
  CUDA_CALL(cudaMalloc(&d_out, sizeof(IdType)));
  // auto out = Tensor::Empty(kI32, {1}, GPU(0), "");
  auto cu_stream = static_cast<cudaStream_t>(stream);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, d_out, this->_o2n_size, cu_stream);
  CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  auto get_out = [d_out, stream, this](){
    IdType cpu_out;
    Device::Get(_ctx)->CopyDataFromTo(d_out, 0, &cpu_out, 0, sizeof(IdType), _ctx, CPU(), stream);
    Device::Get(_ctx)->StreamSync(_ctx, stream);
    return cpu_out;
  };

  input_iter = cubEntryIs<>(this->_o2n_table, 0, 0x80000000);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, d_out, this->_o2n_size, cu_stream);
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  LOG(ERROR) << "Occupied " << get_out();

  input_iter = cubEntryIs<>(this->_o2n_table, 0x80000000, 0x80000000);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, d_out, this->_o2n_size, cu_stream);
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  IdType state1 = get_out();

  input_iter = cubEntryIs<>(this->_o2n_table, 0xffffffff, 0xffffffff);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, d_out, this->_o2n_size, cu_stream);
  Device::Get(_ctx)->StreamSync(_ctx, stream);
  IdType default_state = get_out();
  LOG(ERROR) << "Invalid " << state1 - default_state;
  LOG(ERROR) << "Default " << default_state;
  CUDA_CALL(cudaFree(d_out));
}


}  // namespace common
}  // namespace samgraph

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
#include "cuda_hashtable.h"
#include "cuda_utils.h"

namespace samgraph {
namespace common {
namespace cuda {

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
    const IdType pos = SearchForPositionO2N(id);

    return GetMutableO2N(pos);
  }

  static constexpr IdType MAX_DELTA = 100;
  enum InsertStatus {
    kConflict = 0,
    kFirstSuccess,
    kDupSuccess,
  };
  inline __device__ InsertStatus AttemptInsertAtO2N(const IdType pos, const IdType id,
                                            const IdType val,
                                            const IdType _) {
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
  inline __device__ IteratorO2N InsertO2N(const IdType id, const IdType index,
                                          const IdType version) {
#ifndef SXN_NAIVE_HASHMAP
    IdType pos = HashO2N(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    InsertStatus ret;
    while ((ret = AttemptInsertAtO2N(pos, id, index, version)) == kConflict) {
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(const IdType *const items,
                                        const size_t num_items,
                                        MutableDeviceSimpleHashTable table,
                                        const IdType global_offset,
                                        const IdType version) {
  assert(BLOCK_SIZE == blockDim.x);

  using IteratorO2N = typename MutableDeviceSimpleHashTable::IteratorO2N;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const IteratorO2N bucket = table.InsertO2N(items[index], index, version);
      // since we are only inserting unique items, we know their local id
      // will be equal to their index
    }
  }
}
template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void evict_hashmap_unique(const IdType *const items,
                                    const size_t num_items,
                                    MutableDeviceSimpleHashTable table,
                                    const IdType version) {
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
__global__ void lookup_hashmap_ifexist(const IdType *const items,
                             const size_t num_items,
                             IdType* pos,
                             MutableDeviceSimpleHashTable table,
                             const IdType version) {
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

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void lookup_val_hashmap(const IdType *const items,
                             const size_t num_items,
                             ValType* vals,
                             MutableDeviceSimpleHashTable table,
                             const IdType version) {
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


// DeviceSimpleHashTable implementation
DeviceSimpleHashTable::DeviceSimpleHashTable(const BucketO2N *const o2n_table,
                                               const size_t o2n_size,
                                               const IdType version)
    : _o2n_table(o2n_table),
      _o2n_size(o2n_size),
      _version(version) {}

DeviceSimpleHashTable SimpleHashTable::DeviceHandle() const {
  return DeviceSimpleHashTable(_o2n_table,
      _o2n_size, _version);
}

// SimpleHashTable implementation
SimpleHashTable::SimpleHashTable(const size_t size, Context ctx,
                                   StreamHandle stream, const size_t scale)
    : _o2n_table(nullptr),
#ifndef SXN_NAIVE_HASHMAP
      _o2n_size(TableSize(size, scale)),
#else
      _o2n_size(size),
#endif
      _ctx(ctx),
      _version(0),
      _num_items(0) {
  // make sure we will at least as many buckets as items.
  auto device = Device::Get(_ctx);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  _o2n_table = static_cast<BucketO2N *>(
      device->AllocDataSpace(_ctx, sizeof(BucketO2N) * _o2n_size));

  CUDA_CALL(cudaMemsetAsync(_o2n_table, (int)Constant::kEmptyKey,
                       sizeof(BucketO2N) * _o2n_size, cu_stream));
  device->StreamSync(_ctx, stream);
  LOG(INFO) << "cuda hashtable init with " << _o2n_size
            << " O2N table size";
}

SimpleHashTable::~SimpleHashTable() {
  Timer t;

  auto device = Device::Get(_ctx);
  device->FreeDataSpace(_ctx, _o2n_table);

  LOG(DEBUG) << "free " << t.Passed();
}

void SimpleHashTable::Reset(StreamHandle stream) {
  _version++;
  _num_items = 0;
}


void SimpleHashTable::FillWithUnique(const IdType *const input,
                                      const size_t num_input,
                                      StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  generate_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table,
                                      _num_items, _version);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);

  _num_items += num_input;

  LOG(DEBUG) << "SimpleHashTable::FillWithUnique insert " << num_input
             << " items, now " << _num_items << " in total";
}

void SimpleHashTable::EvictWithUnique(const IdType *const input,
                                       const size_t num_input,
                                       StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  evict_hashmap_unique<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, device_table, _version);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);

  _num_items -= num_input;

  LOG(DEBUG) << "SimpleHashTable::EvictWithUnique remove " << num_input
             << " items, now " << _num_items << " in total";
}

void SimpleHashTable::LookupIfExist(const IdType *const input, const size_t num_input, IdType *pos, StreamHandle stream) {
  const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
  const dim3 grid(num_tiles);
  const dim3 block(Constant::kCudaBlockSize);

  auto device_table = MutableDeviceSimpleHashTable(this);
  auto cu_stream = static_cast<cudaStream_t>(stream);

  lookup_hashmap_ifexist<Constant::kCudaBlockSize, Constant::kCudaTileSize>
      <<<grid, block, 0, cu_stream>>>(input, num_input, pos, device_table, _version);
  // Device::Get(_ctx)->StreamSync(_ctx, stream);
}

template <size_t BLOCK_SIZE, size_t TILE_SIZE>
__global__ void check_cuda_array_(IdType* array, IdType cmp, IdType num_items, bool exp) {
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

void check_cuda_array(IdType* array, IdType cmp, IdType num_items, bool exp, StreamHandle stream) {
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
  auto out = Tensor::Empty(kI32, {1}, GPU(0), "");
  auto cu_stream = static_cast<cudaStream_t>(stream);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, out->Ptr<IdType>(), this->_o2n_size, cu_stream);
  CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  auto get_out = [out, stream](){
    auto cpu_out = Tensor::CopyTo(out, CPU(), stream);
    Device::Get(GPU(0))->StreamSync(GPU(0), stream);
    return cpu_out->CPtr<IdType>()[0];
  };

  input_iter = cubEntryIs<>(this->_o2n_table, 0, 0x80000000);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, out->Ptr<IdType>(), this->_o2n_size, cu_stream);
  Device::Get(GPU(0))->StreamSync(GPU(0), stream);
  LOG(ERROR) << "Occupied " << get_out();

  input_iter = cubEntryIs<>(this->_o2n_table, 0x80000000, 0x80000000);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, out->Ptr<IdType>(), this->_o2n_size, cu_stream);
  Device::Get(GPU(0))->StreamSync(GPU(0), stream);
  IdType state1 = get_out();

  input_iter = cubEntryIs<>(this->_o2n_table, 0xffffffff, 0xffffffff);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter, out->Ptr<IdType>(), this->_o2n_size, cu_stream);
  Device::Get(GPU(0))->StreamSync(GPU(0), stream);
  IdType default_state = get_out();
  LOG(ERROR) << "Invalid " << state1 - default_state;
  LOG(ERROR) << "Default " << default_state;
}

}  // namespace cuda
}  // namespace common
}  // namespace samgraph

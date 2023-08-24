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
#include <cub/cub.cuh>

#include <cassert>
#include <cstdint>

#include "../device.h"
#include "../common.h"
#include "../constant.h"
#include "../logging.h"
#include "cache_hashtable.h"
#ifdef __linux__
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include "../run_config.h"

namespace coll_cache_lib {
namespace common {

/**
 * @brief A `MurmurHash3_32` hash function to hash the given argument on host and device.
 *
 * MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 * -----------------------------------------------------------------------------
 * MurmurHash3 was written by Austin Appleby, and is placed in the public domain. The author
 * hereby disclaims copyright to this source code.
 *
 * Note - The x86 and x64 versions do _not_ produce the same results, as the algorithms are
 * optimized for their respective platforms. You can still compile and run any of them on any
 * platform, but your performance with the non-native version will be less than optimal.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
struct MurmurHash3_32 {
  using argument_type = Key;       ///< The type of the values taken as argument
  using result_type   = uint32_t;  ///< The type of the hash values produced

  /// Default constructor
  __host__ __device__ constexpr MurmurHash3_32() : MurmurHash3_32{0} {}

  /**
   * @brief Constructs a MurmurHash3_32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  __host__ __device__ constexpr MurmurHash3_32(uint32_t seed) : m_seed(seed) {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return A resulting hash value for `key`
   */
  constexpr result_type __host__ __device__ operator()(Key const& key) const noexcept
  {
    constexpr int len         = sizeof(argument_type);
    const uint8_t* const data = (const uint8_t*)&key;
    constexpr int nblocks     = len / 4;

    uint32_t h1           = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];  // getblock32(blocks,i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1         = 0;
    switch (len & 3) {
      case 3: k1 ^= tail[2] << 16;
      case 2: k1 ^= tail[1] << 8;
      case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }

 private:
  constexpr __host__ __device__ uint32_t rotl32(uint32_t x, int8_t r) const noexcept
  {
    return (x << r) | (x >> (32 - r));
  }

  constexpr __host__ __device__ uint32_t fmix32(uint32_t h) const noexcept
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }
  uint32_t m_seed;
};


class DeviceSimpleHashTable {
 public:
  // 1| 111...111 -> Default state, this bucket is not used yet
  //          insert may use this, search must stop at this
  // 1| xxx...xxx -> Invalid bucket,
  //          insert may use this, but search must proceed beyond this
  // 0| xxx...xxx -> bucket inuse,
  //          insert cannot use this, but search must proceed beyond this
  using BucketO2N = BucketO2N;

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
  MurmurHash3_32<IdType> _hasher;
  const BucketO2N *_o2n_table;
 protected:
  const size_t _o2n_size;

  explicit DeviceSimpleHashTable(const BucketO2N *const o2n_table,
                                 MurmurHash3_32<IdType> hasher,
                                  const size_t o2n_size);


  inline __device__ IdType HashO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    return _hasher(id) % _o2n_size;
#else
    return id;
#endif
  }

  friend class SimpleHashTable;
};

class DeviceFlatHashTable {
 public:
  // 1| 111...111 -> Default state, this bucket is not used yet
  //          insert may use this, search must stop at this
  // 1| xxx...xxx -> Invalid bucket,
  //          insert may use this, but search must proceed beyond this
  // 0| xxx...xxx -> bucket inuse,
  //          insert cannot use this, but search must proceed beyond this
  using BucketFlat = BucketFlat;

  typedef const BucketFlat *ConstIterator;

  DeviceFlatHashTable(const DeviceFlatHashTable &other) = default;
  DeviceFlatHashTable &operator=(const DeviceFlatHashTable &other) =
      default;

  inline __device__ IdType SearchForPositionO2N(const IdType id) const {
    if (_flat_table[id].val.data == kEmptyPos) return kEmptyPos;
    return id;
  }

  inline __device__ ConstIterator SearchO2N(const IdType id) const {
    IdType pos = HashO2N(id);
    if (_flat_table[pos].val.data == kEmptyPos) return nullptr;
    return &_flat_table[pos];
  }
  const BucketFlat *_flat_table;
 protected:
  const size_t _flat_size;

  explicit DeviceFlatHashTable(const BucketFlat *const flat_table,
                                  const size_t flat_size);


  inline __device__ IdType HashO2N(const IdType id) const {
    return id;
  }

  friend class FlatHashTable;
};

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename Helper, typename DeviceHashTable_T>
__global__ void lookup_hashmap_with_helper(const IdType *const items,
                             const size_t num_items,
                             Helper helper,
                             DeviceHashTable_T table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      auto iter = table.SearchO2N(items[index]);
      helper(index, items[index], iter);
    }
  }
}
};

class SimpleHashTable {
 public:
  static constexpr size_t kDefaultScale = 2;

  using BucketO2N = typename DeviceSimpleHashTable::BucketO2N;

  SimpleHashTable(const size_t size, Context ctx,
                   StreamHandle stream, const size_t scale = kDefaultScale);
  SimpleHashTable(BucketO2N* table, const size_t size, Context ctx,
                   StreamHandle stream, const size_t scale = kDefaultScale);
  SimpleHashTable(std::function<MemHandle(size_t)> allocator, const size_t size, Context ctx,
                   StreamHandle stream, const size_t scale = kDefaultScale);

  ~SimpleHashTable();

  // Disable copying
  SimpleHashTable(const SimpleHashTable &other) = delete;
  SimpleHashTable &operator=(const SimpleHashTable &other) = delete;

  void FillWithUnique(const IdType *const input, 
                      const ValType *const vals,
                      const size_t num_input,
                      StreamHandle stream);
  template<typename ValMaker>
  void InsertUnique(const IdType *const input, ValMaker val_maker, const size_t num_input, StreamHandle stream);
  void EvictWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);
  void LookupIfExist(const IdType* const input, const size_t num_input, IdType * pos, StreamHandle stream);
  // void LookupVal(const IdType* const input, const size_t num_input, ValType * vals, StreamHandle stream);
  template<typename DefValMaker>
  void LookupValWithDef(const IdType* const input, const size_t num_input, ValType * vals, DefValMaker default_val_maker, StreamHandle stream);
  template<typename Helper>
  void LookupValCustom(const IdType* const input, const size_t num_input, Helper helper, StreamHandle stream) {
    const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
    const dim3 grid(num_tiles);
    const dim3 block(Constant::kCudaBlockSize);

    auto device_table = DeviceHandle();
    auto cu_stream = static_cast<cudaStream_t>(stream);

    lookup_hashmap_with_helper<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(input, num_input, helper, device_table);
    // Device::Get(_ctx)->StreamSync(_ctx, stream);
  }
  void CountEntries(StreamHandle stream);

  DeviceSimpleHashTable DeviceHandle() const;

  BucketO2N *_o2n_table;
  size_t max_efficient_size;
  MurmurHash3_32<IdType> _hasher;
 private:
  Context _ctx;

  size_t _o2n_size;
  MemHandle _o2n_table_handle;
};

class FlatHashTable {
 public:
  using BucketFlat = typename DeviceFlatHashTable::BucketFlat;

  FlatHashTable(const size_t size, Context ctx,
                   StreamHandle stream);
  FlatHashTable(BucketFlat* table, const size_t size, Context ctx,
                   StreamHandle stream);
  FlatHashTable(std::function<MemHandle(size_t)> allocator, const size_t size, Context ctx,
                   StreamHandle stream);

  ~FlatHashTable();

  // Disable copying
  FlatHashTable(const FlatHashTable &other) = delete;
  FlatHashTable &operator=(const FlatHashTable &other) = delete;

  void FillWithUnique(const IdType *const input, 
                      const ValType *const vals,
                      const size_t num_input,
                      StreamHandle stream);
  template<typename ValMaker>
  void InsertUnique(const IdType *const input, ValMaker val_maker, const size_t num_input, StreamHandle stream);
  void EvictWithUnique(const IdType *const input, const size_t num_input,
                      StreamHandle stream);
  void LookupIfExist(const IdType* const input, const size_t num_input, IdType * pos, StreamHandle stream);
  // void LookupVal(const IdType* const input, const size_t num_input, ValType * vals, StreamHandle stream);
  template<typename DefValMaker>
  void LookupValWithDef(const IdType* const input, const size_t num_input, ValType * vals, DefValMaker default_val_maker, StreamHandle stream);
  template<typename Helper>
  void LookupValCustom(const IdType* const input, const size_t num_input, Helper helper, StreamHandle stream) {
    const size_t num_tiles = RoundUpDiv(num_input, Constant::kCudaTileSize);
    const dim3 grid(num_tiles);
    const dim3 block(Constant::kCudaBlockSize);

    auto device_table = DeviceHandle();
    auto cu_stream = static_cast<cudaStream_t>(stream);

    lookup_hashmap_with_helper<Constant::kCudaBlockSize, Constant::kCudaTileSize>
        <<<grid, block, 0, cu_stream>>>(input, num_input, helper, device_table);
    // Device::Get(_ctx)->StreamSync(_ctx, stream);
  }
  void CountEntries(StreamHandle stream);

  DeviceFlatHashTable DeviceHandle() const;

  BucketFlat *_flat_table;
 private:
  Context _ctx;

  size_t _flat_size;
  MemHandle _flat_table_handle;
};

struct CPUFallback {
  int cpu_location_id;
  CPUFallback(int cpu_loc) : cpu_location_id(cpu_loc) {}
  inline __host__ __device__ ValType operator()(const IdType & key) {
    return ValType(cpu_location_id, key);
  }
};
namespace HashTableLookupHelper {
struct OffsetOnly {
  IdType *offset_list;
  OffsetOnly(IdType* offset_list) : offset_list(offset_list) {}
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    if (val) {
      offset_list[idx] = val->val.off();
    } else {
      offset_list[idx] = key;
    };
  }
};
struct OffsetOnlyMock {
  IdType *offset_list;
  size_t empty_feat;
  OffsetOnlyMock(IdType* offset_list) : offset_list(offset_list) { empty_feat = RunConfig::option_empty_feat; }
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    if (val) {
      offset_list[idx] = val->val.off();
    } else {
      offset_list[idx] = key % (1 << empty_feat);
    };
  }
};
struct LocOnly {
  IdType *loc_list;
  int fallback_loc;
  LocOnly(IdType* loc_list, int fallback_loc) : loc_list(loc_list), fallback_loc(fallback_loc) {}
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    if (val) {
      loc_list[idx] = val->val.loc();
    } else {
      loc_list[idx] = fallback_loc;
    };
  }
};
struct SepLocOffset {
  IdType *loc_list;
  IdType *offset_list;
  int fallback_loc;
  SepLocOffset(IdType* loc_list, IdType* offset_list, int fallback_loc) : loc_list(loc_list), offset_list(offset_list), fallback_loc(fallback_loc) {}
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    if (val) {
      loc_list[idx] = val->val.loc();
      offset_list[idx] = val->val.off();
    } else {
      loc_list[idx] = fallback_loc;
      offset_list[idx] = key;
    };
  }
};
struct SepLocOffsetEmpty {
  IdType *loc_list;
  IdType *offset_list;
  int fallback_loc;
  IdType fallback_off_mask;
  SepLocOffsetEmpty(IdType* loc_list, IdType* offset_list, int fallback_loc, IdType fallback_off_mask) : loc_list(loc_list), offset_list(offset_list), fallback_loc(fallback_loc), fallback_off_mask(fallback_off_mask) {}
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    if (val) {
      loc_list[idx] = val->val.loc();
      offset_list[idx] = val->val.off();
    } else {
      loc_list[idx] = fallback_loc;
      offset_list[idx] = key % fallback_off_mask;
    };
  }
};
struct EmbedVal {
  ValType *val_list;
  int fallback_loc;
  EmbedVal(ValType* val_list, IdType* offset_list, int fallback_loc) : val_list(val_list), fallback_loc(fallback_loc) {}
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    if (val) {
      val_list[idx] = val->val;
    } else {
      val_list[idx] = EmbCacheOff(fallback_loc, key);
    };
  }
};
template<bool use_empty_feat, typename IdxStore_T>
struct LookupHelper {
  IdxStore_T idx_store;
  int fallback_loc_;
  size_t empty_feat_;
  LookupHelper(int fallback_loc) : fallback_loc_(fallback_loc), empty_feat_(RunConfig::option_empty_feat) {}
  template<typename Bucket_T>
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const Bucket_T* val) {
    idx_store.set_dst_off(idx, idx);
    if (val) {
      idx_store.set_src(idx, val->val.loc(), val->val.off());
      // idx_store.set_src_loc(idx, val->val.loc());
      // idx_store.set_src_off(idx, val->val.off());
    } else if (use_empty_feat) {
      idx_store.set_src(idx, fallback_loc_, key % (1 << empty_feat_));
      // idx_store.set_src_loc(idx, fallback_loc_);
      // idx_store.set_src_off(idx, key % (1 << empty_feat_));
    } else {
      idx_store.set_src(idx, fallback_loc_, key);
      // idx_store.set_src_loc(idx, fallback_loc_);
      // idx_store.set_src_off(idx, key);
    }
  }
};
};

namespace HashTableInsertHelper {

struct SingleLoc {
  const IdType *offset_list;
  const int loc;
  SingleLoc(const IdType* offset_list, const int loc) : offset_list(offset_list), loc(loc) {}
  inline __host__ __device__ ValType operator()(const IdType & idx) const {
    return ValType(loc, offset_list[idx]);
  }
};
struct SingleLocSeqOff {
  const int loc;
  SingleLocSeqOff(const int loc) : loc(loc) {}
  inline __host__ __device__ ValType operator()(const IdType & idx) const {
    return ValType(loc, idx);
  }
};

}

void check_cuda_array(const IdType* array, IdType cmp, IdType num_items, bool exp, StreamHandle stream);


class CacheEntryManager {
 public:
  template<bool is_place_on>
  struct PlaceOn {
    PlaceOn<is_place_on>(int location_id) : location_id(location_id) {}
    int location_id;
    bool operator()(const uint8_t & placement) {
      return (bool(placement & (1 << location_id))) == is_place_on;
    }
  };
  struct AccessFrom {
    AccessFrom(int location_id) : location_id(location_id) {}
    int location_id;
    bool operator()(const uint8_t access_from) {
      return access_from == location_id;
    }
  };
  TensorPtr _cached_keys;
  TensorPtr _remote_keys;
  TensorPtr _free_offsets;
  std::shared_ptr<SimpleHashTable> _simple_hash_table = nullptr;
  std::shared_ptr<FlatHashTable> _flat_hash_table = nullptr;
  IdType _cache_space_capacity;
  IdType num_total_key;
  int cpu_location_id;
  TensorPtr _keys_for_each_src[9];
  #define COLL_SWITCH_HASH_TABLE(hash_alias, ...)                  \
  {                                                                \
    if (RunConfig::use_flat_hashtable) {                           \
      auto hash_alias = _flat_hash_table.get(); { __VA_ARGS__ ; }  \
    } else {                                                       \
      auto hash_alias = _simple_hash_table.get(); { __VA_ARGS__ ; }\
    }                                                              \
  }

#ifdef DEAD_CODE
  void Lookup(TensorPtr keys, TensorPtr vals, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    _hash_table->LookupValWithDef<>(keys->CPtr<IdType>(), keys->NumItem(), vals->Ptr<ValType>(), CPUFallback(cpu_location_id), stream);
  }
  void Lookup(TensorPtr keys, TensorPtr loc, TensorPtr off, StreamHandle stream) {
    auto helper = HashTableLookupHelper::SepLocOffset(loc->Ptr<IdType>(), off->Ptr<IdType>(), cpu_location_id);
    _hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);
  }
#endif
  void LookupOffset(TensorPtr keys, TensorPtr off, StreamHandle stream) {
    if (RunConfig::option_empty_feat == 0) {
      auto helper = HashTableLookupHelper::OffsetOnly(off->Ptr<IdType>());
      COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);});
    } else {
      auto helper = HashTableLookupHelper::OffsetOnlyMock(off->Ptr<IdType>());
      COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);});
    }
  }
  template<typename IdxStore_T=IdxStoreAPI>
  void LookupSrcDst(const IdType * keys, const size_t num_keys, IdxStore_T idx_store, int fallback_loc, StreamHandle stream) {
    if (RunConfig::option_empty_feat == 0) {
      auto helper = HashTableLookupHelper::LookupHelper<false, IdxStore_T>(fallback_loc);
      helper.idx_store = idx_store;
      COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->LookupValCustom(keys, num_keys, helper, stream);});
    } else {
      auto helper = HashTableLookupHelper::LookupHelper<true, IdxStore_T>(fallback_loc);
      helper.idx_store = idx_store;
      COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->LookupValCustom(keys, num_keys, helper, stream);});
    }
  }
#ifdef DEAD_CODE
  void LookupLoc(TensorPtr keys, TensorPtr loc, StreamHandle stream) {
    auto helper = HashTableLookupHelper::LocOnly(loc->Ptr<IdType>(), cpu_location_id);
    _hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);
  }
#endif
  void InsertWithLoc(TensorPtr keys, TensorPtr off, int loc, StreamHandle stream) {
    auto helper = HashTableInsertHelper::SingleLoc(off->CPtr<IdType>(), loc);
    COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->InsertUnique(keys->CPtr<IdType>(), helper, keys->NumItem(), stream);});
  }
  void InsertSeqOffWithLoc(TensorPtr keys, int loc, StreamHandle stream) {
    auto helper = HashTableInsertHelper::SingleLocSeqOff(loc);
    COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->InsertUnique(keys->CPtr<IdType>(), helper, keys->NumItem(), stream);});
  }
#ifdef DEAD_CODE
  TensorPtr EvictLocal(TensorPtr keys_to_evict, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    auto offsets = Tensor::Empty(kI32, keys_to_evict->Shape(), keys_to_evict->Ctx(), "");
    LookupOffset(keys_to_evict, offsets, stream);
    _hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), keys_to_evict->NumItem(), stream);
    return offsets;
  }
#endif
  void EvictRemote(TensorPtr keys_to_evict, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), keys_to_evict->NumItem(), stream);});
  }
  void Evict(TensorPtr keys_to_evict, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    COLL_SWITCH_HASH_TABLE(_hash_table, {_hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), keys_to_evict->NumItem(), stream);});
  }
  void SortFreeOffsets() {
#ifdef __linux__
    __gnu_parallel::sort(_free_offsets->Ptr<IdType>(), &_free_offsets->Ptr<IdType>()[_free_offsets->NumItem()],
                         std::less<IdType>());
#else
    std::sort(_free_offsets->CPtr<IdType>(), &_free_offsets->CPtr<IdType>()[_free_offsets->NumItem()],
              std::greater<IdType>());
#endif
  }
  void ReturnOffset(TensorPtr new_offsets) {
    IdType old_num_offsets = _free_offsets->NumItem();
    _free_offsets->ForceScale(kI32, {old_num_offsets + new_offsets->NumItem()}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    CHECK(new_offsets->Ctx().device_type == kCPU);
    Device::Get(CPU(CPU_CLIB_MALLOC_DEVICE))->CopyDataFromTo(new_offsets->Data(), 0, _free_offsets->Ptr<IdType>() + old_num_offsets, 0, new_offsets->NumItem() * sizeof(IdType), CPU(), CPU(), nullptr);
  }
  TensorPtr ReserveOffset(IdType num_new_insert_keys) {
    CHECK(_free_offsets->Shape()[0] > num_new_insert_keys);
    auto ret = Tensor::CopyBlob(_free_offsets->CPtr<IdType>() + _free_offsets->NumItem() - num_new_insert_keys, kI32, {num_new_insert_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), CPU(CPU_CLIB_MALLOC_DEVICE), "");
    _free_offsets->ForceScale(kI32, {_free_offsets->NumItem() - num_new_insert_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    return ret;
  }
  TensorPtr ReserveOffsetFront(IdType num_new_insert_keys) {
    CHECK(_free_offsets->Shape()[0] > num_new_insert_keys);
    auto ret = Tensor::CopyBlob(_free_offsets->CPtr<IdType>(), kI32, {num_new_insert_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), CPU(CPU_CLIB_MALLOC_DEVICE), "");
    for (size_t i = 0; i < _free_offsets->NumItem() - num_new_insert_keys; i++) {
      _free_offsets->Ptr<IdType>()[i] = _free_offsets->Ptr<IdType>()[i + num_new_insert_keys];
    }
    _free_offsets->ForceScale(kI32, {_free_offsets->NumItem() - num_new_insert_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    return ret;
  }
  template<typename InputIterator, typename PlaceCond>
  TensorPtr DetectKeysWithPlacement(InputIterator inputs, IdType num_input, TensorPtr nid_to_block, TensorPtr block_placement, PlaceCond cond) {
    CHECK_NOTNULL(inputs);
    CHECK_NOTNULL(nid_to_block);
    CHECK_NOTNULL(block_placement);
    TensorPtr matched_keys = Tensor::Empty(kI32, {_cache_space_capacity}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    size_t num_matched_keys = 0;
    for (IdType idx = 0; idx < num_input; idx++) {
      auto key = inputs[idx];
      IdType block_id = nid_to_block->Ptr<IdType>()[key];
      if (cond(block_placement->Ptr<uint8_t>()[block_id])) {
      // if ((block_placement->Ptr<uint8_t>()[block_id] & (1 << local_location_id)) != 0) {
        CHECK(num_matched_keys < _cache_space_capacity);
        matched_keys->Ptr<IdType>()[num_matched_keys] = key;
        num_matched_keys++;
      }
    }
    matched_keys->ForceScale(kI32, {num_matched_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    return matched_keys;
  }
  template<typename InputIterator, typename AccessCond>
  TensorPtr DetectKeysWithAccess(InputIterator inputs, IdType num_input, TensorPtr nid_to_block, TensorPtr block_access_advise, AccessCond cond) {
    TensorPtr matched_keys = Tensor::Empty(kI32, {_cache_space_capacity}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    size_t num_matched_keys = 0;
    for (IdType idx = 0; idx < num_input; idx++) {
      auto key = inputs[idx];
      IdType block_id = nid_to_block->Ptr<IdType>()[key];
      if (cond(block_access_advise->Ptr<uint8_t>()[block_id])) {
      // if ((block_placement->Ptr<uint8_t>()[block_id] & (1 << local_location_id)) != 0) {
        CHECK(num_matched_keys < _cache_space_capacity);
        matched_keys->Ptr<IdType>()[num_matched_keys] = key;
        num_matched_keys++;
      }
    }
    matched_keys->ForceScale(kI32, {num_matched_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    return matched_keys;
  }
  template<typename InputIterator, typename CondT>
  TensorPtr DetectKeysWithCond(InputIterator inputs, IdType num_input, CondT cond, size_t max_result_storage = 0) {
    if (max_result_storage == 0) {
      LOG(ERROR) << "Please set max result storage for efficient mem usage";
      max_result_storage = _cache_space_capacity;
    }
    TensorPtr matched_keys;
    size_t global_cur_len = 0;
    #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num_per_gpu)
    {
      size_t local_buffer_len = std::min<size_t>(max_result_storage * 1.1 / RunConfig::solver_omp_thread_num_per_gpu, 131072);
      TensorPtr local_tensor = Tensor::Empty(kI32, {local_buffer_len}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
      auto local_arr = local_tensor->Ptr<IdType>();
      size_t local_cur_len = 0;
      #pragma omp for
      for (IdType idx = 0; idx < num_input; idx++) {
        auto key = inputs[idx];
        if (!cond(key)) continue;
        if (local_cur_len == local_buffer_len) {
          #pragma omp critical
          {
            if (matched_keys == nullptr) matched_keys = Tensor::Empty(kI32, {max_result_storage}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
            memcpy(matched_keys->Ptr<IdType>() + global_cur_len, local_arr, local_cur_len * sizeof(IdType));
            global_cur_len += local_cur_len;
            local_cur_len = 0;
          }
        }
        CHECK(local_cur_len < local_buffer_len);
        local_arr[local_cur_len] = key;
        local_cur_len++;
      }
      // local_tensor->ForceScale(kI32, {local_cur_len}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
      #pragma omp critical
      {
        if (local_cur_len > 0) {
          if (matched_keys == nullptr) matched_keys = Tensor::Empty(kI32, {max_result_storage}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
          memcpy(matched_keys->Ptr<IdType>() + global_cur_len, local_arr, local_cur_len * sizeof(IdType));
          global_cur_len += local_cur_len;
        }
      }
    }
    matched_keys->ForceScale(kI32, {global_cur_len}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
#ifdef COLL_HASH_VALID_LEGACY
    __gnu_parallel::sort(matched_keys->Ptr<IdType>(), matched_keys->Ptr<IdType>() + global_cur_len,
                        std::less<IdType>());
#endif
    return matched_keys;
  }
  static void DetectKeysForAllSource(TensorPtr nid_to_block, TensorPtr block_access_advise, int local_location_id, TensorPtr block_density, size_t num_total_item,
      TensorPtr* node_list_of_src, int num_gpu = 8) {
    // LOG(ERROR) << "detecting key for all source with nthread " << RunConfig::solver_omp_thread_num_per_gpu ;
    CHECK_EQ(block_access_advise->Shape().size(), 1);
    // TensorPtr block_access_advise = Tensor::CopyLine(_cache_ctx->_coll_cache->_block_access_advise, local_location_id, CPU(CPU_CLIB_MALLOC_DEVICE), stream); // small
    size_t num_blocks = block_access_advise->Shape()[0];
    size_t per_src_size[9] = {0};
    size_t per_src_cur_size[9] = {0};
    const IdType* nid_to_block_ptr = nid_to_block->CPtr<IdType>();
    const uint8_t* block_access_advise_ptr = block_access_advise->CPtr<uint8_t>();

    for (size_t i = 0; i < num_blocks; i++) {
      IdType src = block_access_advise->CPtr<uint8_t>()[i];
      per_src_size[src] += (block_density->CPtr<double>()[i]) * num_total_item / 100;
    }
    for (auto & per_s_s : per_src_size) {
      per_s_s *= 1.1;
      per_s_s += 100;
    }
    for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
      node_list_of_src[dev_id] = Tensor::Empty(kI32, {per_src_size[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    }

    if (local_location_id == 0) LOG(ERROR) << "per src size local is " << per_src_size[local_location_id];
    #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num_per_gpu)
    {
      std::vector<TensorPtr> local_ten_list(num_gpu);
      std::vector<IdType*> local_ptr_list(num_gpu);
      std::vector<size_t> local_cur_len_list(num_gpu, 0);
      std::vector<size_t> local_max_len_list(num_gpu);
      for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
        local_max_len_list[dev_id] = std::min<size_t>(per_src_size[dev_id] / RunConfig::solver_omp_thread_num_per_gpu, 131072);
        local_ten_list[dev_id] = Tensor::Empty(kI32, {local_max_len_list[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
        local_ptr_list[dev_id] = local_ten_list[dev_id]->Ptr<IdType>();
      }
      #pragma omp for
      for (size_t node_id = 0; node_id < num_total_item; node_id++) {
        IdType block_id = nid_to_block_ptr[node_id];
        int dev_id = block_access_advise_ptr[block_id];
        if (dev_id == num_gpu) continue;
        auto & local_cur_len = local_cur_len_list[dev_id];
        auto local_arr = local_ptr_list[dev_id];
        if (local_cur_len == local_max_len_list[dev_id]) {
          #pragma omp critical
          {
            auto & global_cur_len = per_src_cur_size[dev_id];
            if (node_list_of_src[dev_id] == nullptr) {
              node_list_of_src[dev_id] = Tensor::Empty(kI32, {per_src_size[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
            }
            if (global_cur_len + local_cur_len > per_src_size[dev_id]) {
              CHECK(false) << "estimated per src array is too small on src " << dev_id << " dst " << local_location_id << ", expected: " << per_src_size[dev_id] << ", actual found " << global_cur_len + local_cur_len;
            }
            memcpy(node_list_of_src[dev_id]->Ptr<IdType>() + global_cur_len, local_arr, local_cur_len * sizeof(IdType));
            global_cur_len += local_cur_len;
            local_cur_len = 0;
          }
        }
        CHECK_LT(local_cur_len, local_max_len_list[dev_id]);
        local_arr[local_cur_len] = node_id;
        local_cur_len++;
      }
      // for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
      //   local_ten_list[dev_id]->ForceScale(kI32, {local_cur_len_list[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
      // }
      #pragma omp critical
      {
        for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
          auto local_len = local_cur_len_list[dev_id];
          auto local_arr = local_ptr_list[dev_id];
          auto & dst_len = per_src_cur_size[dev_id];
          if (local_len == 0) continue;
          if (node_list_of_src[dev_id] == nullptr) {
            node_list_of_src[dev_id] = Tensor::Empty(kI32, {per_src_size[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
          }
          memcpy(node_list_of_src[dev_id]->Ptr<IdType>() + dst_len, local_arr, local_len * sizeof(IdType));
          dst_len += local_len;
        }
      }
      #pragma omp barrier
      #pragma omp for
      for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
        node_list_of_src[dev_id]->ForceScale(kI32, {per_src_cur_size[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
        if (RunConfig::coll_hash_impl == kRR || RunConfig::coll_hash_impl == kChunk) {
          __gnu_parallel::sort(node_list_of_src[dev_id]->Ptr<IdType>(), node_list_of_src[dev_id]->Ptr<IdType>() + per_src_cur_size[dev_id],
                               std::less<IdType>());
        } else {
#ifdef COLL_HASH_VALID_LEGACY
          __gnu_parallel::sort(node_list_of_src[dev_id]->Ptr<IdType>(), node_list_of_src[dev_id]->Ptr<IdType>() + per_src_cur_size[dev_id],
                               std::less<IdType>());
#endif
        }
      }
    }
    // LOG(ERROR) << "detecting key for all source - done";
  }
  // void DetectEvictKeys(TensorPtr _nid_to_block, TensorPtr _block_placement) {
  //   TensorPtr evicted_node_list_cpu = Tensor::Empty(kI32, {_cached_keys->Shape()[0] - num_preserved_node}, CPU(CPU_CLIB_MALLOC_DEVICE), "");

  //   size_t num_eviced_node = 0;
  //   for (IdType i = 0; i < _cached_keys->Shape()[0]; i++) {
  //     IdType node_id = cache_node_list_cpu->Ptr<IdType>()[i];
  //     IdType block_id = _nid_to_block->Ptr<IdType>()[node_id];
  //     if ((_block_placement->Ptr<uint8_t>()[block_id] & (1 << _local_location_id)) == 0) {
  //       evicted_node_list_cpu->Ptr<IdType>()[num_eviced_node] = node_id;
  //       num_eviced_node++;
  //     }
  //   }
  // }
};

}  // namespace common
}  // namespace samgraph

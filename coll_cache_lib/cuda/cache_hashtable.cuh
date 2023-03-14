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

  const BucketO2N *_o2n_table;
 protected:
  const size_t _o2n_size;

  explicit DeviceSimpleHashTable(const BucketO2N *const o2n_table,
                                  const size_t o2n_size);


  inline __device__ IdType HashO2N(const IdType id) const {
#ifndef SXN_NAIVE_HASHMAP
    return id % _o2n_size;
#else
    return id;
#endif
  }

  friend class SimpleHashTable;
};

namespace {

template <size_t BLOCK_SIZE, size_t TILE_SIZE, typename Helper>
__global__ void lookup_hashmap_with_helper(const IdType *const items,
                             const size_t num_items,
                             Helper helper,
                             DeviceSimpleHashTable table) {
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
 private:
  Context _ctx;

  size_t _o2n_size;
  MemHandle _o2n_table_handle;
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
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
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
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
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
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
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
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
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
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
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
  inline __host__ __device__ void operator()(const IdType & idx, const IdType & key, const DeviceSimpleHashTable::BucketO2N* val) {
    if (val) {
      val_list[idx] = val->val;
    } else {
      val_list[idx] = EmbCacheOff(fallback_loc, key);
    };
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
  TensorPtr _free_offsets;
  std::shared_ptr<SimpleHashTable> _hash_table;
  IdType _cache_space_capacity;
  IdType num_total_key;
  int cpu_location_id;
  TensorPtr _keys_for_each_src[9];
  void Lookup(TensorPtr keys, TensorPtr vals, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    _hash_table->LookupValWithDef<>(keys->CPtr<IdType>(), keys->NumItem(), vals->Ptr<ValType>(), CPUFallback(cpu_location_id), stream);
  }
  void Lookup(TensorPtr keys, TensorPtr loc, TensorPtr off, StreamHandle stream) {
    auto helper = HashTableLookupHelper::SepLocOffset(loc->Ptr<IdType>(), off->Ptr<IdType>(), cpu_location_id);
    _hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);
  }
  void LookupOffset(TensorPtr keys, TensorPtr off, StreamHandle stream) {
    if (RunConfig::option_empty_feat == 0) {
      auto helper = HashTableLookupHelper::OffsetOnly(off->Ptr<IdType>());
      _hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);
    } else {
      auto helper = HashTableLookupHelper::OffsetOnlyMock(off->Ptr<IdType>());
      _hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);
    }
  }
  void LookupLoc(TensorPtr keys, TensorPtr loc, StreamHandle stream) {
    auto helper = HashTableLookupHelper::LocOnly(loc->Ptr<IdType>(), cpu_location_id);
    _hash_table->LookupValCustom(keys->CPtr<IdType>(), keys->NumItem(), helper, stream);
  }
  void InsertWithLoc(TensorPtr keys, TensorPtr off, int loc, StreamHandle stream) {
    auto helper = HashTableInsertHelper::SingleLoc(off->CPtr<IdType>(), loc);
    _hash_table->InsertUnique(keys->CPtr<IdType>(), helper, keys->NumItem(), stream);
  }
  void InsertSeqOffWithLoc(TensorPtr keys, int loc, StreamHandle stream) {
    auto helper = HashTableInsertHelper::SingleLocSeqOff(loc);
    _hash_table->InsertUnique(keys->CPtr<IdType>(), helper, keys->NumItem(), stream);
  }
  TensorPtr EvictLocal(TensorPtr keys_to_evict, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    auto offsets = Tensor::Empty(kI32, keys_to_evict->Shape(), keys_to_evict->Ctx(), "");
    LookupOffset(keys_to_evict, offsets, stream);
    _hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), keys_to_evict->NumItem(), stream);
    return offsets;
  }
  void EvictRemote(TensorPtr keys_to_evict, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    _hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), keys_to_evict->NumItem(), stream);
  }
  void Evict(TensorPtr keys_to_evict, StreamHandle stream) {
    CHECK(sizeof(ValType) == 4);
    _hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), keys_to_evict->NumItem(), stream);
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
  void DetectNewInsertKeys(
      TensorPtr old_nid_to_block, TensorPtr old_block_placement,
      TensorPtr new_nid_to_block, TensorPtr new_block_placement,
      int local_location) {
    TensorPtr new_local_keys = DetectKeysWithPlacement(cub::CountingInputIterator<IdType>(0), num_total_key, new_nid_to_block, new_block_placement, PlaceOn<true>(local_location));
    TensorPtr new_insert_keys = DetectKeysWithPlacement(new_local_keys->CPtr<IdType>(), new_local_keys->Shape()[0], old_nid_to_block, old_block_placement, PlaceOn<false>(local_location));
    TensorPtr evict_keys = DetectKeysWithPlacement(_cached_keys->CPtr<IdType>(), _cached_keys->Shape()[0], old_nid_to_block, old_block_placement, PlaceOn<false>(local_location));
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
      max_result_storage = _cache_space_capacity;
    }
    TensorPtr matched_keys = Tensor::Empty(kI32, {max_result_storage}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    size_t num_matched_keys = 0;
    for (IdType idx = 0; idx < num_input; idx++) {
      auto key = inputs[idx];
      if (cond(key)) {
      // if ((block_placement->Ptr<uint8_t>()[block_id] & (1 << local_location_id)) != 0) {
        CHECK(num_matched_keys < max_result_storage);
        matched_keys->Ptr<IdType>()[num_matched_keys] = key;
        num_matched_keys++;
      }
    }
    matched_keys->ForceScale(kI32, {num_matched_keys}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
    return matched_keys;
  }
  static void DetectKeysForAllSource(TensorPtr nid_to_block, TensorPtr block_access_advise, int local_location_id, TensorPtr block_density, size_t num_total_item,
      TensorPtr* node_list_of_src, int num_gpu = 8) {
    LOG(ERROR) << "detecting key for all source";
    CHECK_EQ(block_access_advise->Shape().size(), 1);
    // TensorPtr block_access_advise = Tensor::CopyLine(_cache_ctx->_coll_cache->_block_access_advise, local_location_id, CPU(CPU_CLIB_MALLOC_DEVICE), stream); // small
    size_t num_blocks = block_access_advise->Shape()[0];
    size_t per_src_size[9] = {0};
    size_t per_src_cur_size[9] = {0};
    const IdType* nid_to_block_ptr = nid_to_block->CPtr<IdType>();
    const uint8_t* block_access_advise_ptr = block_access_advise->CPtr<uint8_t>();

    for (size_t i = 0; i < num_blocks; i++) {
      IdType src = block_access_advise->CPtr<uint8_t>()[i];
      per_src_size[src] += (block_density->CPtr<double>()[i] + 0.1) * num_total_item / 100;
    }

    #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
    {
      std::vector<TensorPtr> local_k_list_of_src(num_gpu);
      std::vector<IdType*> local_k_list_of_src_ptr(num_gpu);
      std::vector<size_t> local_k_list_of_src_len(num_gpu, 0);
      std::vector<size_t> max_per_src_size(num_gpu);
      for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
        local_k_list_of_src[dev_id] = Tensor::Empty(kI32, {per_src_size[dev_id] / RunConfig::solver_omp_thread_num}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
        max_per_src_size[dev_id] = per_src_size[dev_id] / RunConfig::solver_omp_thread_num;
        local_k_list_of_src_ptr[dev_id] = local_k_list_of_src[dev_id]->Ptr<IdType>();
      }
      #pragma omp for
      for (size_t node_id = 0; node_id < num_total_item; node_id++) {
        IdType block_id = nid_to_block_ptr[node_id];
        int dev_id = block_access_advise_ptr[block_id];
        if (dev_id == num_gpu) continue;
        CHECK_LE(local_k_list_of_src_len[dev_id], max_per_src_size[dev_id]);
        local_k_list_of_src_ptr[dev_id][local_k_list_of_src_len[dev_id]] = node_id;
        local_k_list_of_src_len[dev_id]++;
      }
      for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
        local_k_list_of_src[dev_id]->ForceScale(kI32, {local_k_list_of_src_len[dev_id]}, CPU(CPU_CLIB_MALLOC_DEVICE), "");
      }
      #pragma omp critical
      {
        for (int dev_id = 0; dev_id < num_gpu; dev_id++) {
          auto local_len = local_k_list_of_src_len[dev_id];
          auto local_arr = local_k_list_of_src_ptr[dev_id];
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
        __gnu_parallel::sort(node_list_of_src[dev_id]->Ptr<IdType>(), node_list_of_src[dev_id]->Ptr<IdType>() + per_src_cur_size[dev_id],
                            std::less<IdType>());
      }
    }
    LOG(ERROR) << "detecting key for all source - done";
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

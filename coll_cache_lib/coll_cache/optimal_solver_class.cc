#include "../common.h"
// #include "../run_config.h"
// #include "../logging.h"
#include "../device.h"
#include "../cpu/mmap_cpu_device.h"
#include "../cpu/cpu_utils.h"
#include "../timer.h"
#include "ndarray.h"
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sys/fcntl.h>
#include <cstring>
#include <bitset>
#include "gurobi_c++.h"
#include <tbb/concurrent_unordered_map.h>
#include "optimal_solver_class.h"

namespace coll_cache_lib {
namespace common {
namespace coll_cache {

std::string CollCacheSolver::_shm_name_nid_to_block  = std::string("coll_cache_block_nid_to_block_") + GetEnvStrong("USER");
std::string CollCacheSolver::_shm_name_access        = std::string("coll_cache_block_access_from_") + GetEnvStrong("USER");
std::string CollCacheSolver::_shm_name_place         = std::string("coll_cache_block_placement_") + GetEnvStrong("USER");
std::string CollCacheSolver::_shm_name_dens          = std::string("coll_cache_block_placement_") + GetEnvStrong("USER") + "_density";

std::string CollCacheSolver::_shm_name_alter_nid_to_block  = std::string("coll_cache_block_nid_to_block_") + GetEnvStrong("USER")             + "_old";
std::string CollCacheSolver::_shm_name_alter_access        = std::string("coll_cache_block_access_from_") + GetEnvStrong("USER")                 + "_old";
std::string CollCacheSolver::_shm_name_alter_place         = std::string("coll_cache_block_placement_") + GetEnvStrong("USER")              + "_old";
std::string CollCacheSolver::_shm_name_alter_dens          = std::string("coll_cache_block_placement_") + GetEnvStrong("USER") + "_density" + "_old";

void CollCacheSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<PerT> device_to_cache_percent,
                            std::string mode, double T_local, double T_cpu) {
  Solve(device_to_stream, device_to_cache_percent, mode, T_local, RunConfig::coll_cache_link_desc.AggregatedRemoteTime(), T_cpu);
};

void OptimalSolver::BuildSingleStream(TensorPtr stream_id_list, TensorPtr stream_freq_list, std::vector<int> device_to_stream, const IdType num_node, const TensorPtr nid_to_block_tensor) {
  CHECK_EQ(stream_id_list->Shape().size(), 2);
  CHECK_EQ(stream_freq_list->Shape().size(), 2);
  CHECK_EQ(stream_id_list->Shape(), stream_freq_list->Shape());
  CHECK_EQ(stream_id_list->Shape()[1], num_node);
  IdType num_stream = stream_id_list->Shape()[0];
  CHECK(num_stream == 1);
  {
    const IdType *stream_id_ptr = stream_id_list->CPtr<IdType>();
    const IdType *stream_freq_ptr = stream_freq_list->CPtr<IdType>();
    return BuildSingleStream(
      [stream_id_ptr, stream_freq_ptr](IdType rnk){return stream_id_ptr[rnk]; },
      [stream_id_ptr, stream_freq_ptr](IdType rnk){return stream_freq_ptr[rnk]; },
      [stream_id_ptr, stream_freq_ptr](IdType rnk){return rnk; },
      stream_freq_list->CPtr<IdType>()[0], device_to_stream, num_node, nid_to_block_tensor);
  }
#ifdef DEAD_CODE
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  // coarse-grained slice to reduce asymm solve time
  // symmetric & switch's precision still holds
  max_size_per_block = num_node / 1000;
  if (GetEnv("COLL_BLOCK_SLICE_GRAIN") != "") {
    max_size_per_block = num_node / std::stod(GetEnv("COLL_BLOCK_SLICE_GRAIN"));
  }
  auto freq_to_slot = [this](float freq, uint32_t rank, IdType num_node){ return this->freq_to_slot_1(freq, num_node);};

  TensorPtr nid_to_freq_tensor = Tensor::Empty(kI32, {num_node}, cpu_ctx, "");
  CHECK(nid_to_block_tensor->Shape() == std::vector<size_t>{num_node});

  uint32_t* nid_to_block = nid_to_block_tensor->Ptr<uint32_t>();

  concurrent_full_slot_map slot_array_to_full_block;

  // TensorView<IdType> stream_id_list_view(stream_id_list);
  // TensorView<IdType> stream_freq_list_view(stream_freq_list);
  const IdType *stream_id_ptr = stream_id_list->CPtr<IdType>();
  const IdType *stream_freq_ptr = stream_freq_list->CPtr<IdType>();

  // identify freq boundary of first slot
  // when cache rate is extremely small, better use the largest val as alpha
  if (alpha < stream_freq_ptr[0]) {
    alpha = stream_freq_ptr[0];
  }
  if (GetEnv("COLL_BLOCK_SLICE_BASE") != "") {
    RunConfig::coll_cache_coefficient = std::stod(GetEnv("COLL_BLOCK_SLICE_BASE"));
  }
  LOG(ERROR) << "reconfigured max freq to be " << alpha;
  RunConfig::coll_cache_num_slot = std::floor(std::log2(alpha) / std::log2(RunConfig::coll_cache_coefficient)) + 1;
  LOG(ERROR) << "reconfigured num slot to be " << RunConfig::coll_cache_num_slot;

  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */

  LOG(WARNING) << "counting slots...";
  size_t max_seq_slot = 0;
  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    uint32_t * nid_to_freq = nid_to_freq_tensor->Ptr<uint32_t>();
    std::unordered_map<size_t, full_slot_single_thread> the_map;
    #pragma omp for
    for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
      uint32_t nid = stream_id_ptr[orig_rank];
      nid_to_freq[nid] = stream_freq_ptr[orig_rank];
      double freq = stream_freq_ptr[orig_rank];
      size_t seq_slot_id = freq_to_slot(freq, orig_rank, num_node);
      nid_to_block[nid] = seq_slot_id;
      {
        auto iter = the_map.find(seq_slot_id);
        if (iter == the_map.end()) {
          iter = the_map.insert({seq_slot_id, full_slot_single_thread()}).first;
          auto & val = iter->second;
          val.orig_seq_slot = seq_slot_id;
          val.size = 1;
        } else {
          iter->second.size++;
        }
      }
    }

    #pragma omp critical
    {
      for (auto local_iter = the_map.begin(); local_iter != the_map.end(); local_iter++) {
        auto global_iter = slot_array_to_full_block.the_map.find(local_iter->first);
        if (slot_array_to_full_block.the_map.find(local_iter->first) == slot_array_to_full_block.the_map.end()) {
          max_seq_slot = std::max(max_seq_slot, local_iter->first);
          global_iter = slot_array_to_full_block.the_map.insert({local_iter->first, local_iter->second}).first;
          global_iter->second.remmaped_slot = slot_array_to_full_block.__next_free_slot++;
        } else {
          global_iter->second.size += local_iter->second.size;
        }
      }
    }
  }
  slot_array_to_full_block.next_free_slot.store(slot_array_to_full_block.__next_free_slot);
  LOG(WARNING) << "Final num slot is " << slot_array_to_full_block.next_free_slot.load();
  block_identifer* buckets = new block_identifer[slot_array_to_full_block.next_free_slot.load()];
  next_free_block.store(0);

  LOG(WARNING) << "preparing block granularity...";
  for (auto iter = slot_array_to_full_block.the_map.begin(); iter != slot_array_to_full_block.the_map.end(); iter++) {
    buckets[iter->second.remmaped_slot]._total_nodes = iter->second.size;
  }

  // zero block
  size_t accumulate_size = 0;
  size_t accumulate_num_slice = 0;
  bool found_bound = false;
  for (size_t seq_slot_id = 0; seq_slot_id <= max_seq_slot; seq_slot_id++) {
    auto iter = slot_array_to_full_block.the_map.find(seq_slot_id);
    if (iter == slot_array_to_full_block.the_map.end()) continue;
    accumulate_size += iter->second.size;
    auto & bucket = buckets[iter->second.remmaped_slot];
    if (found_bound == false) {
      auto slice_size = std::min(max_size_per_block, RoundUpDiv<uint32_t>(iter->second.size, device_to_stream.size()));
      bucket.set_max_size(device_to_stream.size(), slice_size);
      bucket.num_slices = RoundUpDiv(iter->second.size, slice_size);
      bucket.num_slices = RoundUp<uint32_t>(bucket.num_slices, device_to_stream.size());
    } else {
      bucket.set_max_size(1, iter->second.size);
      bucket.num_slices = 1;
    }
    if (accumulate_size / (double)num_node >= RunConfig::cache_percentage * device_to_stream.size()) {
      found_bound = true;
    }
    bucket.slice_begin = accumulate_num_slice;
    accumulate_num_slice += bucket.num_slices;
    LOG(ERROR) << "slot " << iter->first << " has " << iter->second.size << " nodes, max_size set to " << buckets[iter->second.remmaped_slot].max_size_this_block
               << ", #slice=" << bucket.num_slices;
  }

  next_free_block.store(accumulate_num_slice);

  LOG(WARNING) << "counting blocks...";
  std::vector<std::uint32_t> seeds(RunConfig::solver_omp_thread_num);
  {
    std::seed_seq seq{1, 2, 3, 4, 5};
    seq.generate(seeds.begin(), seeds.end());
  }
  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    std::mt19937 gen(seeds[omp_get_thread_num()]);
    #pragma omp for
    for (uint32_t nid = 0; nid < num_node; nid++) {
      auto seq_slot_id = nid_to_block[nid];
      auto remapped_block_id = slot_array_to_full_block.the_map[seq_slot_id].remmaped_slot;
      auto &bucket = buckets[remapped_block_id];
      nid_to_block[nid] = std::uniform_int_distribution<uint32_t>(0, bucket.num_slices - 1)(gen) + bucket.slice_begin;
    }
  }
  delete[] buckets;

  uint32_t total_num_blocks = next_free_block.load();
  LOG(WARNING) << "Final num block is " << total_num_blocks;

  /**
   * Sum frequency & density of each block
   */
  LOG(WARNING) << "counting freq and density...";
  // block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "coll_cache.block_density_tensor");
  block_density_tensor = Tensor::CreateShm(this->_shm_name_dens, kF64, {total_num_blocks}, "");
  block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");

  std::memset(block_density_tensor->MutableData(), 0, block_density_tensor->NumBytes());
  std::memset(block_freq_tensor->MutableData(), 0, block_freq_tensor->NumBytes());

  // TensorView<double> block_density_array(block_density_tensor);
  // TensorView<double> block_freq_array(block_freq_tensor);
  double* block_density_array = block_density_tensor->Ptr<double>();
  double* block_freq_array    = block_freq_tensor->Ptr<double>();
  double min_freq = 1e-2;
  if (GetEnv("COLL_MIN_FREQ") != "") {
    min_freq = std::stod(GetEnv("COLL_MIN_FREQ"));
  }

  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    auto local_block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "");
    auto local_block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "");

    auto local_block_density = local_block_density_tensor->Ptr<double>();
    auto local_block_freq    = local_block_freq_tensor->Ptr<double>();

    std::memset(local_block_density_tensor->MutableData(), 0, local_block_density_tensor->NumBytes());
    std::memset(local_block_freq_tensor->MutableData(), 0, local_block_freq_tensor->NumBytes());

    const auto nid_to_freq = nid_to_freq_tensor->CPtr<uint32_t>();

    #pragma omp for
    for (uint32_t nid = 0; nid < num_node; nid++) {
      uint32_t block_id = nid_to_block[nid];
      local_block_density[block_id]++;
      double freq = nid_to_freq[nid];
      freq = std::max(freq, min_freq);
      local_block_freq[block_id] += freq;
    }
    #pragma omp critical
    {
      for (uint32_t block_id = 0; block_id < total_num_blocks; block_id++) {
        block_density_array[block_id] += local_block_density[block_id];
        block_freq_array[block_id] += local_block_freq[block_id];
      }
    }
  }

  /**
   * Average the frequency for each block
   */
  LOG(WARNING) << "averaging freq and density...";
  LOG(WARNING) << block_density_tensor->NumItem();
// #pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num)
  for (uint32_t block_id = 0; block_id < block_density_tensor->NumItem(); block_id++) {
    if (block_density_array[block_id] == 0) continue; 
    block_freq_array[block_id] /= block_density_array[block_id];
    block_density_array[block_id] *= 100/(double)num_node ;
    // std::cout << block_density_array[block_id].ref() << " ";
  }
  // std::cout << "\n";
  block_placement = Tensor::CreateShm(_shm_name_place, kU8, block_density_tensor->Shape(), "coll_cache_block_placement");
#endif
}

void OptimalSolver::BuildSingleStream(IdTypeMapper nid_iter, IdTypeMapper freq_iter, IdTypeMapper rnk_iter, IdType max_freq, std::vector<int> device_to_stream, const IdType num_node, const TensorPtr nid_to_block_tensor) {
  // {
  //   LegacyFreqBuf freq_buf;
  //   freq_rank->GetLegacyFreqRank(&freq_buf, num_node);
  //   auto ranking_nodes_list = Tensor::FromBlob(
  //       freq_buf.rank_vec.data(), coll_cache::get_data_type<IdType>(),
  //       {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_list");
  //   auto ranking_nodes_freq_list = Tensor::FromBlob(
  //       freq_buf.freq_vec.data(), coll_cache::get_data_type<IdType>(),
  //       {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_freq_list");
  //   BuildSingleStream(ranking_nodes_list, ranking_nodes_freq_list, device_to_stream, num_node, nid_to_block_tensor);
  //   return;
  // }

  IdType num_stream = 1;
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  // coarse-grained slice to reduce asymm solve time
  // symmetric & switch's precision still holds
  max_size_per_block = num_node / 1000;
  if (GetEnv("COLL_BLOCK_SLICE_GRAIN_BY_CACHE") != "") {
    max_size_per_block = num_node * RunConfig::cache_percentage / std::stod(GetEnv("COLL_BLOCK_SLICE_GRAIN_BY_CACHE"));
  } else if (GetEnv("COLL_BLOCK_SLICE_GRAIN") != "") {
    max_size_per_block = num_node / std::stod(GetEnv("COLL_BLOCK_SLICE_GRAIN"));
  }
  auto freq_to_slot = [this](float freq, IdType num_node){ return this->freq_to_slot_1(freq, num_node);};

  TensorPtr nid_to_freq_tensor = Tensor::Empty(kI32, {num_node}, cpu_ctx, "");
  CHECK(nid_to_block_tensor->Shape() == std::vector<size_t>{num_node});

  uint32_t* nid_to_block = nid_to_block_tensor->Ptr<uint32_t>();

  concurrent_full_slot_map slot_array_to_full_block;

  // const IdType *stream_id_ptr = stream_id_list->CPtr<IdType>();
  // const IdType *stream_freq_ptr = stream_freq_list->CPtr<IdType>();

  // identify freq boundary of first slot
  // when cache rate is extremely small, better use the largest val as alpha
  if (alpha < max_freq) {
    alpha = max_freq;
  }
  if (GetEnv("COLL_BLOCK_SLICE_BASE") != "") {
    RunConfig::coll_cache_coefficient = std::stod(GetEnv("COLL_BLOCK_SLICE_BASE"));
  }
  LOG(ERROR) << "reconfigured max freq to be " << alpha;
  RunConfig::coll_cache_num_slot = std::floor(std::log2(alpha) / std::log2(RunConfig::coll_cache_coefficient)) + 1;
  LOG(ERROR) << "reconfigured num slot to be " << RunConfig::coll_cache_num_slot;

  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */

  LOG(WARNING) << "counting slots...";
  size_t max_seq_slot = 0;
  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    uint32_t * nid_to_freq = nid_to_freq_tensor->Ptr<uint32_t>();
    std::unordered_map<size_t, full_slot_single_thread> the_map;
    std::uniform_int_distribution<IdType> dist(0, 9);
    std::mt19937 gen(omp_get_thread_num() + 0x789f);
    #pragma omp for
    for (uint32_t iter = 0; iter < num_node; iter++) {
      IdType nid = nid_iter(iter);
      IdType freq = freq_iter(iter);
      nid_to_freq[nid] = freq;
      size_t seq_slot_id = freq_to_slot(freq, num_node);
      if (freq == 0) {
        IdType rnd_rst = dist(gen);
        if (rnd_rst == 0) {rnd_rst = dist(gen);}
        else {rnd_rst += 9;}
        seq_slot_id += rnd_rst;
      }
      nid_to_block[nid] = seq_slot_id;
      {
        auto iter = the_map.find(seq_slot_id);
        if (iter == the_map.end()) {
          iter = the_map.insert({seq_slot_id, full_slot_single_thread()}).first;
          auto & val = iter->second;
          val.orig_seq_slot = seq_slot_id;
          val.size = 1;
        } else {
          iter->second.size++;
        }
      }
    }

    #pragma omp critical
    {
      for (auto local_iter = the_map.begin(); local_iter != the_map.end(); local_iter++) {
        auto global_iter = slot_array_to_full_block.the_map.find(local_iter->first);
        if (slot_array_to_full_block.the_map.find(local_iter->first) == slot_array_to_full_block.the_map.end()) {
          max_seq_slot = std::max(max_seq_slot, local_iter->first);
          global_iter = slot_array_to_full_block.the_map.insert({local_iter->first, local_iter->second}).first;
          global_iter->second.remmaped_slot = slot_array_to_full_block.__next_free_slot++;
        } else {
          global_iter->second.size += local_iter->second.size;
        }
      }
    }
  }
  slot_array_to_full_block.next_free_slot.store(slot_array_to_full_block.__next_free_slot);
  LOG(WARNING) << "Final num slot is " << slot_array_to_full_block.next_free_slot.load();
  block_identifer* buckets = new block_identifer[slot_array_to_full_block.next_free_slot.load()];
  next_free_block.store(0);

  LOG(WARNING) << "preparing block granularity...";
  for (auto iter = slot_array_to_full_block.the_map.begin(); iter != slot_array_to_full_block.the_map.end(); iter++) {
    buckets[iter->second.remmaped_slot]._total_nodes = iter->second.size;
  }

  // zero block
  size_t accumulate_size = 0;
  size_t accumulate_num_slice = 0;
  bool found_bound = false;
  for (size_t seq_slot_id = 0; seq_slot_id <= max_seq_slot; seq_slot_id++) {
    auto iter = slot_array_to_full_block.the_map.find(seq_slot_id);
    if (iter == slot_array_to_full_block.the_map.end()) continue;
    accumulate_size += iter->second.size;
    auto & bucket = buckets[iter->second.remmaped_slot];
    if (found_bound == false) {
      auto slice_size = std::min(max_size_per_block, RoundUpDiv<uint32_t>(iter->second.size, device_to_stream.size()));
      bucket.set_max_size(device_to_stream.size(), slice_size);
      bucket.num_slices = RoundUpDiv(iter->second.size, slice_size);
      bucket.num_slices = RoundUp<uint32_t>(bucket.num_slices, device_to_stream.size());
    } else {
      bucket.set_max_size(1, iter->second.size);
      bucket.num_slices = 1;
    }
    if (accumulate_size / (double)num_node >= RunConfig::cache_percentage * device_to_stream.size()) {
      found_bound = true;
    }
    bucket.slice_begin = accumulate_num_slice;
    accumulate_num_slice += bucket.num_slices;
    LOG(ERROR) << "slot " << iter->first << " has " << iter->second.size << " nodes, max_size set to " << buckets[iter->second.remmaped_slot].max_size_this_block
               << ", #slice=" << bucket.num_slices;
  }

  next_free_block.store(accumulate_num_slice);

  LOG(WARNING) << "counting blocks...";
  std::vector<std::uint32_t> seeds(RunConfig::solver_omp_thread_num);
  {
    std::seed_seq seq{1, 2, 3, 4, 5};
    seq.generate(seeds.begin(), seeds.end());
  }
  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    std::mt19937 gen(seeds[omp_get_thread_num()]);
    #pragma omp for
    for (uint32_t nid = 0; nid < num_node; nid++) {
      auto seq_slot_id = nid_to_block[nid];
      auto remapped_block_id = slot_array_to_full_block.the_map[seq_slot_id].remmaped_slot;
      auto &bucket = buckets[remapped_block_id];
      nid_to_block[nid] = std::uniform_int_distribution<uint32_t>(0, bucket.num_slices - 1)(gen) + bucket.slice_begin;
    }
  }
  delete[] buckets;

  uint32_t total_num_blocks = next_free_block.load();
  LOG(WARNING) << "Final num block is " << total_num_blocks;

  /**
   * Sum frequency & density of each block
   */
  LOG(WARNING) << "counting freq and density...";
  // block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "coll_cache.block_density_tensor");
  block_density_tensor = Tensor::CreateShm(this->_shm_name_dens, kF64, {total_num_blocks}, "");
  block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");

  std::memset(block_density_tensor->MutableData(), 0, block_density_tensor->NumBytes());
  std::memset(block_freq_tensor->MutableData(), 0, block_freq_tensor->NumBytes());

  // TensorView<double> block_density_array(block_density_tensor);
  // TensorView<double> block_freq_array(block_freq_tensor);
  double* block_density_array = block_density_tensor->Ptr<double>();
  double* block_freq_array    = block_freq_tensor->Ptr<double>();
  double min_freq = 5e-3;
  if (GetEnv("COLL_MIN_FREQ") != "") {
    min_freq = std::stod(GetEnv("COLL_MIN_FREQ"));
  }

  #pragma omp parallel num_threads(RunConfig::solver_omp_thread_num)
  {
    auto local_block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "");
    auto local_block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "");

    auto local_block_density = local_block_density_tensor->Ptr<double>();
    auto local_block_freq    = local_block_freq_tensor->Ptr<double>();

    std::memset(local_block_density_tensor->MutableData(), 0, local_block_density_tensor->NumBytes());
    std::memset(local_block_freq_tensor->MutableData(), 0, local_block_freq_tensor->NumBytes());

    const auto nid_to_freq = nid_to_freq_tensor->CPtr<uint32_t>();

    #pragma omp for
    for (uint32_t nid = 0; nid < num_node; nid++) {
      uint32_t block_id = nid_to_block[nid];
      local_block_density[block_id]++;
      double freq = nid_to_freq[nid];
      local_block_freq[block_id] += freq;
    }
    #pragma omp critical
    {
      for (uint32_t block_id = 0; block_id < total_num_blocks; block_id++) {
        block_density_array[block_id] += local_block_density[block_id];
        block_freq_array[block_id] += local_block_freq[block_id];
      }
    }
  }

  /**
   * Average the frequency for each block
   */
  LOG(WARNING) << "averaging freq and density...";
  LOG(WARNING) << block_density_tensor->NumItem();
// #pragma omp parallel for num_threads(RunConfig::solver_omp_thread_num)
  for (uint32_t block_id = 0; block_id < block_density_tensor->NumItem(); block_id++) {
    if (block_density_array[block_id] == 0) continue; 
    if (block_freq_array[block_id] == 0) {
      block_freq_array[block_id] = min_freq;
    }
    block_freq_array[block_id] /= block_density_array[block_id];
    block_density_array[block_id] *= 100/(double)num_node ;
    // std::cout << block_density_array[block_id].ref() << " ";
  }
  // std::cout << "\n";
  block_placement = Tensor::CreateShm(_shm_name_place, kU8, block_density_tensor->Shape(), "coll_cache_block_placement");
}

void OptimalSolver::BuildSingleStream(ContFreqBuf* freq_rank, std::vector<int> device_to_stream, const IdType num_node, const TensorPtr nid_to_block_tensor) {
  IdType num_stream = 1;
  {
    return BuildSingleStream(
      [](IdType nid){return nid; },
      [freq_rank](IdType nid){return freq_rank->get(nid); },
      [](IdType nid){ CHECK(false); return IdType(0); },
      freq_rank->buf[0].cnt, device_to_stream, num_node, nid_to_block_tensor);
  }
}

void OptimalSolver::Build(TensorPtr stream_id_list, TensorPtr stream_freq_list, std::vector<int> device_to_stream, const IdType num_node, const TensorPtr nid_to_block_tensor) {
  CHECK_EQ(stream_id_list->Shape().size(), 2);
  CHECK_EQ(stream_freq_list->Shape().size(), 2);
  CHECK_EQ(stream_id_list->Shape(), stream_freq_list->Shape());
  CHECK_EQ(stream_id_list->Shape()[1], num_node);
  IdType num_stream = stream_id_list->Shape()[0];
  if (num_stream == 1) {
    return BuildSingleStream(stream_id_list, stream_freq_list, device_to_stream, num_node, nid_to_block_tensor);
  }
  auto cpu_ctx = CPU(CPU_CLIB_MALLOC_DEVICE);
  // coarse-grained slice to reduce asymm solve time
  // symmetric & switch's precision still holds
  max_size_per_block = num_node / 1000;
  if (GetEnv("COLL_BLOCK_SLICE_GRAIN") != "") {
    max_size_per_block = num_node / std::stod(GetEnv("COLL_BLOCK_SLICE_GRAIN"));
  }
  auto freq_to_slot = [this](float freq, uint32_t rank, IdType num_node){ return this->freq_to_slot_1(freq, num_node);};

  TensorPtr nid_to_rank_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_rank");
  TensorPtr nid_to_slot_tensor  = Tensor::Empty(kI32, {num_node, num_stream}, cpu_ctx, "coll_cache.nid_to_slot");
  // nid_to_block_tensor = Tensor::Empty(kI32, {num_node}, cpu_ctx, "coll_cache.nid_to_block");
  CHECK(nid_to_block_tensor->Shape() == std::vector<size_t>{num_node});

  TensorView<uint32_t> nid_to_rank(nid_to_rank_tensor);
  TensorView<uint32_t> nid_to_slot(nid_to_slot_tensor);
  TensorView<uint32_t> nid_to_block(nid_to_block_tensor);

  concurrent_full_slot_map slot_array_to_full_block;

  TensorView<IdType> stream_id_list_view(stream_id_list);
  TensorView<IdType> stream_freq_list_view(stream_freq_list);

  // identify freq boundary of first slot
  // when cache rate is extremely small, better use the largest val as alpha
  for (IdType i = 0; i < num_stream; i++) {
    if (alpha < stream_freq_list_view[i][0].ref()) {
      alpha = stream_freq_list_view[i][0].ref();
    }
  }
  if (GetEnv("COLL_BLOCK_SLICE_BASE") != "") {
    RunConfig::coll_cache_coefficient = std::stod(GetEnv("COLL_BLOCK_SLICE_BASE"));
  }
  LOG(ERROR) << "reconfigured max freq to be " << alpha;
  RunConfig::coll_cache_num_slot = std::floor(std::log2(alpha) / std::log2(RunConfig::coll_cache_coefficient)) + 1;
  LOG(ERROR) << "reconfigured num slot to be " << RunConfig::coll_cache_num_slot;

  /**
   * Map each node to a rank for each stream.
   * Nodes with same rank for every stream forms a block.
   */
  LOG(WARNING) << "mapping nid to rank...";
  Timer t_nid_to_rank;
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t orig_rank = 0; orig_rank < num_node; orig_rank++) {
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t nid = stream_id_list_view[stream_idx][orig_rank].ref();
      nid_to_rank[nid][stream_idx].ref() = orig_rank;
    }
  }
  LOG(ERROR) << "mapping nid to rank takes " << t_nid_to_rank.Passed();
  auto slot_list_to_full_block_id = [num_stream](const TensorView<uint32_t>& slot_array){
    CHECK(*slot_array._shape == num_stream);
    size_t ret = 0;
    for (size_t i = 0; i < num_stream; i++) {
      ret *= num_stream;
      ret += slot_array._data[i];
    }
    return ret;
  };
  LOG(WARNING) << "counting slots...";
  size_t max_seq_slot = 0;
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(max: max_seq_slot)
  for (uint32_t nid = 0; nid < num_node; nid++) {
    // for each nid, prepare a slot list
    for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
      uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
      double freq = stream_freq_list_view[stream_idx][orig_rank].ref();
      int slot_id = freq_to_slot(freq, orig_rank, num_node);
      nid_to_slot[nid][stream_idx].ref() = slot_id;
    }
    // map the slot list to block
    size_t seq_id = slot_list_to_full_block_id(nid_to_slot[nid]);
    max_seq_slot = std::max(max_seq_slot, seq_id);
    nid_to_block[nid].ref() = slot_array_to_full_block.register_bucket(seq_id);
  }
  slot_array_to_full_block.next_free_slot.store(slot_array_to_full_block.__next_free_slot);
  LOG(WARNING) << "Final num slot is " << slot_array_to_full_block.next_free_slot.load();
  block_identifer* buckets = new block_identifer[slot_array_to_full_block.next_free_slot.load()];
  next_free_block.store(0);

  LOG(WARNING) << "preparing block granularity...";
  for (auto iter = slot_array_to_full_block.the_map.begin(); iter != slot_array_to_full_block.the_map.end(); iter++) {
    buckets[iter->second.remmaped_slot]._total_nodes = iter->second.size;
  }
  // #pragma omp parallel for num_threads(8)
  // for (uint32_t nid = 0; nid < num_node; nid++) {
  //   // fixme: the total number of each full size block has already be counted at slot_array_to_full_block.
  //   buckets[nid_to_block[nid].ref()].measure_total_node();
  // }
  #ifdef DEAD_CODE
  for (uint32_t bid = 0; bid < slot_array_to_full_block.next_free_slot.load(); bid++) {
    buckets[bid].set_max_size(device_to_stream.size(), max_size_per_block);
  }
  LOG(ERROR) << "block -1 has " << slot_array_to_full_block.the_map.find(max_seq_slot)->second.size << " nodes";
  if (max_seq_slot > 0) LOG(ERROR) << "block -2 has " << slot_array_to_full_block.the_map.find(max_seq_slot - 1)->second.size << " nodes";
  for (auto iter = slot_array_to_full_block.the_map.begin(); iter != slot_array_to_full_block.the_map.end(); iter++) {
    LOG(ERROR) << "slot " << iter->first << " has " << iter->second.size << " nodes";
  }
  #endif
  // zero block
  size_t accumulate_size = 0;
  for (size_t seq_slot_id = 0; seq_slot_id <= max_seq_slot; seq_slot_id++) {
    auto iter = slot_array_to_full_block.the_map.find(seq_slot_id);
    if (iter == slot_array_to_full_block.the_map.end()) continue;
    accumulate_size += iter->second.size;
    if (accumulate_size / (double)num_node < RunConfig::cache_percentage * device_to_stream.size()) {
      buckets[iter->second.remmaped_slot].set_max_size(device_to_stream.size(), std::min(max_size_per_block, RoundUpDiv<uint32_t>(iter->second.size, device_to_stream.size())));
    } else {
      buckets[iter->second.remmaped_slot].set_max_size(1, num_node);
    }
    LOG(ERROR) << "slot " << iter->first << " has " << iter->second.size << " nodes, max_size set to " << buckets[iter->second.remmaped_slot].max_size_this_block;
  }

  // if (slot_array_to_full_block.the_map.find(max_seq_slot)->second.size / (double)num_node < (1 - RunConfig::cache_percentage * device_to_stream.size())) {
  //   buckets[slot_array_to_full_block.the_map.find(max_seq_slot)->second.remmaped_slot].set_max_size(1, num_node);
  // } else {
  //   buckets[slot_array_to_full_block.the_map.find(max_seq_slot)->second.remmaped_slot].set_max_size(1, num_node / 100);
  // }
  // if (max_seq_slot > 0) {
  //   if (slot_array_to_full_block.the_map.find(max_seq_slot)->second.size + slot_array_to_full_block.the_map.find(max_seq_slot - 1)->second.size
  //     / (double)num_node < (1 - RunConfig::cache_percentage * device_to_stream.size())) {
  //     buckets[slot_array_to_full_block.the_map.find(max_seq_slot - 1)->second.remmaped_slot].set_max_size(1, num_node);
  //   } else {
  //     buckets[slot_array_to_full_block.the_map.find(max_seq_slot - 1)->second.remmaped_slot].set_max_size(1, num_node / 1000);
  //   }
  // }

  LOG(WARNING) << "counting blocks...";
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num / 2)
// #pragma omp parallel for num_threads(8)
  for (uint32_t nid = 0; nid < num_node; nid++) {
    block_identifer &bucket = buckets[nid_to_block[nid].ref()];
    nid_to_block[nid].ref() = buckets[nid_to_block[nid].ref()].add_node(this);
  }
  delete[] buckets;

  uint32_t total_num_blocks = next_free_block.load();
  LOG(WARNING) << "Final num block is " << total_num_blocks;

  /**
   * Sum frequency & density of each block
   */
  LOG(WARNING) << "counting freq and density...";
  // block_density_tensor = Tensor::Empty(kF64, {total_num_blocks}, cpu_ctx, "coll_cache.block_density_tensor");
  block_density_tensor = Tensor::CreateShm(_shm_name_dens, kF64, {total_num_blocks}, "");
  block_freq_tensor    = Tensor::Empty(kF64, {total_num_blocks, num_stream}, cpu_ctx, "coll_cache.block_freq_tensor");

  std::memset(block_density_tensor->MutableData(), 0, block_density_tensor->NumBytes());
  std::memset(block_freq_tensor->MutableData(), 0, block_freq_tensor->NumBytes());

  TensorView<double> block_density_array(block_density_tensor);
  TensorView<double> block_freq_array(block_freq_tensor);
  double min_freq = 1e-2;
  if (GetEnv("COLL_MIN_FREQ") != "") {
    min_freq = std::stod(GetEnv("COLL_MIN_FREQ"));
  }
#pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (int thread_idx = 0; thread_idx < RunConfig::omp_thread_num; thread_idx++) {
    for (uint32_t nid = 0; nid < num_node; nid++) {
      uint32_t block_id = nid_to_block[nid].ref();
      if (std::hash<uint64_t>()(block_id) % RunConfig::omp_thread_num != thread_idx) {
        continue;
      }
      block_density_array[block_id].ref() += 1;
      for (uint32_t stream_idx = 0; stream_idx < num_stream; stream_idx++) {
        uint32_t orig_rank = nid_to_rank[nid][stream_idx].ref();
        double freq = stream_freq_list_view[stream_idx][orig_rank].ref();
        // assign all zero freq a minimal freq to handle touched node << cache space
        freq = std::max(freq, min_freq);
        block_freq_array[block_id][stream_idx].ref() += freq;
      }
    }
  }

  /**
   * Average the frequency for each block
   */
  LOG(WARNING) << "averaging freq and density...";
  LOG(WARNING) << block_density_tensor->NumItem();
// #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  for (uint32_t block_id = 0; block_id < block_density_tensor->NumItem(); block_id++) {
    if (block_density_array[block_id].ref() == 0) continue; 
    for (uint32_t stream_id = 0; stream_id < num_stream; stream_id++) {
      block_freq_array[{block_id,stream_id}].ref() /= block_density_array[block_id].ref() ;
    }
    block_density_array[block_id].ref() *= 100/(double)num_node ;
    // std::cout << block_density_array[block_id].ref() << " ";
  }
  // std::cout << "\n";
  block_placement = Tensor::CreateShm(_shm_name_place, kU8, block_density_tensor->Shape(), "coll_cache_block_placement");
}

void OptimalSolver::Solve(std::vector<int> device_to_stream, std::vector<PerT> device_to_cache_percent, std::string mode, double T_local, double T_remote, double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "OptimalSolver does not support asymm link topo";
  CHECK(mode == "BIN");
  CHECK(block_density_tensor->Defined());
  double* block_density_array = block_density_tensor->Ptr<double>();
  TensorView<double> block_freq_array(block_freq_tensor);
  PerT cache_percent   = device_to_cache_percent.at(0);
  uint32_t num_device = device_to_stream.size();
  IdType num_stream   = block_freq_tensor->Shape().at(1);
  uint32_t num_block  = block_density_tensor->Shape().at(0);
  LOG(INFO) << "constructing optimal solver, device=" << num_device << ", stream=" << num_stream;

  std::cerr << num_block << " blocks, " << num_device << " devices\n";

  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "cppsolver.log");
  // env.set(GRB_IntParam_Presolve, 0);
  // // env.set(GRB_IntParam_Method, 2);
  // // env.set(GRB_IntParam_Method, -1);
  // env.set(GRB_IntParam_Method, 3);
  // env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num);
  // env.set(GRB_DoubleParam_BarConvTol, 1e-3);
  // env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
  // env.set(GRB_DoubleParam_MIPGap, 2e-3);


  env.set(GRB_IntParam_ConcurrentMIP, 50);
  // env.set(GRB_DoubleParam_Heuristics, 0.5);
  // env.set(GRB_DoubleParam_NoRelHeurTime, 10);
  env.set(GRB_IntParam_Presolve, 1);
  env.set(GRB_IntParam_MIPFocus, 1);
  // env.set(GRB_IntParam_Crossover, 0);
  // env.set(GRB_IntParam_Presolve, -1);
  env.set(GRB_IntParam_Method, 2);
  // env.set(GRB_IntParam_NodeMethod, 2);
  // env.set(GRB_IntParam_Method, -1);
  // env.set(GRB_IntParam_Method, 3);
  env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num*2);
  env.set(GRB_DoubleParam_BarConvTol, 1e-4);
  env.set(GRB_DoubleParam_OptimalityTol, 2e-2);
  env.set(GRB_DoubleParam_MIPGap, 2e-2);

  env.start();

  GRBModel model = GRBModel(env);

  GRBVar z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");
  TensorPtr c_list_tensor = Tensor::Empty(kI64, {num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");
  TensorPtr x_list_tensor = Tensor::Empty(kI64, {num_block, num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "x_list");

  TensorView<GRBVar> c_list(c_list_tensor);
  TensorView<GRBVar> x_list(x_list_tensor);
  std::vector<GRBLinExpr> time_list(num_device);
  std::vector<GRBLinExpr> local_weight_list(num_device);
  std::vector<double>     total_weight_list(num_device);
  std::vector<GRBLinExpr> cpu_weight_list(num_device);

  auto constraint_connect_c_x = [&](GRBModel & model, uint32_t block_id) {
    if (block_density_array[block_id] == 0) return;
    GRBLinExpr expr;
    expr += c_list[block_id].ref();
    FOR_LOOP(device_id, num_device) {
      expr += x_list[block_id][device_id].ref();
    }
    model.addConstr(expr >= 1);
    
    if (mode == "BIN") {
      GRBLinExpr expr;
      expr += c_list[block_id].ref() * num_device;
      FOR_LOOP(device_id, num_device) {
        expr += x_list[block_id][device_id].ref();
      }
      model.addConstr(expr <= num_device);
    }
  };
  auto  constraint_connect_r_x = [&](GRBModel & model, uint32_t block_id, uint32_t device_id) {
    if (mode == "CONT") {
      if (block_density_array[block_id] == 0) return;
      model.addConstr(c_list[block_id].ref() + x_list[block_id][device_id].ref() <= 1);
    }
  };
  auto constraint_capacity = [&](GRBModel & model, uint32_t device_id) {
    GRBLinExpr expr;
    FOR_LOOP(block_id, num_block) {
      if (block_density_array[block_id] == 0) continue;
      expr += x_list[block_id][device_id].ref() * block_density_array[block_id];
    }
    model.addConstr(expr <= cache_percent);
  };
  auto constraint_time = [&](GRBModel & model, uint32_t device_id) {
    uint32_t stream_id = device_to_stream[device_id];
    double sum_weight = 0;
    GRBLinExpr &expr = time_list[device_id];
    FOR_LOOP(block_id, num_block) {
      double weight = block_density_array[block_id] * block_freq_array[block_id][stream_id].ref();
      if (weight == 0) continue;
      sum_weight += weight;
      expr += c_list[block_id].ref() * (weight * (T_cpu - T_remote)) - x_list[block_id][device_id].ref() * (weight * (T_remote - T_local));

      local_weight_list[device_id] +=  weight * x_list[block_id][device_id].ref();
      cpu_weight_list[device_id]   +=  weight * c_list[block_id].ref();
    }
    expr += sum_weight * T_remote;
    total_weight_list[device_id] = sum_weight;
    model.addConstr(expr <= z, "time_" + std::to_string(device_id));
  };

  LOG(INFO) << "Add Var...";
  char var_type = (mode == "BIN") ? GRB_BINARY : GRB_CONTINUOUS;
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, block_density_array[block_id])) {
      continue;
    }
    c_list[block_id].ref() = model.addVar(0, 1, 0, var_type);
    FOR_LOOP(device_id, num_device) {
      x_list[block_id][device_id].ref() = model.addVar(0, 1, 0, var_type);
    }
  }

  LOG(INFO) << "Capacity...";
  FOR_LOOP(device_id, num_device) {constraint_capacity(model, device_id);}

  LOG(INFO) << "Connect CPU...";
  FOR_LOOP(block_id, num_block) {constraint_connect_c_x(model, block_id);}

  LOG(INFO) << "Connect Remote...";
  FOR_LOOP(block_id, num_block) {
    FOR_LOOP(device_id, num_device) {constraint_connect_r_x(model, block_id, device_id);}
  }

  LOG(INFO) << "Time...";
  FOR_LOOP(device_id, num_device) {constraint_time(model, device_id);}

  model.setObjective(z + 0, GRB_MINIMIZE);

  model.optimize();

  CHECK(num_device <= 8);
  CHECK(block_placement->Shape() == std::vector<size_t>{num_block});
  TensorView<uint8_t> block_placement_array(block_placement);
  LOG(INFO) << "Coll Cache init block placement array";
  // #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
  FOR_LOOP(block_id, num_block) {
    block_placement_array[block_id].ref() = 0;
    // std::ios_base::fmtflags f( std::cerr.flags() );
    // std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref();
    if (!ignore_block(block_id, block_density_array[block_id])) {
      FOR_LOOP(device_id, num_device) {
        uint8_t x_result = (uint8_t)std::round(x_list[block_id][device_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X));
        block_placement_array[block_id].ref() |= (x_result << device_id);
      }
    }
    std::bitset<8> bs(block_placement_array[block_id].ref());
    // std::cerr << "  storage is " << bs << "\n";
    // std::cerr.flags(f);
  }
  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << local_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << 1 - (local_weight_list[part_id].getValue() + cpu_weight_list[part_id].getValue()) / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << cpu_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "z=" << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X) << "\n";
  LOG(INFO) << "Coll Cache init block placement array done";
  model.reset(1);
  LOG(INFO) << "Coll Cache model reset done";
}

void OptimalAsymmLinkSolver::Solve(std::vector<int> device_to_stream, std::vector<PerT> device_to_cache_percent, std::string mode, double T_local, double T_cpu) {
  CHECK(mode == "BIN");
  CHECK(block_density_tensor->Defined());
  double*            block_density_array = block_density_tensor->Ptr<double>();
  TensorView<double> block_freq_array(block_freq_tensor);
  PerT      cache_percent  = device_to_cache_percent.at(0);
  uint32_t num_device     = device_to_stream.size();
  IdType   num_stream     = block_freq_tensor->Shape().at(1);
  uint32_t num_block      = block_density_tensor->Shape().at(0);
  uint32_t num_link       = link_src[0].size(); // we simply assume all dst device have same num link
  LOG(WARNING) << "constructing optimal solver, device=" << num_device << ", stream=" << num_stream;

  std::cerr << num_block << " blocks, " << num_device << " devices\n";

  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "cppsolver.log");
  // env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num*2);
  env.set(GRB_IntParam_Threads, RunConfig::solver_omp_thread_num);

  // env.set(GRB_IntParam_LogToConsole, 0);

  env.set(GRB_DoubleParam_TimeLimit, 200);

  // // old parameters for cpu, then concurrent remote, then local
  // env.set(GRB_DoubleParam_BarConvTol, 1e-4);
  // env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
  // env.set(GRB_IntParam_Method, 3);
  // env.set(GRB_DoubleParam_MIPGap, 0.03);
  // env.set(GRB_IntParam_MIPFocus, 2);
  // env.set(GRB_IntParam_ConcurrentMIP, 36);
  // env.set(GRB_IntParam_ConcurrentMIP, 10);
  // env.set(GRB_IntParam_BranchDir, 1);
  // env.set(GRB_IntParam_AggFill, 100);
  // env.set(GRB_IntParam_NormAdjust, 3);
  // env.set(GRB_IntParam_Presolve, 2);
  // env.set(GRB_IntParam_SimplexPricing, 2);
  // env.set(GRB_IntParam_DegenMoves, 0);
  // env.set(GRB_IntParam_CutPasses, 5);
  // env.set(GRB_IntParam_PrePasses, 8);
  // env.set(GRB_DoubleParam_Heuristics, 0.001);
  // env.set(GRB_IntParam_ScaleFlag, 0);
  // env.set(GRB_IntParam_StrongCGCuts, 0);
  // env.set(GRB_IntParam_MIRCuts, 1);
  // env.set(GRB_IntParam_Cuts, 3);

  // new parameters for [           cpu          ]
  //                    [local][cuncurrent remote]
  env.set(GRB_DoubleParam_MIPGap, 0.05);
  // env.set(GRB_IntParam_Presolve, 2);
  // env.set(GRB_IntParam_AggFill, 100);
  // env.set(GRB_IntParam_Aggregate, 0);
  // env.set(GRB_IntParam_GomoryPasses, 0);
  /** todo: the following param seems hurt performance. needs investigation */
  // env.set(GRB_IntParam_Method, 2);
  // env.set(GRB_IntParam_DegenMoves, 0);
  // env.set(GRB_IntParam_PrePasses, 5);
  // env.set(GRB_IntParam_NormAdjust, 0);
  // env.set(GRB_IntParam_FlowCoverCuts, 2);

  if (GetEnv("COLL_GUROBI_EXP_PARAM") == "1") {
    env.set(GRB_IntParam_Cuts, 2);
    env.set(GRB_IntParam_DegenMoves, 2);
    env.set(GRB_IntParam_ScaleFlag, 1);
  }

  env.start();

  GRBModel model = GRBModel(env);

  /**
   * z: final goal, max time of each dst gpu, minimize it
   * x: storage of each block on each src gpu
   * a: access flag of each block: a[block][dst][src] means whether dst read this block from src
   */
  GRBVar z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");
  TensorPtr c_list_tensor = Tensor::Empty(kI64, {num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");
  TensorPtr x_list_tensor = Tensor::Empty(kI64, {num_block, num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "x_list");
  TensorPtr a_list_tensor = Tensor::Empty(kI64, {num_block, num_device, num_link}, CPU(CPU_CLIB_MALLOC_DEVICE), "a_list");
  TensorPtr max_remote_time_tensor = Tensor::Empty(kI64, {num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");

  TensorView<GRBVar> c_list(c_list_tensor);
  TensorView<GRBVar> x_list(x_list_tensor);
  TensorView<GRBVar> a_list(a_list_tensor);
  TensorView<GRBVar> max_remote_time(max_remote_time_tensor);

  // std::vector<GRBLinExpr> time_list(num_device);
  // std::vector<GRBLinExpr> local_cpu_time_list(num_device);
  std::vector<GRBLinExpr> cpu_time_list(num_device);
  std::vector<GRBLinExpr> local_time_list(num_device);
  vec<vec<GRBLinExpr>>    remote_time_list(num_device, vec<GRBLinExpr>(num_link));
  std::vector<GRBLinExpr> local_weight_list(num_device);
  std::vector<double>     total_weight_list(num_device);
  std::vector<GRBLinExpr> cpu_weight_list(num_device);
  try {

  // for each dst, 
  // sum a_dst_src <= 1
  // src
  auto constraint_connect_a_c = [&](GRBModel & model, uint32_t block_id) {
    if (block_density_array[block_id] == 0) return;
    FOR_LOOP(dst_dev, num_device) {
      GRBLinExpr expr;
      FOR_LOOP(src_link, num_link) {
        expr += a_list[block_id][dst_dev][src_link].ref();
      }
      model.addConstr(expr + c_list[block_id].ref() + x_list[block_id][dst_dev].ref() == 1);
    }
  };

  // for each src,
  // x_src >= max(a_dst_src)
  //          dst
  auto constraint_connect_a_x = [&](GRBModel & model, uint32_t block_id) {
    if (block_density_array[block_id] == 0) return;
    FOR_LOOP(dst_dev, num_device) {
      FOR_LOOP(src_link, num_link) {
        GRBLinExpr expr;
        for (auto src_dev : link_src[dst_dev][src_link]) {
          expr += x_list[block_id][src_dev].ref();
        }
        model.addConstr(expr >= a_list[block_id][dst_dev][src_link].ref());
      }
    }
    /** try reduce num of constr. but seems results in longer solve time*/
    // FOR_LOOP(src_dev, num_device) {
    //   GRBLinExpr expr;
    //   FOR_LOOP(dst_dev, num_device) {
    //     FOR_LOOP(src_link, num_link) {
    //       CHECK(link_src[dst_dev][src_link].size() == 1);
    //       if (link_src[dst_dev][src_link][0] != src_dev) continue;
    //       expr += a_list[block_id][dst_dev][src_link].ref();
    //     }
    //   }
    //   model.addConstr(x_list[block_id][src_dev].ref() * num_device >= expr);
    // }
  };

  // for each src,
  //  sum x_src < cache size
  // block
  auto constraint_capacity = [&](GRBModel & model, uint32_t src_dev) {
    GRBLinExpr expr;
    FOR_LOOP(block_id, num_block) {
      if (block_density_array[block_id] == 0) continue;
      expr += x_list[block_id][src_dev].ref() * block_density_array[block_id];
    }
    model.addConstr(expr <= cache_percent);
  };
  auto constraint_time = [&](GRBModel & model, uint32_t dst_dev) {
    uint32_t stream_id = device_to_stream[dst_dev];
    double sum_weight = 0;
    // GRBLinExpr &local_cpu_time = local_cpu_time_list[dst_dev];
    GRBLinExpr &cpu_time = cpu_time_list[dst_dev];
    GRBLinExpr &local_time = local_time_list[dst_dev];
    FOR_LOOP(block_id, num_block) {
      double weight = block_density_array[block_id] * block_freq_array[block_id][stream_id].ref();
      if (weight == 0) continue;
      sum_weight += weight;
      cpu_time += c_list[block_id].ref() * T_cpu * weight;
      local_time += x_list[block_id][dst_dev].ref() * T_local * weight;

      local_weight_list[dst_dev] +=  weight * x_list[block_id][dst_dev].ref();
      cpu_weight_list[dst_dev]   +=  weight * c_list[block_id].ref();
    }
    GRBLinExpr total_time;
    FOR_LOOP(src_link, num_link) {
      GRBLinExpr &remote_time = remote_time_list[dst_dev][src_link];
      FOR_LOOP(block_id, num_block) {
        double weight = block_density_array[block_id] * block_freq_array[block_id][stream_id].ref();
        if (weight == 0) continue;
        remote_time += a_list[block_id][dst_dev][src_link].ref() * link_time[dst_dev][src_link] * weight;
      }
      if (true) {
      // if (RunConfig::concurrent_link_impl == kMPSPhase ||
      //     RunConfig::concurrent_link_impl == kSMMaskPhase) {
        model.addConstr(remote_time <= z);
        total_time += remote_time * RunConfig::coll_cache_link_desc.link_sm[dst_dev][src_link];
      } else {
        model.addConstr(remote_time <= max_remote_time[dst_dev].ref());
      }
      // model.addConstr(remote_time + local_cpu_time <= z);
    }
    if (true) {
    // if (RunConfig::concurrent_link_impl == kMPSPhase ||
    //     RunConfig::concurrent_link_impl == kSMMaskPhase) {
      model.addConstr(local_time <= z);
      // local sm + cpu sm is always total sm
      int total_sm = RunConfig::coll_cache_link_desc.local_sm[dst_dev] + RunConfig::coll_cache_link_desc.cpu_sm[dst_dev];
      LOG(ERROR) << "dst " << dst_dev << ", total sm is " << total_sm;
      total_time += local_time * RunConfig::coll_cache_link_desc.local_sm[dst_dev];
      total_time += (cpu_time + sum_weight * RunConfig::coll_cache_cpu_addup * T_cpu) * RunConfig::coll_cache_link_desc.cpu_sm[dst_dev];
      model.addConstr(total_time <= z * total_sm);
    } else {
      model.addConstr(max_remote_time[dst_dev].ref() + local_time <= z);
    }
    model.addConstr(cpu_time + sum_weight * RunConfig::coll_cache_cpu_addup * T_cpu <= z);
    total_weight_list[dst_dev] = sum_weight;
  };

  LOG(WARNING) << "Add Var...";
  char var_type = (mode == "BIN") ? GRB_BINARY : GRB_CONTINUOUS;
  FOR_LOOP(block_id, num_block) {
    if (ignore_block(block_id, block_density_array[block_id])) {
      continue;
    }
    c_list[block_id].ref() = model.addVar(0, 1, 0, var_type);
    FOR_LOOP(dst_dev, num_device) {
      x_list[block_id][dst_dev].ref() = model.addVar(0, 1, 0, var_type);
      FOR_LOOP(src_link, num_link) {
        a_list[block_id][dst_dev][src_link].ref() = model.addVar(0, 1, 0, var_type);
      }
    }
  }
  FOR_LOOP(dst_dev, num_device) {
    max_remote_time[dst_dev].ref() = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS);
  }

  LOG(WARNING) << "Capacity...";
  FOR_LOOP(device_id, num_device) {constraint_capacity(model, device_id);}

  LOG(WARNING) << "Connect CPU...";
  FOR_LOOP(block_id, num_block) {constraint_connect_a_c(model, block_id);}
  LOG(WARNING) << "Connect Access To Storage...";
  FOR_LOOP(block_id, num_block) {
    FOR_LOOP(device_id, num_device) {constraint_connect_a_x(model, block_id);}
  }
  LOG(WARNING) << "Time...";
  FOR_LOOP(device_id, num_device) {constraint_time(model, device_id);}

  model.setObjective(z + 0, GRB_MINIMIZE);

  model.write("asymm.lp");

  if (GetEnv("SAVE_MILP_ONLY") != "") {
    std::string cmd = "mv asymm.lp " + GetEnv("COLL_LOG_BASE") + ".lp";
    system(cmd.c_str());
    abort();
  }

  model.optimize();
  } catch (GRBException e) {
    LOG(FATAL) << e.getMessage();
    abort();
  }

  CHECK(num_device <= 8);
  CHECK(block_placement->Shape() == std::vector<size_t>{num_block});
  // num_link + local + cpu always <= 8
  block_access_from = Tensor::CreateShm(_shm_name_access, kU8, {num_device, num_block}, "coll_cache_block_access");
  TensorView<uint8_t> block_placement_array(block_placement);
  TensorView<uint8_t> block_access_from_array(block_access_from);
  LOG(WARNING) << "Coll Cache init block placement array";
  // #pragma omp parallel for num_threads(RunConfig::omp_thread_num)

  auto get_val = [](GRBVar & var) { return var.get(GRB_DoubleAttr::GRB_DoubleAttr_X);};
  auto get_int = [](GRBVar & var) { return std::round(var.get(GRB_DoubleAttr::GRB_DoubleAttr_X));};

  vec<vec<uint8_t>> link_bitmap(num_device, vec<uint8_t>(num_link, 0));
  for (uint32_t dev_id = 0; dev_id < num_device; dev_id++) {
    for (uint32_t link = 0; link < num_link; link++) {
      uint8_t & bitmap = link_bitmap[dev_id][link];
      for (auto src_dev : this->link_src[dev_id][link]) {
        bitmap |= 1 << src_dev;
      }
    }
  }

  vec<uint8_t> bitmap_to_src_dev(1 << num_device, 0);
  PreDecideSrc(num_device, num_device + 1, bitmap_to_src_dev.data());

  FOR_LOOP(block_id, num_block) {
    // by default, this block is placed at cpu
    block_placement_array[block_id].ref() = 0;
    FOR_LOOP(device_id, num_device) {
      block_access_from_array[device_id][block_id].ref() = num_device; // num_device is treat as cpu
    }

    // std::ios_base::fmtflags f( std::cerr.flags() );
    // std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref();

    // x == 1 -> access from = local dev id
    // x == 0, a != 0 -> <bitmap of this link> & <storage bitmap>, then choose one from it
    // x == 0, a == 0 -> cpu
    if (ignore_block(block_id, block_density_array[block_id])) {
      continue;
    }
    FOR_LOOP(device_id, num_device) {
      uint8_t x_result = (uint8_t)std::round(x_list[block_id][device_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X));
      block_placement_array[block_id].ref() |= (x_result << device_id);
    }
    // build access from
    FOR_LOOP(device_id, num_device) {
      if (get_int(x_list[block_id][device_id].ref())) {
        block_access_from_array[device_id][block_id].ref() = device_id;
        continue;
      }
      if (get_int(c_list[block_id].ref())) {
        block_access_from_array[device_id][block_id].ref() = num_device; // num_device is treat as cpu
        continue;
      }
      for (uint32_t src_link = 0; src_link < num_link; src_link ++) {
        if (get_int(a_list[block_id][device_id][src_link].ref()) == 0) continue;
        const uint8_t link_src_bitmap = link_bitmap[device_id][src_link];
        const uint8_t storage_bit_map = block_placement_array[block_id].ref();
        const uint8_t candidate_bit_map = link_src_bitmap & storage_bit_map;
        block_access_from_array[device_id][block_id].ref() = bitmap_to_src_dev[candidate_bit_map];
        break;
      }
    }
    // std::bitset<8> bs(block_placement_array[block_id].ref());
    // std::cerr << "  storage is " << bs << "\n";
    // std::cerr.flags(f);
  }
  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << local_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << 1 - (local_weight_list[part_id].getValue() + cpu_weight_list[part_id].getValue()) / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << cpu_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "z=" << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X) << "\n";
  LOG(WARNING) << "Coll Cache init block placement array done";
  model.reset(1);
  LOG(WARNING) << "Coll Cache model reset done";
}
void OptimalAsymmLinkSolver::PreDecideSrc(int num_bits,
                                          int cpu_location_id,
                                          uint8_t *placement_to_src) {
  auto g = std::mt19937(RunConfig::seed);
  // auto g = std::default_random_engine(
  //     std::chrono::system_clock::now().time_since_epoch().count());
  for (int placement = 0; placement < (1 << num_bits); placement++) {
    if (placement == 0) {
      placement_to_src[placement] = cpu_location_id;
    } else {
      int num_nz = cpu::CountBits(placement);
      int location = 0;
      // scan all 1, and choose current 1 uniformly
      for (; location < num_bits; location++) {
        if ((placement & (1 << location)) == 0) continue;
        int choice = std::uniform_int_distribution<int>(1, num_nz)(g);
        if (choice == 1) {
          break;
        }
        num_nz--;
      }
      placement_to_src[placement] = location;
    }
  }
}

void SingleStreamSolverBase::Build(TensorPtr stream_id_list,
                                   TensorPtr stream_freq_list,
                                   std::vector<int> device_to_stream,
                                   const IdType num_node,
                                   const TensorPtr nid_to_block_tensor) {
  this->stream_id_list = stream_id_list;
  this->stream_freq_list = stream_freq_list;
  this->nid_to_block = nid_to_block_tensor;
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
}
void IntuitiveSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<PerT> device_to_cache_percent,
                            std::string mode, double T_local, double T_cpu) {
  // CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "IntuitiveSolver does not support asymm link topo";
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 2;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);
  IdType partition_size_min = 0;
  IdType partition_size_max = (num_device == 1) ? 0 : std::min(num_cached_nodes, (num_node - num_cached_nodes) / (num_device - 1));

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  LOG(ERROR) << "[" << partition_size_min << "," << partition_size_max << "]";

  // IdType partition_size = 0;

  auto partition_lb = [&](IdType partition_size) {
    return num_cached_nodes - partition_size;
  };
  auto partition_rb = [&](IdType partition_size) {
    return std::max<IdType>(num_cached_nodes + partition_size * (num_device - 1), 1) - 1;
  };

  // const double T_partition = (T_local + (num_device - 1) * T_remote) / num_device;
  double T_remote = 0;
  if (RunConfig::coll_cache_link_desc._topo_type == AsymmLinkDesc::kHardWiredAsymm) {
    if (RunConfig::num_device != 8) {
      CHECK(false) << "unimplemented intuitive solver on asymm hardwired platform with <8 GPUs";
    }
    // asumming 8V100 platform
    T_remote = (6 * T_remote + 3 * T_cpu) / 7;
  } else {
    T_remote = RunConfig::coll_cache_link_desc.AggregatedRemoteTime();
  }
  const double mu = 1 + (T_cpu - T_remote) / (T_remote - T_local) * num_device;
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  LOG(ERROR) << "mu = " << mu;

  /**
   * |   replicate  |   p    |  p * (n_d-1) | cpu |
   *        ^           ^    ^
   *       >mu     mu  <mu   1
   *       max             min
   */

  IdType partition_size = partition_size_min;

  double min_freq = 0;
  if (GetEnv("COLL_INTUITIVE_MIN_FREQ") != "") {
    min_freq = std::stod(GetEnv("COLL_INTUITIVE_MIN_FREQ"));
  }

  if (mu < 1) {
    // the best choice is to replicate as much as possible. no partition
    partition_size = partition_size_min;
  } else if (freq_array[partition_lb(partition_size_max)] < std::max<double>(min_freq, freq_array[partition_rb(partition_size_max)]) * mu) {
    // we have to choose largest partition
    // build the mapping
    partition_size = partition_size_max;
  } else {
    // now we need to iterate from min to max to find the mu
    while (partition_size_max - partition_size_min > 1) {
      partition_size = (partition_size_min + partition_size_max) / 2;
      if (freq_array[partition_lb(partition_size)] < std::max<double>(min_freq, freq_array[partition_rb(partition_size)]) * mu) {
        // go left
        partition_size_min = partition_size;
      } else {
        // go right
        partition_size_max = partition_size;
      }
      LOG(DEBUG) << "[" << partition_size_min << "," << partition_size_max << "]";
    }
    partition_size = partition_size_max;
  }
  
  double rep_w = 0, cpu_w = 0, total_w = 0;
  IdType rep_d = 0, cpu_d = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w, rep_d, cpu_d)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < partition_lb(partition_size)) {
      // replicate this
      nid_to_block->Ptr<IdType>()[node_id] = 0;
      rep_w += freq_array[rank];
      rep_d ++;
    } else if (rank <= partition_rb(partition_size)) {
      nid_to_block->Ptr<IdType>()[node_id] = (rank % num_device) + 1;
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = num_device + 1;
      cpu_w += freq_array[rank];
      cpu_d++;
    }
    total_w += freq_array[rank];
  }
  double local_w = rep_w + (total_w - cpu_w - rep_w) / num_device;
  double remote_w = (total_w - cpu_w - rep_w) / num_device * (num_device - 1);
  std::cout << "coll_cache:optimal_rep_storage=" << partition_lb(partition_size) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << (partition_rb(partition_size) - partition_lb(partition_size)) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_rb(partition_size) / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << (partition_lb(partition_size) + partition_size) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (num_device - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";
  {
    // estimate time using coll extract mechanism
    auto & desc = RunConfig::coll_cache_link_desc;
    double z = 0;
    double actual_cpu_w = total_w - local_w;
    double per_partition_w = remote_w / (num_device - 1);
    z = std::max(z, local_w * T_local * 100 / num_node);
    double total_area = 0;
    total_area += local_w * T_local * 100 / num_node * desc.local_sm[0];
    for (int link_id = 0; link_id < desc.link_sm[0].size(); link_id++) {
      actual_cpu_w -= per_partition_w;
      z = std::max(z, per_partition_w * desc.link_time[0][link_id] * 100 / num_node);
      total_area += per_partition_w * desc.link_time[0][link_id] * 100 / num_node * desc.link_sm[0][link_id];
    }
    z = std::max(z, actual_cpu_w * T_cpu * 100 / num_node);
    total_area += actual_cpu_w * T_cpu * 100 / num_node * desc.cpu_sm[0];
    z = std::max(z, total_area / (desc.local_sm[0] + desc.cpu_sm[0]));
    std::cout << "z.mps_phase=" << z << "\n";
  }

  block_placement = Tensor::CreateShm(_shm_name_place, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i + 1] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device + 1] = 0;

  block_access_from = Tensor::CreateShm(_shm_name_access, kU8, {(size_t)num_device, static_cast<size_t>(num_block)}, "coll_cache_advise");
  TensorView<uint8_t> block_access_from_array(block_access_from);
  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    block_access_from_array[dev_id][0].ref() = dev_id; // block 0 is replicated
    for (int block_id = 1; block_id <= num_device; block_id++) {
      block_access_from_array[dev_id][block_id].ref() = block_id - 1; // block 1~n partitioned: gpu 0~n-1
    }
    block_access_from_array[dev_id][num_device + 1].ref() = num_device; // cpu
  }
  block_density_tensor = Tensor::CreateShm(_shm_name_dens, kF64, {static_cast<unsigned long>(num_block)}, "");
  block_density_tensor->Ptr<double>()[0] = (double)partition_lb(partition_size) * 100 / (double)num_node;
  for (int block_id = 1; block_id <= num_device; block_id++) {
    block_density_tensor->Ptr<double>()[block_id] = partition_size *(double)100 / (double)num_node;
  }
  block_density_tensor->Ptr<double>()[num_device + 1] = 100 - (partition_rb(partition_size) / (double)num_node) * 100;
}

void PartitionSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<PerT> device_to_cache_percent,
                            std::string mode, double T_local, double T_remote,
                            double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "PartitionSolver does not support asymm link topo";
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 1;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  // LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  CHECK_EQ(stream_freq_list->Type(), kI32);
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType partition_size = std::min(num_cached_nodes, num_node / num_device);

  double cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < partition_size * num_device) {
      nid_to_block->Ptr<IdType>()[node_id] = rank % num_device;
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = num_device;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
  }
  double local_w = (total_w - cpu_w) / num_device;
  double remote_w = (total_w - cpu_w) / num_device * (num_device - 1);
  std::cout << "coll_cache:optimal_rep_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * num_device / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_size * num_device / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << partition_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (num_device - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm(_shm_name_place, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device] = 0;
}

void PartRepSolver::Solve(std::vector<int> device_to_stream,
                            std::vector<PerT> device_to_cache_percent,
                            std::string mode, double T_local, double T_remote,
                            double T_cpu) {
  CHECK(RunConfig::coll_cache_link_desc._topo_type != AsymmLinkDesc::kHardWiredAsymm) << "PartRepSolver does not support asymm link topo";
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = num_device + 2;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType partition_size = (num_device == 1) ? num_cached_nodes : std::min(num_cached_nodes, (num_node - num_cached_nodes)/(num_device-1));
  const IdType replicate_size = num_cached_nodes - partition_size;
  CHECK_LE(replicate_size + partition_size * num_device, num_node);
  const IdType cpu_size = num_node - replicate_size - partition_size * num_device;

  double rep_w = 0, cpu_w = 0, total_w = 0;
  // block 0 -> replication
  // block 1-n -> partition,
  // block n+1 -> cpu
#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < replicate_size) {
      nid_to_block->Ptr<IdType>()[node_id] = 0;
      rep_w += freq_array[rank];
    } else if (rank < replicate_size + partition_size * num_device) {
      nid_to_block->Ptr<IdType>()[node_id] = (rank % num_device) + 1;
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = num_device + 1;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
  }
  double partition_w = total_w - cpu_w - rep_w;
  double local_w = rep_w + partition_w / num_device;
  double remote_w = partition_w / num_device * (num_device - 1);
  std::cout << "coll_cache:optimal_rep_storage=" << replicate_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * num_device / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << cpu_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << (replicate_size + partition_size) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (num_device - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + remote_w * 100 / num_node * T_remote + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm(_shm_name_place, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");

  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  for (int i = 0; i < num_device; i++) {
    block_placement->Ptr<uint8_t>()[i + 1] = (1 << i);
  }
  block_placement->Ptr<uint8_t>()[num_device + 1] = 0;
}

void CollFineGrainSolver::Solve(std::vector<int> device_to_stream, std::vector<PerT> device_to_cache_percent, std::string mode, double T_local, double T_cpu) {
  {
    // special care for block_freq_array, block_density_array, nid_to_block
    IdType num_node = stream_freq_list->Shape().at(1);
    IdType num_block = num_node;
    IdType num_slot = 1;
    if (GetEnv("COLL_FINE_SOLVE_1_SLOT") != "") {
      num_slot = std::stoi(GetEnv("COLL_FINE_SOLVE_1_SLOT"));
      CHECK(num_block % num_slot == 0);
      num_block /= num_slot;
    }
    block_placement = Tensor::CreateShm(_shm_name_place, kU8, {num_block}, "coll_cache_block_placement");
    block_density_tensor = Tensor::CreateShm(_shm_name_dens, kF64, {static_cast<unsigned long>(num_block)}, "");
    LOG(ERROR) << " fine solver has " << num_block << " blocks and " << num_slot << "slots";
    for (int block_id = 0; block_id < num_block; block_id++) {
      for (IdType slot_id = 0; slot_id < num_slot; slot_id ++) {
        IdType offset = block_id;
        IdType nid = slot_id * num_block + offset;
        nid_to_block->Ptr<uint32_t>()[nid] = block_id;
      }
      block_density_tensor->Ptr<double>()[block_id] = (double)100 / (double)num_node * num_slot;
    }
  }
  
  CHECK(mode == "BIN");
  CHECK(block_density_tensor->Defined());
  double*            block_density_array = block_density_tensor->Ptr<double>();
  TensorView<IdType> block_freq_array(stream_freq_list);
  PerT      cache_percent  = device_to_cache_percent.at(0);
  uint32_t num_device     = device_to_stream.size();
  IdType   num_stream     = stream_freq_list->Shape().at(0);
  uint32_t num_block      = block_density_tensor->Shape().at(0);
  uint32_t num_link       = link_src[0].size(); // we simply assume all dst device have same num link
  LOG(WARNING) << "constructing coll fine grain solver, device=" << num_device << ", stream=" << num_stream;

  std::cerr << num_block << " blocks, " << num_device << " devices\n";

  GRBEnv env = GRBEnv(true);
  env.set("LogFile", "cppsolver.log");
  // env.set(GRB_IntParam_Threads, RunConfig::omp_thread_num*2);
  env.set(GRB_IntParam_Threads, RunConfig::solver_omp_thread_num);

  // env.set(GRB_IntParam_LogToConsole, 0);

  // env.set(GRB_DoubleParam_TimeLimit, 200);

  // // old parameters for cpu, then concurrent remote, then local
  // env.set(GRB_DoubleParam_BarConvTol, 1e-4);
  // env.set(GRB_DoubleParam_OptimalityTol, 1e-2);
  // env.set(GRB_IntParam_Method, 3);
  // env.set(GRB_DoubleParam_MIPGap, 0.03);
  // env.set(GRB_IntParam_MIPFocus, 2);
  // env.set(GRB_IntParam_ConcurrentMIP, 36);
  // env.set(GRB_IntParam_ConcurrentMIP, 10);
  // env.set(GRB_IntParam_BranchDir, 1);
  // env.set(GRB_IntParam_AggFill, 100);
  // env.set(GRB_IntParam_NormAdjust, 3);
  // env.set(GRB_IntParam_Presolve, 2);
  // env.set(GRB_IntParam_SimplexPricing, 2);
  // env.set(GRB_IntParam_DegenMoves, 0);
  // env.set(GRB_IntParam_CutPasses, 5);
  // env.set(GRB_IntParam_PrePasses, 8);
  // env.set(GRB_DoubleParam_Heuristics, 0.001);
  // env.set(GRB_IntParam_ScaleFlag, 0);
  // env.set(GRB_IntParam_StrongCGCuts, 0);
  // env.set(GRB_IntParam_MIRCuts, 1);
  // env.set(GRB_IntParam_Cuts, 3);

  // new parameters for [           cpu          ]
  //                    [local][cuncurrent remote]
  env.set(GRB_DoubleParam_MIPGap, 0.05);
  // env.set(GRB_IntParam_Presolve, 2);
  // env.set(GRB_IntParam_AggFill, 100);
  // env.set(GRB_IntParam_Aggregate, 0);
  // env.set(GRB_IntParam_GomoryPasses, 0);
  /** todo: the following param seems hurt performance. needs investigation */
  // env.set(GRB_IntParam_Method, 2);
  // env.set(GRB_IntParam_DegenMoves, 0);
  // env.set(GRB_IntParam_PrePasses, 5);
  // env.set(GRB_IntParam_NormAdjust, 0);
  // env.set(GRB_IntParam_FlowCoverCuts, 2);

  if (GetEnv("COLL_GUROBI_EXP_PARAM") == "1") {
    env.set(GRB_IntParam_Cuts, 2);
    env.set(GRB_IntParam_DegenMoves, 2);
    env.set(GRB_IntParam_ScaleFlag, 1);
  }

  env.start();

  GRBModel model = GRBModel(env);

  /**
   * z: final goal, max time of each dst gpu, minimize it
   * x: storage of each block on each src gpu
   * a: access flag of each block: a[block][dst][src] means whether dst read this block from src
   */
  GRBVar z = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS, "z");
  TensorPtr c_list_tensor = Tensor::Empty(kI64, {num_block}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");
  TensorPtr x_list_tensor = Tensor::Empty(kI64, {num_block, num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "x_list");
  TensorPtr a_list_tensor = Tensor::Empty(kI64, {num_block, num_device, num_link}, CPU(CPU_CLIB_MALLOC_DEVICE), "a_list");
  TensorPtr max_remote_time_tensor = Tensor::Empty(kI64, {num_device}, CPU(CPU_CLIB_MALLOC_DEVICE), "c_list");

  TensorView<GRBVar> c_list(c_list_tensor);
  TensorView<GRBVar> x_list(x_list_tensor);
  TensorView<GRBVar> a_list(a_list_tensor);
  TensorView<GRBVar> max_remote_time(max_remote_time_tensor);

  // std::vector<GRBLinExpr> time_list(num_device);
  // std::vector<GRBLinExpr> local_cpu_time_list(num_device);
  std::vector<GRBLinExpr> cpu_time_list(num_device);
  std::vector<GRBLinExpr> local_time_list(num_device);
  vec<vec<GRBLinExpr>>    remote_time_list(num_device, vec<GRBLinExpr>(num_link));
  std::vector<GRBLinExpr> local_weight_list(num_device);
  std::vector<double>     total_weight_list(num_device);
  std::vector<GRBLinExpr> cpu_weight_list(num_device);
  try {

  // for each dst, 
  // sum a_dst_src <= 1
  // src
  auto constraint_connect_a_c = [&](GRBModel & model, uint32_t block_id) {
    FOR_LOOP(dst_dev, num_device) {
      GRBLinExpr expr;
      FOR_LOOP(src_link, num_link) {
        expr += a_list[block_id][dst_dev][src_link].ref();
      }
      model.addConstr(expr + c_list[block_id].ref() + x_list[block_id][dst_dev].ref() == 1);
    }
  };

  // for each src,
  // x_src >= max(a_dst_src)
  //          dst
  auto constraint_connect_a_x = [&](GRBModel & model, uint32_t block_id) {
    FOR_LOOP(dst_dev, num_device) {
      FOR_LOOP(src_link, num_link) {
        GRBLinExpr expr;
        for (auto src_dev : link_src[dst_dev][src_link]) {
          expr += x_list[block_id][src_dev].ref();
        }
        model.addConstr(expr >= a_list[block_id][dst_dev][src_link].ref());
      }
    }
    /** try reduce num of constr. but seems results in longer solve time*/
    // FOR_LOOP(src_dev, num_device) {
    //   GRBLinExpr expr;
    //   FOR_LOOP(dst_dev, num_device) {
    //     FOR_LOOP(src_link, num_link) {
    //       CHECK(link_src[dst_dev][src_link].size() == 1);
    //       if (link_src[dst_dev][src_link][0] != src_dev) continue;
    //       expr += a_list[block_id][dst_dev][src_link].ref();
    //     }
    //   }
    //   model.addConstr(x_list[block_id][src_dev].ref() * num_device >= expr);
    // }
  };

  // for each src,
  //  sum x_src < cache size
  // block
  auto constraint_capacity = [&](GRBModel & model, uint32_t src_dev) {
    GRBLinExpr expr;
    FOR_LOOP(block_id, num_block) {
      expr += x_list[block_id][src_dev].ref() * block_density_array[block_id];
    }
    model.addConstr(expr <= cache_percent);
  };
  auto constraint_time = [&](GRBModel & model, uint32_t dst_dev) {
    uint32_t stream_id = device_to_stream[dst_dev];
    double sum_weight = 0;
    // GRBLinExpr &local_cpu_time = local_cpu_time_list[dst_dev];
    GRBLinExpr &cpu_time = cpu_time_list[dst_dev];
    GRBLinExpr &local_time = local_time_list[dst_dev];
    FOR_LOOP(block_id, num_block) {
      // stream_freq_list is at <stream,nid> order
      double weight = block_density_array[block_id] * block_freq_array[0][block_id].ref();
      if (weight == 0) continue;
      sum_weight += weight;
      cpu_time += c_list[block_id].ref() * T_cpu * weight;
      local_time += x_list[block_id][dst_dev].ref() * T_local * weight;

      local_weight_list[dst_dev] +=  weight * x_list[block_id][dst_dev].ref();
      cpu_weight_list[dst_dev]   +=  weight * c_list[block_id].ref();
    }
    GRBLinExpr total_time;
    FOR_LOOP(src_link, num_link) {
      GRBLinExpr &remote_time = remote_time_list[dst_dev][src_link];
      FOR_LOOP(block_id, num_block) {
        double weight = block_density_array[block_id] * block_freq_array[0][block_id].ref();
        if (weight == 0) continue;
        remote_time += a_list[block_id][dst_dev][src_link].ref() * link_time[dst_dev][src_link] * weight;
      }
      if (true) {
      // if (RunConfig::concurrent_link_impl == kMPSPhase ||
      //     RunConfig::concurrent_link_impl == kSMMaskPhase) {
        model.addConstr(remote_time <= z);
        total_time += remote_time * RunConfig::coll_cache_link_desc.link_sm[dst_dev][src_link];
      } else {
        model.addConstr(remote_time <= max_remote_time[dst_dev].ref());
      }
      // model.addConstr(remote_time + local_cpu_time <= z);
    }
    if (true) {
    // if (RunConfig::concurrent_link_impl == kMPSPhase ||
    //     RunConfig::concurrent_link_impl == kSMMaskPhase) {
      model.addConstr(local_time <= z);
      // local sm + cpu sm is always total sm
      int total_sm = RunConfig::coll_cache_link_desc.local_sm[dst_dev] + RunConfig::coll_cache_link_desc.cpu_sm[dst_dev];
      LOG(ERROR) << "dst " << dst_dev << ", total sm is " << total_sm;
      total_time += local_time * RunConfig::coll_cache_link_desc.local_sm[dst_dev];
      total_time += (cpu_time + sum_weight * RunConfig::coll_cache_cpu_addup * T_cpu) * RunConfig::coll_cache_link_desc.cpu_sm[dst_dev];
      model.addConstr(total_time <= z * total_sm);
    } else {
      model.addConstr(max_remote_time[dst_dev].ref() + local_time <= z);
    }
    model.addConstr(cpu_time + sum_weight * RunConfig::coll_cache_cpu_addup * T_cpu <= z);
    total_weight_list[dst_dev] = sum_weight;
  };

  LOG(WARNING) << "Add Var...";
  char var_type = (mode == "BIN") ? GRB_BINARY : GRB_CONTINUOUS;
  FOR_LOOP(block_id, num_block) {
    c_list[block_id].ref() = model.addVar(0, 1, 0, var_type);
    FOR_LOOP(dst_dev, num_device) {
      x_list[block_id][dst_dev].ref() = model.addVar(0, 1, 0, var_type);
      FOR_LOOP(src_link, num_link) {
        a_list[block_id][dst_dev][src_link].ref() = model.addVar(0, 1, 0, var_type);
      }
    }
  }
  FOR_LOOP(dst_dev, num_device) {
    max_remote_time[dst_dev].ref() = model.addVar(0.0, std::numeric_limits<double>::max(), 0.0, GRB_CONTINUOUS);
  }

  LOG(WARNING) << "Capacity...";
  FOR_LOOP(device_id, num_device) {constraint_capacity(model, device_id);}

  LOG(WARNING) << "Connect CPU...";
  FOR_LOOP(block_id, num_block) {constraint_connect_a_c(model, block_id);}
  LOG(WARNING) << "Connect Access To Storage...";
  FOR_LOOP(block_id, num_block) {
    FOR_LOOP(device_id, num_device) {constraint_connect_a_x(model, block_id);}
  }
  LOG(WARNING) << "Time...";
  FOR_LOOP(device_id, num_device) {constraint_time(model, device_id);}

  model.setObjective(z + 0, GRB_MINIMIZE);

  model.write("asymm.lp");

  if (GetEnv("SAVE_MILP_ONLY") != "") {
    std::string cmd = "mv asymm.lp " + GetEnv("COLL_LOG_BASE") + ".lp";
    system(cmd.c_str());
    abort();
  }

  model.optimize();
  } catch (GRBException e) {
    LOG(FATAL) << e.getMessage();
    abort();
  }

  CHECK(num_device <= 8);
  CHECK(block_placement->Shape() == std::vector<size_t>{num_block});
  // num_link + local + cpu always <= 8
  block_access_from = Tensor::CreateShm(_shm_name_access, kU8, {num_device, num_block}, "coll_cache_block_access");
  TensorView<uint8_t> block_placement_array(block_placement);
  TensorView<uint8_t> block_access_from_array(block_access_from);
  LOG(WARNING) << "Coll Cache init block placement array";
  // #pragma omp parallel for num_threads(RunConfig::omp_thread_num)

  auto get_val = [](GRBVar & var) { return var.get(GRB_DoubleAttr::GRB_DoubleAttr_X);};
  auto get_int = [](GRBVar & var) { return std::round(var.get(GRB_DoubleAttr::GRB_DoubleAttr_X));};

  vec<vec<uint8_t>> link_bitmap(num_device, vec<uint8_t>(num_link, 0));
  for (uint32_t dev_id = 0; dev_id < num_device; dev_id++) {
    for (uint32_t link = 0; link < num_link; link++) {
      uint8_t & bitmap = link_bitmap[dev_id][link];
      for (auto src_dev : this->link_src[dev_id][link]) {
        bitmap |= 1 << src_dev;
      }
    }
  }

  vec<uint8_t> bitmap_to_src_dev(1 << num_device, 0);
  OptimalAsymmLinkSolver::PreDecideSrc(num_device, num_device + 1, bitmap_to_src_dev.data());

  FOR_LOOP(block_id, num_block) {
    // by default, this block is placed at cpu
    block_placement_array[block_id].ref() = 0;
    FOR_LOOP(device_id, num_device) {
      block_access_from_array[device_id][block_id].ref() = num_device; // num_device is treat as cpu
    }

    // std::ios_base::fmtflags f( std::cerr.flags() );
    // std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[0][block_id].ref();

    // x == 1 -> access from = local dev id
    // x == 0, a != 0 -> <bitmap of this link> & <storage bitmap>, then choose one from it
    // x == 0, a == 0 -> cpu
    FOR_LOOP(device_id, num_device) {
      uint8_t x_result = (uint8_t)std::round(x_list[block_id][device_id].ref().get(GRB_DoubleAttr::GRB_DoubleAttr_X));
      block_placement_array[block_id].ref() |= (x_result << device_id);
    }
    // build access from
    FOR_LOOP(device_id, num_device) {
      if (get_int(x_list[block_id][device_id].ref())) {
        block_access_from_array[device_id][block_id].ref() = device_id;
        continue;
      }
      if (get_int(c_list[block_id].ref())) {
        block_access_from_array[device_id][block_id].ref() = num_device; // num_device is treat as cpu
        continue;
      }
      for (uint32_t src_link = 0; src_link < num_link; src_link ++) {
        if (get_int(a_list[block_id][device_id][src_link].ref()) == 0) continue;
        const uint8_t link_src_bitmap = link_bitmap[device_id][src_link];
        const uint8_t storage_bit_map = block_placement_array[block_id].ref();
        const uint8_t candidate_bit_map = link_src_bitmap & storage_bit_map;
        block_access_from_array[device_id][block_id].ref() = bitmap_to_src_dev[candidate_bit_map];
        break;
      }
    }
    // std::bitset<8> bs(block_placement_array[block_id].ref());
    // std::cerr << "  storage is " << bs << "\n";
    // std::cerr.flags(f);
  }
  std::cout << "coll_cache:optimal_local_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << local_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_remote_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << 1 - (local_weight_list[part_id].getValue() + cpu_weight_list[part_id].getValue()) / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=";
  FOR_LOOP(part_id, num_device) { std::cout << cpu_weight_list[part_id].getValue() / total_weight_list[part_id] << ","; }
  std::cout << "\n";
  std::cout << "z=" << z.get(GRB_DoubleAttr::GRB_DoubleAttr_X) << "\n";
  LOG(WARNING) << "Coll Cache init block placement array done";
  model.reset(1);
  LOG(WARNING) << "Coll Cache model reset done";
}

void RepSolver::Solve(std::vector<int> device_to_stream,
                      std::vector<PerT> device_to_cache_percent,
                      std::string mode, double T_local, double T_cpu) {
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const int num_device = device_to_stream.size();
  const int num_block = 2;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  const IdType replicate_size = num_cached_nodes;
  const IdType cpu_size = num_node - replicate_size;

  double rep_w = 0, cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : rep_w, cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    if (rank < replicate_size) {
      nid_to_block->Ptr<IdType>()[node_id] = 0;
      rep_w += freq_array[rank];
    } else {
      nid_to_block->Ptr<IdType>()[node_id] = 1;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
  }
  double local_w = rep_w;
  std::cout << "coll_cache:optimal_rep_storage=" << replicate_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << cpu_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << replicate_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << 0 << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=" << local_w * 100 / num_node * T_local + cpu_w * 100 / num_node * T_cpu << "\n";

  block_placement = Tensor::CreateShm(_shm_name_place, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");
  block_placement->Ptr<uint8_t>()[0] = (1 << num_device) - 1;
  block_placement->Ptr<uint8_t>()[1] = 0;
  block_access_from = Tensor::CreateShm(_shm_name_access, kU8, {(size_t)num_device, static_cast<size_t>(num_block)}, "coll_cache_advise");
  TensorView<uint8_t> block_access_from_array(block_access_from);
  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    block_access_from_array[dev_id][0].ref() = dev_id;
    block_access_from_array[dev_id][1].ref() = num_device;
  }
  block_density_tensor = Tensor::CreateShm(_shm_name_dens, kF64, {num_block}, "");
  block_density_tensor->Ptr<double>()[0] = (double)replicate_size * 100 / (double)num_node;
  block_density_tensor->Ptr<double>()[1] = (double)cpu_size * 100 / (double)num_node;
}

void CliquePartSolver::Solve(std::vector<int> device_to_stream,
                             std::vector<PerT> device_to_cache_percent,
                             std::string mode, double T_local, double T_cpu) {
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const size_t num_device = device_to_stream.size();
  const int num_clique = RoundUpDiv<size_t>(num_device, clique_size);
  const int num_block = clique_size + 1;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = std::ceil(num_node * (device_to_cache_percent[0] / (double)100));

  // LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  CHECK_EQ(stream_freq_list->Type(), kI32);
  const IdType * freq_array = stream_freq_list->Ptr<IdType>();

  // num_cached_nodes is calculated by cache rate, but the total cache space may exceed num node
  // so the actual num_cached_nodes per device should be smaller
  const IdType partition_size = std::min(num_cached_nodes, RoundUpDiv<IdType>(num_node, clique_size));

  double cpu_w = 0, total_w = 0;

  if (RunConfig::coll_hash_impl == kRR || RunConfig::coll_hash_impl == kChunk) {
    CHECK(partition_size * clique_size >= num_node);
  }

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = stream_id_list->Ptr<IdType>()[rank];
    IdType block_id = clique_size;
    auto hash_base = rank;
    if (rank < partition_size * clique_size) {
      if (RunConfig::coll_hash_impl == kRR) {
        block_id = node_id % clique_size;
      } else if (RunConfig::coll_hash_impl == kChunk) {
        block_id = node_id / partition_size;
      } else {
        CHECK(RunConfig::coll_hash_impl == kDefault);
        block_id = rank % clique_size;
      }
    } else {
      block_id = clique_size;
      cpu_w += freq_array[rank];
    }
    total_w += freq_array[rank];
    nid_to_block->Ptr<IdType>()[node_id] = block_id;
  }
  double local_w = (total_w - cpu_w) / clique_size;
  double remote_w = (total_w - cpu_w) / clique_size * (clique_size - 1);
  double one_partition_w = local_w;

  std::vector<double> z_list(num_device, 0);

  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    z_list[dev_id] += local_w * 100 / num_node * T_local;
    auto & link_list = RunConfig::coll_cache_link_desc.link_src[dev_id];
    auto & link_time_list = RunConfig::coll_cache_link_desc.link_time[dev_id];
    for (IdType link_id = 0; link_id < link_list.size(); link_id++) {
      if (link_list[link_id].size() == 0) continue;
      z_list[dev_id] += one_partition_w * 100 / num_node * link_time_list[link_id];
    }
    z_list[dev_id] += cpu_w * 100 / num_node * T_cpu;
  }

  std::cout << "coll_cache:optimal_rep_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * clique_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_size * clique_size / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << partition_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (clique_size - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << z_list[dev_id] << ","; }
  std::cout << "\n";

  auto dev_id_to_clique_id = [&](IdType dev_id) {
    return dev_id / clique_size;
  };
  auto block_id_to_storage_dev_in_clique = [&](IdType block_id, IdType clique_id) {
    return clique_id * clique_size + block_id;
  };
  auto block_id_to_storage_bitmap = [&](IdType block_id) {
    uint8_t storage = 0;
    for (IdType clique_id = 0; clique_id < num_clique; clique_id++) {
      storage |= 1 << block_id_to_storage_dev_in_clique(block_id, clique_id);
    }
    return storage;
  };
  auto block_id_to_src_dev = [&](IdType block_id, IdType dst_dev) {
    IdType clique_id = dev_id_to_clique_id(dst_dev);
    return block_id_to_storage_dev_in_clique(block_id, clique_id);
  };

  block_placement = Tensor::CreateShm(_shm_name_place, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");
  block_access_from = Tensor::CreateShm(_shm_name_access, kU8, {num_device, static_cast<size_t>(num_block)}, "coll_cache_advise");
  block_density_tensor = Tensor::CreateShm(_shm_name_dens, kF64, {(size_t)num_block}, "");
  TensorView<uint8_t> block_access_from_array(block_access_from);
  // there are clique_size + 1 blocks; first clique_size block is clique-partitioned, the last is on cpu
  for (int block_id = 0; block_id < clique_size; block_id++) {
    block_placement->Ptr<uint8_t>()[block_id] = block_id_to_storage_bitmap(block_id);
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      block_access_from_array[dev_id][block_id].ref() = block_id_to_src_dev(block_id, dev_id);
    }
    block_density_tensor->Ptr<double>()[block_id] = (double)partition_size * 100 / (double)num_node;
  }
  // the one on cpu
  block_placement->Ptr<uint8_t>()[clique_size] = 0;
  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    block_access_from_array[dev_id][clique_size].ref() = num_device;
  }
  block_density_tensor->Ptr<double>()[clique_size] = ((double)num_node - partition_size * clique_size) * 100 / (double)num_node;

  FOR_LOOP(block_id, num_block) {
    // std::ios_base::fmtflags f( std::cerr.flags() );
    std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref()
    ;
    std::bitset<8> bs(block_placement->CPtr<uint8_t>()[block_id]);
    std::cerr << " storage is " << bs << "\n";
    // std::cerr.flags(f);
    std::cerr << "\taccess is\t";
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      std::cerr << int(block_access_from_array[dev_id][block_id].ref()) << "\t";
    }
    std::cerr << "\n";
  }
}

void CliquePartByDegreeSolver::Solve(std::vector<int> device_to_stream,
                             std::vector<PerT> device_to_cache_percent,
                             std::string mode, double T_local, double T_cpu) {
  CHECK(ranking_nodes->Shape().size() == 1);
  CHECK(stream_id_list->Shape()[0] == 1);
  CHECK(stream_freq_list->Shape()[0] == 1);
  CHECK(std::accumulate(device_to_stream.begin(), device_to_stream.end(), 0, std::plus<>()) == 0);
  const size_t num_device = device_to_stream.size();
  const int num_clique = RoundUpDiv<size_t>(num_device, clique_size);
  const int num_block = clique_size + 1;
  const IdType num_node = stream_freq_list->Shape()[1];
  // the freq list must already be sorted
  // now calculate the boundary
  const IdType num_cached_nodes = num_node * (device_to_cache_percent[0] / (double)100);

  // LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;
  CHECK_EQ(stream_freq_list->Type(), kI32);

  // map node id to freq
  TensorPtr node_id_to_freq = Tensor::Empty(kI32, {num_node}, CPU_CLIB(), "");
  {
    const IdType * freq_id_array = stream_id_list->Ptr<IdType>();
    const IdType * freq_array = stream_freq_list->Ptr<IdType>();
    #pragma omp parallel for num_threads(RunConfig::omp_thread_num)
    for (size_t i = 0; i < num_node; i++) {
      IdType node_id = freq_id_array[i];
      IdType freq = freq_array[i];
      node_id_to_freq->Ptr<IdType>()[node_id] = freq;
    }
  }

  // calculate partition
  const IdType partition_size = std::min(num_cached_nodes, num_node / clique_size);

  // distribute nodes onto GPU/CPU
  double cpu_w = 0, total_w = 0;

#pragma omp parallel for num_threads(RunConfig::omp_thread_num) reduction(+ : cpu_w, total_w)
  for (IdType rank = 0; rank < num_node; rank++) {
    IdType node_id = ranking_nodes->CPtr<IdType>()[rank];
    IdType block_id = clique_size;
    IdType freq = node_id_to_freq->CPtr<IdType>()[node_id];
    if (rank < partition_size * clique_size) {
      block_id = rank % clique_size;
    } else {
      block_id = clique_size;
      cpu_w += freq;
    }
    total_w += freq;
    nid_to_block->Ptr<IdType>()[node_id] = block_id;
  }
  double local_w = (total_w - cpu_w) / clique_size;
  double remote_w = (total_w - cpu_w) / clique_size * (clique_size - 1);
  double one_partition_w = local_w;

  std::vector<double> z_list(num_device, 0);

  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    z_list[dev_id] += local_w * 100 / num_node * T_local;
    auto & link_list = RunConfig::coll_cache_link_desc.link_src[dev_id];
    auto & link_time_list = RunConfig::coll_cache_link_desc.link_time[dev_id];
    for (IdType link_id = 0; link_id < link_list.size(); link_id++) {
      if (link_list[link_id].size() == 0) continue;
      z_list[dev_id] += one_partition_w * 100 / num_node * link_time_list[link_id];
    }
    z_list[dev_id] += cpu_w * 100 / num_node * T_cpu;
  }

  std::cout << "coll_cache:optimal_rep_storage=" << 0 << "\n";
  std::cout << "coll_cache:optimal_part_storage=" << partition_size * clique_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_cpu_storage=" << 1 - (partition_size * clique_size / (double)num_node) << "\n";
  std::cout << "coll_cache:optimal_local_storage=" << partition_size / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_remote_storage=" << partition_size * (clique_size - 1) / (double)num_node << "\n";
  std::cout << "coll_cache:optimal_local_rate=" << local_w / total_w << "\n";
  std::cout << "coll_cache:optimal_remote_rate=" << remote_w / total_w << "\n";
  std::cout << "coll_cache:optimal_cpu_rate=" << cpu_w / total_w << "\n";
  std::cout << "z=";
  FOR_LOOP(dev_id, num_device) { std::cout << z_list[dev_id] << ","; }
  std::cout << "\n";

  auto dev_id_to_clique_id = [&](IdType dev_id) {
    return dev_id / clique_size;
  };
  auto block_id_to_storage_dev_in_clique = [&](IdType block_id, IdType clique_id) {
    return clique_id * clique_size + block_id;
  };
  auto block_id_to_storage_bitmap = [&](IdType block_id) {
    uint8_t storage = 0;
    for (IdType clique_id = 0; clique_id < num_clique; clique_id++) {
      storage |= 1 << block_id_to_storage_dev_in_clique(block_id, clique_id);
    }
    return storage;
  };
  auto block_id_to_src_dev = [&](IdType block_id, IdType dst_dev) {
    IdType clique_id = dev_id_to_clique_id(dst_dev);
    return block_id_to_storage_dev_in_clique(block_id, clique_id);
  };

  block_placement = Tensor::CreateShm(_shm_name_place, kU8, {static_cast<size_t>(num_block)}, "coll_cache_block_placement");
  block_access_from = Tensor::CreateShm(_shm_name_access, kU8, {num_device, static_cast<size_t>(num_block)}, "coll_cache_advise");
  block_density_tensor = Tensor::CreateShm(_shm_name_dens, kF64, {(size_t)num_block}, "");
  TensorView<uint8_t> block_access_from_array(block_access_from);
  // there are clique_size + 1 blocks; first clique_size block is clique-partitioned, the last is on cpu
  for (int block_id = 0; block_id < clique_size; block_id++) {
    block_placement->Ptr<uint8_t>()[block_id] = block_id_to_storage_bitmap(block_id);
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      block_access_from_array[dev_id][block_id].ref() = block_id_to_src_dev(block_id, dev_id);
    }
    block_density_tensor->Ptr<double>()[block_id] = (double)partition_size * 100 / (double)num_node;
  }
  // the one on cpu
  block_placement->Ptr<uint8_t>()[clique_size] = 0;
  for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
    block_access_from_array[dev_id][clique_size].ref() = num_device;
  }
  block_density_tensor->Ptr<double>()[clique_size] = ((double)num_node - partition_size * clique_size) * 100 / (double)num_node;

  FOR_LOOP(block_id, num_block) {
    // std::ios_base::fmtflags f( std::cerr.flags() );
    std::cerr << "block " << block_id
    //           << std::fixed << std::setw(8) << std::setprecision(6)
    //           << ", density=" << block_density_array[block_id]
    //           << std::fixed << std::setw(8) << std::setprecision(3)
    //           << ", freq=" << block_freq_array[block_id][0].ref()
    ;
    std::bitset<8> bs(block_placement->CPtr<uint8_t>()[block_id]);
    std::cerr << " storage is " << bs << "\n";
    // std::cerr.flags(f);
    std::cerr << "\taccess is\t";
    for (IdType dev_id = 0; dev_id < num_device; dev_id++) {
      std::cerr << int(block_access_from_array[dev_id][block_id].ref()) << "\t";
    }
    std::cerr << "\n";
  }
}

} // namespace coll_cache
}
}
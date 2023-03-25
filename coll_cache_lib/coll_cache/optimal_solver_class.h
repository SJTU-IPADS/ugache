#pragma once
// #include "../common.h"
// #include "../cpu/mmap_cpu_device.h"
// #include "../device.h"
// #include "../logging.h"
#include "../run_config.h"
#include "gurobi_c++.h"
// #include "ndarray.h"
#include "asymm_link_desc.h"
#include "../freq_recorder.h"
#include "ndarray.h"
#include <bitset>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <tbb/concurrent_unordered_map.h>
#include <unistd.h>
#include <vector>

#define FOR_LOOP(iter, len) for (uint32_t iter = 0; iter < (len); iter++)
#define FOR_LOOP_1(iter, len) for (uint32_t iter = 1; iter < (len); iter++)

namespace coll_cache_lib {
namespace common {
namespace coll_cache {
static_assert(sizeof(GRBVar) == sizeof(Id64Type),
              "size of GRBVar is not 8byte, cannot use tensor to hold it..");

using PerT = double;

class CollCacheSolver {
 public:
  virtual ~CollCacheSolver() {}
  virtual void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
                     std::vector<int> device_to_stream,
                     const IdType num_node,
                     const TensorPtr nid_to_block_tensor) = 0;
  virtual void BuildSingleStream(ContFreqBuf* freq_rank,
                     std::vector<int> device_to_stream,
                     const IdType num_node,
                     const TensorPtr nid_to_block_tensor) { CHECK(false) << "Unimplemented"; };
  virtual void Solve(std::vector<int> device_to_stream,
                     std::vector<PerT> device_to_cache_percent, std::string mode,
                     double T_local, double T_cpu);

 protected:
  virtual void Solve(std::vector<int> device_to_stream,
                     std::vector<PerT> device_to_cache_percent, std::string mode,
                     double T_local, double T_remote,
                     double T_cpu) {CHECK(false) << "Unimplemented";}
 public:
  TensorPtr block_density_tensor;
  TensorPtr block_freq_tensor;
  TensorPtr block_placement;
  TensorPtr block_access_from;

  static std::string _shm_name_nid_to_block ; // = Constant::kCollCacheNIdToBlockShmName;
  static std::string _shm_name_access       ; // = Constant::kCollCacheAccessShmName;
  static std::string _shm_name_place        ; // = Constant::kCollCachePlacementShmName;
  static std::string _shm_name_dens         ; // = Constant::kCollCachePlacementShmName + "_density";

  static std::string _shm_name_alter_nid_to_block ; // = Constant::kCollCacheNIdToBlockShmName             + "_old";
  static std::string _shm_name_alter_access       ; // = Constant::kCollCacheAccessShmName                 + "_old";
  static std::string _shm_name_alter_place        ; // = Constant::kCollCachePlacementShmName              + "_old";
  static std::string _shm_name_alter_dens         ; // = Constant::kCollCachePlacementShmName + "_density" + "_old";

};

class OptimalSolver : public CollCacheSolver {
public:
  // virtual ~OptimalSolver() {}
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node,
             const TensorPtr nid_to_block_tensor) override;
  void BuildSingleStream(ContFreqBuf* freq_rank,
                     std::vector<int> device_to_stream,
                     const IdType num_node,
                     const TensorPtr nid_to_block_tensor) override;
  void BuildSingleStream(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node,
             const TensorPtr nid_to_block_tensor);
  using CollCacheSolver::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_remote,
             double T_cpu) override;

protected:
  /** ==================================================================
   * @brief status related to building blocks.
   *  only depends on dataset and its frequency
   *  does not depends on system configuration(i.e. cache size, num device)
   *  ==================================================================
   */

  // the frequency of node at top 1%. slot/block is sliced based on this value
  double alpha = 0;
  // how many slot to slice for each stream
  std::atomic_uint32_t next_free_block{0};
  uint32_t max_size_per_block = 10000;

  int freq_to_slot_1(float freq, IdType num_node) {
    if (freq == 0)
      return RunConfig::coll_cache_num_slot - 1;
    if (freq >= alpha)
      return 0;
    double exp = std::log2(alpha / (double)freq) / std::log2(RunConfig::coll_cache_coefficient);
    int slot = (int)std::floor(exp);
    slot = std::min(slot, (int)RunConfig::coll_cache_num_slot - 2);
    return slot;
  }
  int freq_to_slot_2(float freq, uint32_t rank, IdType num_node) {
    return rank * (uint64_t)RunConfig::coll_cache_num_slot / num_node;
  }

  /**
   * each <slot freq> is mapped to a block, but the block may get too large.
   * so we split block into smaller blocks:
   * initially, each <slot freq> maps to a default block;
   * by inserting vertex into block, if the block exceed size limit, alloc a new
   * block (by atomic adding next_free_block) and update the <slot freq>
   * mapping.
   */
  inline IdType alloc_block() { return next_free_block.fetch_add(1); }

  struct block_identifer {
    std::atomic_uint64_t _current_block_is_for_num_le_than{0xffffffff00000000};
    std::atomic_uint32_t _registered_node{0};
    std::atomic_uint32_t _done_node{0};
    std::atomic_uint32_t _total_nodes{0};
    volatile uint32_t max_size_this_block = 0;
    uint32_t num_slices = 0;
    uint32_t slice_begin = 0;

    void measure_total_node() {
      _total_nodes.fetch_add(1);
    }
    void set_max_size(int num_worker, uint32_t min_boundary) {
      // this->max_size_this_block = std::min<uint32_t>(RoundUpDiv<uint32_t>(_total_nodes, num_worker*2), min_boundary);
      // this->max_size_this_block = std::min<uint32_t>(RoundUpDiv<uint32_t>(_total_nodes, num_worker), min_boundary);
      this->max_size_this_block = min_boundary;
    }

    uint32_t add_node(OptimalSolver * solver) {
      const uint32_t insert_order = _registered_node.fetch_add(1);
      uint64_t old_val = _current_block_is_for_num_le_than.load();
      uint32_t covered_num = old_val & 0xffffffff;
      if ((insert_order % max_size_this_block) == 0) {
        while (_done_node.load() < insert_order) {}
        old_val = _current_block_is_for_num_le_than.load();
        // alloc a new block
        uint32_t selected_block = solver->alloc_block();
        uint64_t new_val = (((uint64_t) selected_block) << 32) | (insert_order + max_size_this_block);
        CHECK(_current_block_is_for_num_le_than.compare_exchange_strong(old_val, new_val));
        _done_node.fetch_add(1);
        return selected_block;
      } else {
        while (covered_num <= insert_order) {
          old_val = _current_block_is_for_num_le_than.load();
          covered_num = old_val & 0xffffffff;
        }
        uint32_t selected_block = old_val >> 32;
        _done_node.fetch_add(1);
        return selected_block;
      }
    }
  };

  struct full_slot {
    volatile size_t remmaped_slot = 0xffffffffffffffff;
    tbb::atomic<uint32_t> size{0};
    size_t orig_seq_slot;
  };
  struct full_slot_single_thread {
    uint32_t remmaped_slot = 0xffffffff;
    uint32_t size = 0;
    size_t orig_seq_slot;
  };

  struct concurrent_full_slot_map {
    #ifdef DEAD_CODE
    const size_t _place_holder = 0xffffffffffffffff;
    std::atomic_uint32_t next_free_slot{0};
    // tbb::concurrent_unordered_map<size_t, volatile size_t> the_map;
    tbb::concurrent_unordered_map<size_t, full_slot> the_map;
    concurrent_full_slot_map() {}
    uint32_t register_bucket(size_t slot_array_seq_id) {
      auto rst = the_map.insert({slot_array_seq_id, full_slot()});
      if (rst.second == true) {
        rst.first->second.orig_seq_slot = slot_array_seq_id; // i.e. the key
        rst.first->second.remmaped_slot =
            next_free_slot.fetch_add(1); // the allcoated block identifer
      } else {
        while (rst.first->second.remmaped_slot == _place_holder) {
        }
      }
      rst.first->second.size.fetch_and_increment();
      return rst.first->second.remmaped_slot;
    }
    #endif
    // const size_t _place_holder = 0xffffffffffffffff;
    std::atomic_uint32_t next_free_slot{0};
    uint32_t __next_free_slot = 0;
    std::unordered_map<size_t, full_slot_single_thread> the_map;
    uint32_t register_bucket(size_t slot_array_seq_id) {
      auto iter = the_map.find(slot_array_seq_id);
      if (iter == the_map.end()) {
        auto val = full_slot_single_thread();
        val.orig_seq_slot = slot_array_seq_id;
        val.remmaped_slot = __next_free_slot++;
        val.size = 1;
        auto rst = the_map.insert({slot_array_seq_id, val});
        return val.remmaped_slot;
      } else {
        iter->second.size++;
        return iter->second.remmaped_slot;
      }
    }
  };

  /** ==================================================================
   * @brief status for solving optimal policy based on slot/block slice result
   *  depends on system configuration(i.e. cache size, num device)
   *  ==================================================================
   */

  static inline bool ignore_block(uint32_t block_id, double weight) {
    return (weight == 0) && (block_id > 0);
  }
};

class OptimalAsymmLinkSolver : public OptimalSolver {
  template<typename T>
  using vec=std::vector<T>;
 public:
  OptimalAsymmLinkSolver() : link_src(RunConfig::coll_cache_link_desc.link_src), link_time(RunConfig::coll_cache_link_desc.link_time) {}
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local,double T_cpu) override;
  vec<vec<vec<int>>> link_src;
  vec<vec<double>> link_time;
  static void PreDecideSrc(int num_bits, int cpu_location_id, uint8_t *placement_to_src);
 private:
  using OptimalSolver::Solve;
};

class SingleStreamSolverBase : public CollCacheSolver {
 public:
  void Build(TensorPtr stream_id_list, TensorPtr stream_freq_list,
             std::vector<int> device_to_stream,
             const IdType num_node, const TensorPtr nid_to_block_tensor) override;
  void BuildSingleStream(ContFreqBuf* freq_rank,
                     std::vector<int> device_to_stream,
                     const IdType num_node,
                     const TensorPtr nid_to_block_tensor) override {
    freq_rank->GetLegacyFreqRank(&freq_buf, num_node);
    auto ranking_nodes_list = Tensor::FromBlob(
        freq_buf.rank_vec.data(), coll_cache::get_data_type<IdType>(),
        {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_list");
    auto ranking_nodes_freq_list = Tensor::FromBlob(
        freq_buf.freq_vec.data(), coll_cache::get_data_type<IdType>(),
        {1, num_node}, CPU(CPU_FOREIGN), "ranking_nodes_freq_list");
    Build(ranking_nodes_list, ranking_nodes_freq_list, device_to_stream, num_node, nid_to_block_tensor);
  }
  TensorPtr stream_id_list;
  TensorPtr stream_freq_list;
  TensorPtr nid_to_block;
  LegacyFreqBuf freq_buf;
};

class IntuitiveSolver : public SingleStreamSolverBase {
public:
  using CollCacheSolver::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_remote, double T_cpu);
};
class PartitionSolver : public SingleStreamSolverBase {
public:
  using CollCacheSolver::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_remote, double T_cpu);
};
class PartRepSolver : public SingleStreamSolverBase {
public:
  using CollCacheSolver::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_remote, double T_cpu);
};

class RepSolver : public SingleStreamSolverBase {
public:
  using SingleStreamSolverBase::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_cpu);
};

class CliquePartSolver : public SingleStreamSolverBase {
public:
  CliquePartSolver() : clique_size(RunConfig::coll_cache_link_desc.CliqueSize()) {}
  using SingleStreamSolverBase::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_cpu);
  int clique_size;
};

class CliquePartByDegreeSolver : public SingleStreamSolverBase {
public:
  CliquePartByDegreeSolver(TensorPtr ranking_nodes) : clique_size(RunConfig::coll_cache_link_desc.CliqueSize()), ranking_nodes(ranking_nodes) {}
  using SingleStreamSolverBase::Solve;
  void Solve(std::vector<int> device_to_stream,
             std::vector<PerT> device_to_cache_percent, std::string mode,
             double T_local, double T_cpu);
  int clique_size;
  TensorPtr ranking_nodes;
};

} // namespace coll_cache
} // namespace common
} // namespace coll_cache_lib
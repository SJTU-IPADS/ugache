#ifndef SAMGRAPH_RUN_CONFIG_H
#define SAMGRAPH_RUN_CONIFG_H

#include <string>
#include <vector>

#include "common.h"
#include "cpu/cpu_common.h"
#include "logging.h"

namespace samgraph {
namespace common {

struct RunConfig {
  // Configs passed from application
  static std::string dataset_path;
  static RunArch run_arch;
  static SampleType sample_type;
  static std::vector<int> fanout;
  static size_t batch_size;
  static size_t num_epoch;
  static Context sampler_ctx;
  static Context trainer_ctx;
  static double cache_percentage;

  static cpu::CPUHashType cpu_hash_type;

  // Environment variables
  static bool option_profile_cuda;
  static bool option_log_node_access;
  static bool option_sanity_check;

  static size_t kPipelineDepth;
  static int kOMPThreadNum;

  static inline bool UseGPUCache() { return cache_percentage > 0; }
  static void LoadConfigFromEnv();
};

}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_RUN_CONFIG_H
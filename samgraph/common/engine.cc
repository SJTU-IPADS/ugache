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

#include "engine.h"

#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <parallel/algorithm>
#include <parallel/numeric>

#include "common.h"
#include "constant.h"
#include "dist/dist_engine.h"
#include "logging.h"
#include "profiler.h"
#include "run_config.h"
#include "timer.h"
#include "device.h"
#include "utils.h"

namespace samgraph {
namespace common {

namespace {

void shuffle(uint32_t * data, size_t num_data, const size_t* shuffle_range=nullptr, uint64_t seed= 0x1234567890abcdef) {
  auto g = std::default_random_engine(seed);
  if (shuffle_range == nullptr) {
    shuffle_range = & num_data;
  }
  for (size_t i = 0; i < *shuffle_range; i++) {
    std::uniform_int_distribution<size_t> d(i, num_data - 1);
    size_t candidate = d(g);
    std::swap(data[i], data[candidate]);
  }
}

};

Engine* Engine::_engine = nullptr;

void Engine::Create() {
  if (_engine) {
    return;
  }

  switch (RunConfig::run_arch) {
    case kArch0:
      break;
    case kArch1:
    case kArch2:
    case kArch3:
    case kArch4:
    case kArch7:
      break;
    case kArch5:
    case kArch6:
    case kArch9:
      LOG(INFO) << "Use Dist Engine (Arch " << RunConfig::run_arch << ")";
      _engine = new dist::DistEngine();
      break;
    default:
      CHECK(0);
  }
}

void Engine::LoadGraphDataset() {
  Timer t;
  // Load graph dataset from disk by mmap and copy the graph
  // topology data into the target CUDA device.
  _dataset = new Dataset;
  std::unordered_map<std::string, size_t> meta;
  std::unordered_map<std::string, Context> ctx_map = GetGraphFileCtx();

  // default feature type is 32-bit float.
  // legacy dataset doesnot have this meta
  DataType feat_data_type = kF32;

  if (_dataset_path.back() != '/') {
    _dataset_path.push_back('/');
  }

  // Parse the meta data
  std::ifstream meta_file(_dataset_path + Constant::kMetaFile);
  std::string line;
  while (std::getline(meta_file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> kv{std::istream_iterator<std::string>{iss},
                                std::istream_iterator<std::string>{}};

    if (kv.size() < 2) {
      break;
    }

    if (kv[0] == Constant::kMetaFeatDataType) {
      feat_data_type = DataTypeParseName(kv[1]);
    } else {
      meta[kv[0]] = std::stoull(kv[1]);
    }
  }

  CHECK(meta.count(Constant::kMetaNumNode) > 0);
  CHECK(meta.count(Constant::kMetaNumEdge) > 0);
  CHECK(meta.count(Constant::kMetaFeatDim) > 0);
  CHECK(meta.count(Constant::kMetaNumClass) > 0);

  CHECK(ctx_map.count(Constant::kIndptrFile) > 0);
  CHECK(ctx_map.count(Constant::kIndicesFile) > 0);
  CHECK(ctx_map.count(Constant::kFeatFile) > 0);
  CHECK(ctx_map.count(Constant::kLabelFile) > 0);
  CHECK(ctx_map.count(Constant::kTrainSetFile) > 0);
  CHECK(ctx_map.count(Constant::kTestSetFile) > 0);
  CHECK(ctx_map.count(Constant::kValidSetFile) > 0);
  CHECK(ctx_map.count(Constant::kAliasTableFile) > 0);
  CHECK(ctx_map.count(Constant::kProbTableFile) > 0);
  CHECK(ctx_map.count(Constant::kInDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kOutDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByDegreeFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByHeuristicFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByDegreeHopFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByFakeOptimalFile) > 0);
  CHECK(ctx_map.count(Constant::kCacheByRandomFile) > 0);

  _dataset->num_node = meta[Constant::kMetaNumNode];
  _dataset->num_edge = meta[Constant::kMetaNumEdge];
  _dataset->num_class = meta[Constant::kMetaNumClass];

  if (RunConfig::option_fake_feat_dim != 0) {
    meta[Constant::kMetaFeatDim] = RunConfig::option_fake_feat_dim;
    _dataset->feat = Tensor::EmptyNoScale(feat_data_type,
                                          {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
                                          ctx_map[Constant::kFeatFile], "dataset.feat");
  } else if (FileExist(_dataset_path + Constant::kFeatFile) && RunConfig::option_empty_feat == 0) {
    _dataset->feat = Tensor::FromMmap(
        _dataset_path + Constant::kFeatFile, feat_data_type,
        {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
        ctx_map[Constant::kFeatFile], "dataset.feat");
  } else {
    if (RunConfig::option_empty_feat != 0) {
      _dataset->feat = Tensor::EmptyNoScale(
          feat_data_type,
          {1ull << RunConfig::option_empty_feat, meta[Constant::kMetaFeatDim]},
          ctx_map[Constant::kFeatFile], "dataset.feat");
    } else {
      _dataset->feat = Tensor::EmptyNoScale(
          feat_data_type,
          {meta[Constant::kMetaNumNode], meta[Constant::kMetaFeatDim]},
          ctx_map[Constant::kFeatFile], "dataset.feat");
    }
  }

  if (FileExist(_dataset_path + Constant::kLabelFile)) {
    _dataset->label =
        Tensor::FromMmap(_dataset_path + Constant::kLabelFile, DataType::kI64,
                         {meta[Constant::kMetaNumNode]},
                         ctx_map[Constant::kLabelFile], "dataset.label");
  } else {
    _dataset->label =
        Tensor::EmptyNoScale(DataType::kI64, {meta[Constant::kMetaNumNode]},
                             ctx_map[Constant::kLabelFile], "dataset.label");
  }

  if (RunConfig::UseGPUCache()) {
    switch (RunConfig::cache_policy) {
      case kCliquePartByDegree:
      case kCacheByDegree:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByDegreeFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByDegreeFile], "dataset.ranking_nodes");
        break;
      case kCacheByHeuristic:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByHeuristicFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByHeuristicFile], "dataset.ranking_nodes");
        break;
      case kCacheByPreSample:
      case kCacheByPreSampleStatic:
        break;
      case kCacheByDegreeHop:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByDegreeHopFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByDegreeHopFile], "dataset.ranking_nodes");
        break;
      case kCacheByFakeOptimal:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByFakeOptimalFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByFakeOptimalFile], "dataset.ranking_nodes");
        break;
      case kCacheByRandom:
        _dataset->ranking_nodes = Tensor::FromMmap(
            _dataset_path + Constant::kCacheByRandomFile, DataType::kI32,
            {meta[Constant::kMetaNumNode]},
            ctx_map[Constant::kCacheByRandomFile], "dataset.ranking_nodes");
        break;
      case kDynamicCache:
      case kCollCache:
      case kCollCacheIntuitive:
      case kCollCacheAsymmLink:
      case kPartitionCache:
      case kPartRepCache:
      case kRepCache:
      case kCliquePart:
        break;
      default:
        CHECK(0);
    }
  }

  double loading_time = t.Passed();
  LOG(INFO) << "SamGraph loaded dataset(" << _dataset_path << ") successfully ("
            << loading_time << " secs)";
  LOG(DEBUG) << "dataset(" << _dataset_path << ") has "
             << _dataset->num_node << " nodes, "
             << _dataset->num_edge << " edges ";
}

bool Engine::IsAllThreadFinish(int total_thread_num) {
  int k = _joined_thread_cnt.fetch_add(0);
  return (k == total_thread_num);
};

void Engine::ForwardBarrier() {
  outer_counter++;
}
void Engine::ForwardInnerBarrier() {
  inner_counter++;
}

}  // namespace common
}  // namespace samgraph

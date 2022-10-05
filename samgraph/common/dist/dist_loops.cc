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

#include "dist_loops.h"

#include "../device.h"
#include "../logging.h"
#include "../profiler.h"
#include "../run_config.h"
#include "../timer.h"

#include "dist_engine.h"
#include "../cuda/cuda_utils.h"
#include "../cpu/cpu_utils.h"

namespace samgraph {
namespace common {
namespace dist {

#ifdef SAMGRAPH_COLL_CACHE_ENABLE
void DoCollFeatLabelExtract(TaskPtr task) {
  static size_t first_batch_num_input = 0;
  auto dataset = DistEngine::Get()->GetGraphDataset();

  auto feat = dataset->feat;
  auto label = dataset->label;

  auto feat_dim = dataset->feat->Shape()[1];
  auto feat_type = dataset->feat->Type();
  auto label_type = dataset->label->Type();

  auto input_nodes  = reinterpret_cast<const IdType *>(task->input_nodes->Data());
  auto output_nodes = reinterpret_cast<const IdType *>(task->output_nodes->Data());
  auto num_input = task->input_nodes->Shape()[0];
  auto num_ouput = task->output_nodes->Shape()[0];
  auto train_ctx = DistEngine::Get()->GetTrainerCtx();
  auto copy_stream = DistEngine::Get()->GetTrainerCopyStream();
  // shuffler make first 1 batch larger. make sure all later batch is smaller
  // to avoid wasted memory
  if (first_batch_num_input == 0) {
    first_batch_num_input = RoundUp<size_t>(num_input, 8);
  } else {
    CHECK_LE(num_input, first_batch_num_input) << "first batch is smaller than this one" << first_batch_num_input << ", " << num_input;
  }
  task->input_feat = Tensor::EmptyNoScale(feat_type, {first_batch_num_input, feat_dim}, DistEngine::Get()->GetTrainerCtx(),
                    "task.train_feat_cuda_" + std::to_string(task->key));
  task->input_feat->ForceScale(feat_type, {num_input, feat_dim}, DistEngine::Get()->GetTrainerCtx(), "task.train_feat_cuda_" + std::to_string(task->key));
  task->output_label = Tensor::Empty(label_type, {num_ouput}, DistEngine::Get()->GetTrainerCtx(),
                     "task.train_label_cuda_" + std::to_string(task->key));
  DistEngine::Get()->GetCollCacheManager()->ExtractFeat(input_nodes, task->input_nodes->Shape()[0], task->input_feat->MutableData(), copy_stream, task->key);
  Timer t_label;
  {
    CHECK_EQ(label_type, kF32);
    auto label_dst = task->output_label->Ptr<float>();
    size_t num_pos = task->unsupervised_positive_edges;
    size_t num_neg = num_ouput - num_pos;
    cuda::ArrangeArray<float>(label_dst, num_pos, 1, 0, copy_stream);
    cuda::ArrangeArray<float>(label_dst + num_pos, num_neg, 0, 0, copy_stream);
    Device::Get(train_ctx)->StreamSync(train_ctx, copy_stream);
  }
  double time_label = t_label.Passed();
  Profiler::Get().LogStep(task->key, kLogL3LabelExtractTime, time_label);
  Profiler::Get().LogStep(task->key, kLogL1LabelBytes, task->output_label->NumBytes());
  LOG(DEBUG) << "CollFeatExtract: process task with key " << task->key;
}
#endif

} // dist
} // common
} // samgraph

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

#ifndef SAMGRAPH_DIST_ENGINE_H
#define SAMGRAPH_DIST_ENGINE_H

#include <atomic>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include "../common.h"
#include "../cuda/cuda_utils.h"
#include "../engine.h"
#include "../logging.h"
#include "collaborative_cache_manager.h"

namespace samgraph {
namespace common {
namespace dist {

class DistSharedBarrier {
 public:
  DistSharedBarrier(int count);
  ~ DistSharedBarrier() {
    munmap(_barrier_ptr, sizeof(pthread_barrier_t));
  }
  void Wait();
 private:
  pthread_barrier_t* _barrier_ptr;
};

enum class DistType {Sample = 0, Extract, Switch, Default};

class DistEngine : public Engine {
 public:
  DistEngine();

  void Init() override;
  void Start() override;
  void Shutdown() override;
  void SampleInit(int worker_id, Context ctx);
  void TrainInit(int worker_id, Context ctx, DistType dist_type);
  /**
   * @param count: the total times to loop extract
   */
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
  CollCacheManager* GetCollCacheManager() { return _coll_cache_manager; }
  CollCacheManager* GetCollLabelManager() { return _coll_label_manager; }
#endif
  DistType GetDistType() { return _dist_type; }
  StreamHandle GetTrainerCopyStream() { return _trainer_copy_stream; }

  DistSharedBarrier* GetSamplerBarrier() { return _sampler_barrier; }
  DistSharedBarrier* GetTrainerBarrier() { return _trainer_barrier; }
  DistSharedBarrier* GetGlobalBarrier() { return _global_barrier; }

  static DistEngine* Get() { return dynamic_cast<DistEngine*>(Engine::_engine); }

 private:
  // Task queue
  std::vector<std::thread*> _threads;

  StreamHandle _trainer_copy_stream;
#ifdef SAMGRAPH_COLL_CACHE_ENABLE
  // Collaborative cache manager
  CollCacheManager* _coll_cache_manager;
  CollCacheManager* _coll_label_manager;
#endif

  void ArchCheck() override;
  std::unordered_map<std::string, Context> GetGraphFileCtx() override;
  // Dist type: Sample or Extract
  DistType _dist_type;

  DistSharedBarrier *_sampler_barrier;
  DistSharedBarrier *_trainer_barrier;
  DistSharedBarrier *_global_barrier;
};

}  // namespace dist
}  // namespace common
}  // namespace samgraph

#endif  // SAMGRAPH_DIST_ENGINE_H

/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 */

#pragma once

#include <curand_kernel.h>

namespace cuco {

namespace {
__device__ __forceinline__ unsigned int GlobalThreadId() {
  unsigned int smid;
  unsigned int warpid;
  unsigned int laneid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return smid * 2048 + warpid * 32 + laneid;
}

}  // namespace

class initializer {
  curandState *states_;

 public:
  initializer(curandState *states) : states_(states) {}

  __device__ float operator()() const {
    float val = curand_uniform(states_ + GlobalThreadId());
    val = (val - 0.5) * 0.1;
    return val;
  }
};

}  // namespace cuco
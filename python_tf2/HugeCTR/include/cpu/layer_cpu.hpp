/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cpu_resource.hpp>
#include <fstream>
#include <functional>
#include <general_buffer2.hpp>
#include <string>
#include <vector>

namespace HugeCTR {
/**
 * @brief
 * Definition of a basic layer class.
 */
class LayerCPU {
 protected:
  /*
   * stores the weight tensors of this layer.
   */
  Tensors2<float> weights_;

 public:
  /*
   * Forward pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void fprop(bool is_train) = 0;
  /*
   * Backward pass
   * @param stream: the CUDA stream that the forward function will be executed on.
   */
  virtual void bprop() = 0;

  LayerCPU() {}
  LayerCPU(const LayerCPU&) = delete;
  LayerCPU& operator=(const LayerCPU&) = delete;
  virtual ~LayerCPU() = default;

  /*
   * Some of the layers requires initialize like fully connected layer
   */
  virtual void initialize() {}
};

}  // namespace HugeCTR

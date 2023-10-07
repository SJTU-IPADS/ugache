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

#include "config.h"
#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class NopDep : public OpKernel {
 public:
  explicit NopDep(OpKernelConstruction *ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext *ctx) override {

    Tensor const *dense = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("dense", &dense));
    Tensor const *emb = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb", &emb));

    // do forward propagation
    try {
      ctx->set_output(0, *dense);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
};

REGISTER_KERNEL_BUILDER(Name("NopDep").Device(DEVICE_GPU),
                        NopDep<GPUDevice>);

}  // namespace tensorflow
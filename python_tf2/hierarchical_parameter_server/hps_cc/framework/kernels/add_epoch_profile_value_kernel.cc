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

#include <cstdint>
#include <exception>

#include "config.h"
#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class AddEpochProfileValue : public OpKernel {
 public:
  explicit AddEpochProfileValue(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
    // OP_REQUIRES(ctx, global_batch_size_ > 0,
    //             errors::Aborted(__FILE__, ":", __LINE__, " ", "global_batch_size must be > 0."));

    // OP_REQUIRES_OK(ctx, ctx->GetAttr("ps_config_file", &ps_config_file_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const Tensor* profile_type_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("profile_type", &profile_type_tensor));
    const Tensor* value_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("value", &value_tensor));
    try {
      int global_replica_id = global_replica_id_tensor->scalar<int32_t>()(0);
      int64_t profile_type = profile_type_tensor->scalar<int64_t>()(0);
      double value = value_tensor->scalar<double>()(0);

      auto device_ctx = ctx->op_device_context();
      OP_REQUIRES(ctx, device_ctx == nullptr, errors::Aborted("should have no device context."));

      HierarchicalParameterServer::Facade::instance()->add_epoch_profile_value(global_replica_id,
                                                                               profile_type, value);
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("AddEpochProfileValue")
                            .Device(DEVICE_CPU)
                            .HostMemory("global_replica_id")
                            .HostMemory("profile_type")
                            .HostMemory("value"),
                        AddEpochProfileValue<CPUDevice>);

}  // namespace tensorflow

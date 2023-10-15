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
#include "lookup_manager.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class Lookup : public OpKernel {
 public:
  explicit Lookup(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("emb_vec_size", &emb_vec_size_));
  }

  void Compute(OpKernelContext *ctx) override {
    // Tensor const *status_tensor = nullptr;
    // OP_REQUIRES_OK(ctx, ctx->input("init_status", &status_tensor));
    // std::string init_status = status_tensor->flat<tstring>()(0);
    // OP_REQUIRES(ctx, init_status == "OK",
    //             errors::Aborted("hierarchical parameter server is not initialized."));

    Tensor const *values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

    Tensor const *global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const int32_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();

    // allocate output
    Tensor *emb_vector_tensor = nullptr;
    TensorShape emb_vector_tensor_shape = values_tensor->shape();
    emb_vector_tensor_shape.AppendShape({emb_vec_size_});

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape, &emb_vector_tensor));

    // do forward propagation
    try {
      coll_cache_lib::LookupManager::instance()->forward(
          model_name_.c_str(), table_id_, global_replica_id_value, values_tensor, emb_vector_tensor,
          ctx);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  std::string model_name_;
  tensorflow::int32 table_id_;
  tensorflow::int32 emb_vec_size_;
};

REGISTER_KERNEL_BUILDER(Name("LookupColl").Device(DEVICE_GPU).HostMemory("global_replica_id"),
                        Lookup<GPUDevice>);

template <typename Device>
class RecordHotness : public OpKernel {
 public:
  explicit RecordHotness(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_name", &model_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("table_id", &table_id_));
  }

  void Compute(OpKernelContext *ctx) override {
    // Tensor const *status_tensor = nullptr;
    // OP_REQUIRES_OK(ctx, ctx->input("init_status", &status_tensor));
    // std::string init_status = status_tensor->flat<tstring>()(0);
    // OP_REQUIRES(ctx, init_status == "OK",
    //             errors::Aborted("hierarchical parameter server is not initialized."));

    Tensor const *values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));

    Tensor const *global_replica_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
    const int32_t global_replica_id_value = global_replica_id_tensor->scalar<int32_t>()();

    // do freq record
    try {
      coll_cache_lib::LookupManager::instance()->record_hotness(
          model_name_.c_str(), table_id_, global_replica_id_value, values_tensor, ctx);
    } catch (std::exception const &error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }

 private:
  std::string model_name_;
  tensorflow::int32 table_id_;
};

REGISTER_KERNEL_BUILDER(Name("RecordHotnessColl").Device(DEVICE_GPU).HostMemory("global_replica_id"),
                        RecordHotness<GPUDevice>);

}  // namespace tensorflow
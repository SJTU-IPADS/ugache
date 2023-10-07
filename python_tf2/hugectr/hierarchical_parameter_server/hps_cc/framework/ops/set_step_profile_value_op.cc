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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("SetStepProfileValue")
    .Input("global_replica_id: int32")
    .Input("profile_type: int64")
    .Input("value: double")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle input_shape_0 = ctx->input(0);
      DimensionHandle input_num_elem_0 = ctx->NumElements(input_shape_0);
      if (1 != ctx->Value(input_num_elem_0))
        return errors::InvalidArgument("global_replica_id must be a scalar.");

      ShapeHandle input_shape_1 = ctx->input(1);
      DimensionHandle input_num_elem_1 = ctx->NumElements(input_shape_1);
      if (1 != ctx->Value(input_num_elem_1))
        return errors::InvalidArgument("profile_type must be a scalar.");

      ShapeHandle input_shape_2 = ctx->input(2);
      DimensionHandle input_num_elem_2 = ctx->NumElements(input_shape_2);
      if (1 != ctx->Value(input_num_elem_2))
        return errors::InvalidArgument("value must be a scalar.");
      return Status::OK();
    });

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

#include <loss.hpp>
#include <metrics.hpp>
#include <optimizer.hpp>
#include <parser.hpp>
#include <regularizers/l1_regularizer.hpp>
#include <regularizers/l2_regularizer.hpp>
#include <regularizers/no_regularizer.hpp>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

OptParams get_optimizer_param(const nlohmann::json& j_optimizer) {
  // create optimizer
  auto optimizer_name = get_value_from_json<std::string>(j_optimizer, "type");
  Optimizer_t optimizer_type = Optimizer_t::DEFAULT;
  if (!find_item_in_map(optimizer_type, optimizer_name, OPTIMIZER_TYPE_MAP)) {
    HCTR_OWN_THROW(Error_t::WrongInput, "No such optimizer: " + optimizer_name);
  }

  OptHyperParams opt_hyper_params;
  OptParams opt_params;

  Update_t update_type = Update_t::Local;
  if (has_key_(j_optimizer, "update_type")) {
    std::string update_name = get_value_from_json<std::string>(j_optimizer, "update_type");
    if (!find_item_in_map(update_type, update_name, UPDATE_TYPE_MAP)) {
      HCTR_OWN_THROW(Error_t::WrongInput, "No such update type: " + update_name);
    }
  } else if (has_key_(j_optimizer, "global_update")) {
    bool global_update = get_value_from_json<bool>(j_optimizer, "global_update");
    if (global_update) update_type = Update_t::Global;
  } else {
    HCTR_LOG(INFO, ROOT, "update_type is not specified, using default: local\n");
  }

  switch (optimizer_type) {
    case Optimizer_t::Adam: {
      auto j_hparam = get_json(j_optimizer, "adam_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float beta1 = get_value_from_json<float>(j_hparam, "beta1");
      float beta2 = get_value_from_json<float>(j_hparam, "beta2");
      float epsilon = get_value_from_json<float>(j_hparam, "epsilon");
      opt_hyper_params.adam.beta1 = beta1;
      opt_hyper_params.adam.beta2 = beta2;
      opt_hyper_params.adam.epsilon = epsilon;
      opt_params = {Optimizer_t::Adam, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::AdaGrad: {
      auto j_hparam = get_json(j_optimizer, "adagrad_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float initial_accu_value = get_value_from_json<float>(j_hparam, "initial_accu_value");
      float epsilon = get_value_from_json<float>(j_hparam, "epsilon");
      opt_hyper_params.adagrad.initial_accu_value = initial_accu_value;
      opt_hyper_params.adagrad.epsilon = epsilon;
      opt_params = {Optimizer_t::AdaGrad, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::MomentumSGD: {
      auto j_hparam = get_json(j_optimizer, "momentum_sgd_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float momentum_factor = get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.momentum.factor = momentum_factor;
      opt_params = {Optimizer_t::MomentumSGD, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::Nesterov: {
      auto j_hparam = get_json(j_optimizer, "nesterov_hparam");
      float learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      float momentum_factor = get_value_from_json<float>(j_hparam, "momentum_factor");
      opt_hyper_params.nesterov.mu = momentum_factor;
      opt_params = {Optimizer_t::Nesterov, learning_rate, opt_hyper_params, update_type};
      break;
    }
    case Optimizer_t::SGD: {
      auto j_hparam = get_json(j_optimizer, "sgd_hparam");
      auto learning_rate = get_value_from_json<float>(j_hparam, "learning_rate");
      if (has_key_(j_hparam, "atomic_update")) {
        opt_hyper_params.sgd.atomic_update = get_value_from_json<bool>(j_hparam, "atomic_update");
      }
      opt_params = {Optimizer_t::SGD, learning_rate, opt_hyper_params, update_type};
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  return opt_params;
}

}  // namespace HugeCTR

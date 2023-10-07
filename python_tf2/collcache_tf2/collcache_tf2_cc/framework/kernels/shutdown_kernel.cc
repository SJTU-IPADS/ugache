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

#include <sys/wait.h>

#include <exception>

#include "config.h"
#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace {

std::string ToReadableSize(size_t nbytes) {
  constexpr size_t kGigabytes = 1ull << 30;
  constexpr size_t kMegabytes = 1ull << 20;
  constexpr size_t kKilobytes = 1ull << 10;
  char buf[100];
  if (nbytes > kGigabytes) {
    double new_size = (float)nbytes / kGigabytes;
    sprintf(buf, "%.2lf GB", new_size);
    return std::string(buf);
  } else if (nbytes > kMegabytes) {
    double new_size = (float)nbytes / kMegabytes;
    sprintf(buf, "%.2lf MB", new_size);
    return std::string(buf);
  } else if (nbytes > kKilobytes) {
    double new_size = (float)nbytes / kKilobytes;
    sprintf(buf, "%.2lf KB", new_size);
    return std::string(buf);
  } else {
    double new_size = (float)nbytes;
    sprintf(buf, "%.2lf Bytes", new_size);
    return std::string(buf);
  }
}
};

template <typename Device>
class Shutdown : public OpKernel {
 public:
  explicit Shutdown(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    try {
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      std::stringstream ss;
      ss << "[CUDA] worker" << std::getenv("HPS_WORKER_ID") << " cuda mem usage: " << ToReadableSize(total  - free) << "\n";
      std::cout << ss.str();
      HierarchicalParameterServer::Facade::instance()->report_cache();
      if (std::string(std::getenv("HPS_WORKER_ID")) == "0") {
        HierarchicalParameterServer::Facade::instance()->report_avg();
      }
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }

    Tensor* status_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &status_tensor));
    status_tensor->flat<tstring>()(0) = "OK";
  }
};

REGISTER_KERNEL_BUILDER(Name("Shutdown").Device(DEVICE_GPU).HostMemory("status"),
                        Shutdown<GPUDevice>);

extern "C" {

int wait_one_child() {
  int child_stat;
  pid_t pid = waitpid(-1, &child_stat, 0);
  if (WEXITSTATUS(child_stat) != 0) {
    std::cerr << "detect a terminated child " << pid << ", status is " << WEXITSTATUS(child_stat)
              << "\n";
    return 1;
  } else if (WIFSIGNALED(child_stat)) {
    std::cerr << "detect an abnormal terminated child, signal is "
              << strsignal(WTERMSIG(child_stat));
    return 1;
  } else
    return 0;
}
}

}  // namespace tensorflow

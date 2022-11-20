#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../logging.h"
#include "../device.h"

namespace coll_cache_lib {
namespace common {
namespace cuda {

inline void check_have_affinity_support(int dev_o) {
  
  CUdevice dev;
  CU_CALL(cuDeviceGet(&dev, dev_o));
  int rst;
  CU_CALL(cuDeviceGetExecAffinitySupport(&rst, CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT, dev));
  if (rst != 1) {
    std::cerr << "No affinity support! Please enable MPS\n";
    abort();
  }
}
#define check_current_ctx_is(ctx) { \
  CUcontext cctx;                   \
  CU_CALL(cuCtxGetCurrent(&cctx));  \
  CHECK(cctx == (ctx));             \
}
#define check_top_ctx_is(ctx) { \
  CUcontext cctx;                  \
  CU_CALL(cuCtxPopCurrent(&cctx)); \
  CHECK(cctx == (ctx));            \
  CU_CALL(cuCtxPushCurrent(cctx)); \
}
inline CUcontext create_ctx_with_sm_count(int dev_o, int sm_count) {
  // LOG(ERROR) << "creating mps ctx at dev " << dev_o << " with sm=" << sm_count;
  CUdevice dev;
  CU_CALL(cuDeviceGet(&dev, dev_o));
  CUcontext sctx;
  CUexecAffinityParam param;
  param.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
  param.param.smCount.val = sm_count; // this value cannot exceed `total amount of sm`
  check_have_affinity_support(dev_o);
  CU_CALL(cuCtxCreate_v3(&sctx, &param, 1, 0, dev));
  check_current_ctx_is(sctx);
  check_top_ctx_is(sctx);
  return sctx;
}

inline size_t query_sm_count(int dev_o) {
  CUdevice dev;
  CU_CALL(cuDeviceGet(&dev, dev_o));
  int sm_count;
  CU_CALL(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  return sm_count;
}
inline int query_affinity_sm_count(int dev_o) {
  CUexecAffinityParam param;
  CU_CALL(cuCtxGetExecAffinity(&param, CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT));
  return param.param.smCount.val;
}

#define check_primary_ctx_active(dev_o) {                    \
  CUdevice dev;                                              \
  CU_CALL(cuDeviceGet(&dev, 0));                             \
  unsigned int flags = 0;                                    \
  int active = 0;                                            \
  CU_CALL(cuDevicePrimaryCtxGetState(dev, &flags, &active)); \
  CHECK(active == 1) << "primary ctx is not active";         \
}

}
}
}
#pragma push_macro("LOG")
#pragma push_macro("CHECK")

#include "coll_cache_lib/logging.h"

// #define LOG_LEVELS
#undef CHECK
#define COLL_CHECK(x) \
  if (!(x))      \
  common::LogMessageFatal(__FILE__, __LINE__) << "Check failed: " #x << ' '

// #define CHECK_LT
// #define CHECK_GT
// #define CHECK_LE
// #define CHECK_GE
// #define CHECK_EQ
// #define CHECK_NE
// #define CHECK_NOTNULL
// #define CUDA_CALL
// #define CU_CALL
// #define CUSPARSE_CALL
// #define NCCLCHECK
// #define _LOG_TRACE
// #define _LOG_DEBUG
// #define _LOG_INFO
// #define _LOG_WARNING
// #define _LOG_ERROR
// #define _LOG_FATAL
// #define _LOG
// #define _LOG_RANK
// #define GET_LOG
#undef LOG
#define COLL_LOG(...) GET_LOG(__VA_ARGS__, _LOG_RANK, _LOG)(__VA_ARGS__)
namespace coll_cache_lib {
using common::LogLevel;
}
// #define DEBUG_PREFIX
// #define WARNING_PREFIX

#pragma pop_macro("CHECK")
#pragma pop_macro("LOG")
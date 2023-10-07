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

#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <base/debug/logger.hpp>
#include <chrono>
#include <common.hpp>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <resource_managers/resource_manager_ext.hpp>
#include <string>
#include <thread>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

thread_local std::string THREAD_NAME;

const std::string& hctr_get_thread_name() { return THREAD_NAME; }

void hctr_set_thread_name(const std::string& name) { THREAD_NAME = name; }

void Logger::print_exception(const std::exception& e, int depth) {
  Logger::get().log(LOG_ERROR_LEVEL, true, false, "%d. %s\n", depth, e.what());
  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception& e) {
    print_exception(e, depth + 1);
  } catch (...) {
  }
}

Logger& Logger::get() {
  static std::unique_ptr<Logger> instance;
  static std::once_flag once_flag;

  std::call_once(once_flag, []() { instance.reset(new Logger()); });
  return *instance;
}

Logger::~Logger() {
  // if stdout and stderr are in use, we don't do fclose to prevent the situations where
  //   (1) the fds are taken in opening other files or
  //   (2) writing to the closed fds occurs, which is UB.
  // Due to the same reason, we don't wrap FILE* with a smart pointer.
  if (log_to_file_) {
    for (int level = LOG_ERROR_LEVEL; level <= max_level_; level++) {
      if (level != LOG_SILENCE_LEVEL) {
        fclose(log_file_[level]);
      }
    }
  }
}

void Logger::log(const int level, bool per_rank, bool with_prefix, const char* format, ...) const {
  if (level == LOG_SILENCE_LEVEL || level > max_level_) {
    return;
  }

  if (rank_ == 0 || per_rank) {
    std::ostringstream os;
    if (with_prefix) {
      write_log_prefix(os, level);
    }
    os << format;
    const std::string& new_format = os.str();

    if (log_to_std_) {
      va_list args;
      va_start(args, format);
      FILE* const file = log_std_.at(level);
      vfprintf(file, new_format.c_str(), args);
      va_end(args);
      fflush(file);
    }

    if (log_to_file_) {
      va_list args;
      va_start(args, format);
      FILE* const file = log_file_.at(level);
      vfprintf(file, new_format.c_str(), args);
      va_end(args);
      fflush(file);
    }
  }
}

Logger::DeferredEntry::~DeferredEntry() {
  if (logger_) {
    if (logger_->log_to_std_) {
      FILE* const file = logger_->log_std_.at(level_);
      fputs(os_.str().c_str(), file);
      fflush(file);
    }

    if (logger_->log_to_file_) {
      FILE* const file = logger_->log_file_.at(level_);
      fputs(os_.str().c_str(), file);
      fflush(file);
    }
  }
}

Logger::DeferredEntry Logger::log(const int level, bool per_rank, bool with_prefix) const {
  if (level == LOG_SILENCE_LEVEL || level > max_level_) {
    return {nullptr, level, per_rank, false};
  } else if (rank_ == 0 || per_rank) {
    return {this, level, per_rank, with_prefix};
  } else {
    return {nullptr, level, per_rank, false};
  }
}

void Logger::abort(const SrcLoc& loc, const char* const format, ...) const {
  if (format) {
    std::string hint;
    {
      va_list args;
      va_start(args, format);
      hint.resize(vsnprintf(nullptr, 0, format, args) + 1);
      va_end(args);
    }
    {
      va_list args;
      va_start(args, format);
      vsprintf(hint.data(), format, args);
      va_end(args);
    }
    log(-1, true, true,
        "Check Failed!\n"
        "\tFile: %s:%u\n"
        "\tFunction: %s\n"
        "\tExpression: %s\n"
        "\tHint: %s\n",
        loc.file, loc.line, loc.func, loc.expr, hint.c_str());
  } else {
    log(-1, true, true,
        "Check Failed!\n"
        "\tFile: %s:%u\n"
        "\tFunction: %s\n"
        "\tExpression: %s\n",
        loc.file, loc.line, loc.func, loc.expr);
  }
  std::abort();
}

void Logger::do_throw(HugeCTR::Error_t error_type, const SrcLoc& loc,
                      const std::string& message) const {
  std::ostringstream os;
  os << "Runtime error: " << message << std::endl;
  os << '\t' << loc.expr << " at " << loc.func << '(' << loc.file << ':' << loc.line << ')';
  std::throw_with_nested(internal_runtime_error(error_type, os.str()));
}

int Logger::get_rank() { return rank_; }

#ifdef HCTR_LEVEL_MAP_
#error HCTR_LEVEL_MAP_ already defined!
#else
#define HCTR_LEVEL_MAP_(MAP, NAME) MAP[LOG_LEVEL(NAME)] = #NAME
#endif

Logger::Logger() : rank_(0), max_level_(DEFAULT_LOG_LEVEL), log_to_std_(true), log_to_file_(false) {
  hctr_set_thread_name("main");

#ifdef ENABLE_MPI
  MPILifetimeService::init();
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank_) != MPI_SUCCESS) {
    std::cerr << "MPI rank initialization failed!" << std::endl;
    std::abort();
  }
#endif

  const char* const max_level_str = std::getenv("HUGECTR_LOG_LEVEL");
  if (max_level_str != nullptr && max_level_str[0] != '\0') {
    int max_level;
    if (sscanf(max_level_str, "%d", &max_level) == 1) {
      max_level_ = max_level;
    }
  }

  const char* const log_to_file_str = std::getenv("HUGECTR_LOG_TO_FILE");
  if (log_to_file_str != nullptr && log_to_file_str[0] != '\0') {
    int log_to_file_val = 0;
    if (sscanf(log_to_file_str, "%d", &log_to_file_val) == 1) {
      log_to_std_ = log_to_file_val < 2;
      log_to_file_ = log_to_file_val > 0;
    }
  }

  HCTR_LEVEL_MAP_(level_name_, ERROR);
  HCTR_LEVEL_MAP_(level_name_, SILENCE);
  HCTR_LEVEL_MAP_(level_name_, INFO);
  HCTR_LEVEL_MAP_(level_name_, WARNING);
  HCTR_LEVEL_MAP_(level_name_, DEBUG);
  HCTR_LEVEL_MAP_(level_name_, TRACE);

  if (log_to_file_) {
    for (int level = LOG_ERROR_LEVEL; level <= max_level_; level++) {
      if (level != LOG_SILENCE_LEVEL) {
        std::string level_name = level_name_[level];
        std::transform(level_name.begin(), level_name.end(), level_name.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        std::string log_fname = "hctr_" + std::to_string(getpid()) + "_" + std::to_string(rank_) +
                                "_" + level_name + ".log";
        log_file_[level] = fopen(log_fname.c_str(), "w");
      } else {
        log_file_[LOG_SILENCE_LEVEL] = nullptr;
      }
    }
  }

  if (log_to_std_) {
    log_std_[LOG_ERROR_LEVEL] = stderr;
    log_std_[LOG_SILENCE_LEVEL] = nullptr;
    for (int level = LOG_INFO_LEVEL; level <= max_level_; level++) {
      log_std_[level] = stdout;
    }
  }
}

void Logger::write_log_prefix(std::ostringstream& os, const int level) const {
  // Base & time
  os << "[HCTR][";
  {
    struct timeval now;
    gettimeofday(&now, nullptr);
    std::tm now_local;
    localtime_r(&now.tv_sec, &now_local);

    // %H:%M:%S = [00-23]:[00-59]:[00-60] == e.g., 23:59:60 = 8 bytes + 1 zero terminate.
    // (60 = for second-time-shift years)
    char buffer[8 + 1];
    std::strftime(buffer, sizeof(buffer), "%T", &now_local);
    os << buffer << '.' << std::setfill('0') << std::setw(3) << now.tv_usec / 1000;
  }

  // Level
  os << "][";
  {
    const auto& level_it = level_name_.find(level);
    if (level_it != level_name_.end()) {
      os << level_it->second;
    } else {
      os << "LEVEL" << level;
    }
  }

  // Rank
  os << "][RK" << rank_;

  // Thread
  os << "][";
  const std::string& thread_name = hctr_get_thread_name();
  if (thread_name.empty()) {
    os << "tid #" << std::this_thread::get_id();
  } else {
    os << thread_name;
  }

  // Prompt & return.
  os << "]: ";
}

}  // namespace HugeCTR

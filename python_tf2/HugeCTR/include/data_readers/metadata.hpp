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

#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "HugeCTR/include/common.hpp"

namespace HugeCTR {

struct FileStats {
  long long num_rows;
  FileStats(long long num_rows) : num_rows(num_rows) {}
#ifdef ENABLE_ARROW_PARQUET
  long long num_groups;
  std::vector<long long> row_groups_offset;  //
  FileStats(long long num_rows, long long num_groups, std::vector<long long> row_groups_offset)
      : num_rows(num_rows), num_groups(num_groups), row_groups_offset(row_groups_offset) {}

#endif
};

struct Cols {
  std::string col_name;
  int index;
};
/**
 * @brief
 * Definition of a basic layer class.
 */
class Metadata {
 private:
  std::vector<Cols> cat_names_;
  std::vector<Cols> cont_names_;
  std::vector<Cols> label_names_;
  std::unordered_map<std::string, FileStats> file_stats_;
  bool loaded_;
  long long num_rows_total_files_;
  std::vector<long long> rows_file_offset_;

 public:
  // ctor
  Metadata()
      : cat_names_(),
        cont_names_(),
        label_names_(),
        file_stats_(),
        loaded_(false),
        num_rows_total_files_(0),
        rows_file_offset_(){};

  // initialize everything
  void get_parquet_metadata(std::string file_name);

  std::vector<Cols> get_cat_names() { return this->cat_names_; }
  std::vector<Cols> get_cont_names() { return this->cont_names_; }
  std::vector<Cols> get_label_names() { return this->label_names_; }
  std::vector<long long> get_rows_file_offset() { return this->rows_file_offset_; }
  FileStats get_file_stats(std::string file_name) {
    FileStats fs(0);
    try {
      fs = this->file_stats_.at(file_name);
    } catch (const std::runtime_error& rt_err) {
      HCTR_LOG_S(ERROR, WORLD) << "getting file" << file_name << " stats error" << std::endl;
      HCTR_OWN_THROW(Error_t::BrokenFile, "failed to get file stats");
      throw;
    }
    return fs;
  }
  bool get_metadata_status() { return loaded_; };
  long long get_num_rows_total_files() { return num_rows_total_files_; }
};
}  // namespace HugeCTR

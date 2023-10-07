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

#include <memory>
#include <string>
#include <vector>

namespace HugeCTR {
class FileSystem {
 public:
  FileSystem() = default;

  FileSystem(const FileSystem&) = delete;

  virtual ~FileSystem() = default;

  FileSystem& operator=(const FileSystem&) = delete;

  /**
   * @brief Get the file size of target file from the remote file system
   *
   * @param path Remote file path.
   * @return File size in bytes.
   */
  virtual size_t get_file_size(const std::string& path) const = 0;

  /**
   * @brief Create a dir in the remote file system
   *
   * @param path Remote directory path.
   */
  virtual void create_dir(const std::string& path) = 0;

  /**
   * @brief Delete a file or files in directory from the remote file system
   *
   * @param path Remote path.
   * @param recursive wheter to delete recursively
   */
  virtual void delete_file(const std::string& path, bool recursive = true) = 0;

  /**
   * @brief Copy a file from remote file system to Local
   *
   * @param source_path Remote file path.
   * @param target_path Local path.
   */
  virtual void fetch(const std::string& source_path, const std::string& target_path) = 0;

  /**
   * @brief Copy a file from local file system to remote file system
   *
   * @param source_path
   * @param target_path
   */
  virtual void upload(const std::string& source_path, const std::string& target_path) = 0;

  /**
   * @brief Write file from butter to the specified path in the file system
   *
   * @param path Remote path of the file to write into.
   * @param data The data stream to write.
   * @param data_size The size of the data stream.
   * @param overwrite Whether to overwrite or append.
   * @return Number of successfully written bytes.
   */
  virtual int write(const std::string& path, const void* data, size_t data_size,
                    bool overwrite) = 0;

  /**
   * @brief Read file from the file system to the buffer
   *
   * @param path Remote path of the file from which to read.
   * @param buffer Buffer to hold the read data.
   * @param buffer_size The number of bytes to read.
   * @param offset Offset within the file from which to start reading.
   * @return Number of successfully read bytes.
   */
  virtual int read(const std::string& path, void* buffer, size_t buffer_size, size_t offset) = 0;

  /**
   * @brief Copy a specific file within a file system.
   *
   * @param source_file Source file path.
   * @param target_file Target file path.
   */
  virtual void copy(const std::string& source_file, const std::string& target_file) = 0;

  /**
   * @brief Copy all files under the remote source directory to local target directory.
   *
   * @param source_dir Source dir path.
   * @param target_dir Target dir path.
   * @return Number of files copied.
   */
  virtual int batch_fetch(const std::string& source_dir, const std::string& target_dir) = 0;

  /**
   * @brief Copy all files under the local source directory to remote target directory.
   *
   * @param source_dir
   * @param target_dir
   * @return Number of files copied.
   */
  virtual int batch_upload(const std::string& source_dir, const std::string& target_dir) = 0;
};

enum class DataSourceType_t { Local, HDFS, S3, Other };

struct DataSourceParams {
  DataSourceType_t type;
  std::string server;
  int port;

  DataSourceParams(const DataSourceType_t type, const std::string& server, const int port)
      : type(type), server(server), port(port){};
  DataSourceParams() : type(DataSourceType_t::Local), server("localhost"), port(9000){};

  FileSystem* create() const;

  std::unique_ptr<FileSystem> create_unique() const {
    return std::unique_ptr<FileSystem>{create()};
  }
};

}  // namespace HugeCTR
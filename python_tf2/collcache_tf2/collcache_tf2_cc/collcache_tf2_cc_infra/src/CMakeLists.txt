#
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#

cmake_minimum_required(VERSION 3.8)
set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDAToolkit)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPSTL_USE_PARALLEL_POLICIES=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../build
)

file(GLOB_RECURSE collcache_tf2_cc_infra_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc *.cpp *.cu)
add_library(collcache_tf2_cc_infra SHARED ${collcache_tf2_cc_infra_src})
target_link_libraries(collcache_tf2_cc_infra PUBLIC rt ${CUDA_LIBRARIES} ${TF_LINK_FLAGS} coll_cache)

# target_compile_features(huge_ctr_hps PUBLIC cxx_std_17)
set_target_properties(collcache_tf2_cc_infra PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(collcache_tf2_cc_infra PROPERTIES CUDA_ARCHITECTURES OFF)

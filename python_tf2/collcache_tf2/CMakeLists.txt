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
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.17)
set(CUDA_SEPARABLE_COMPILATION ON)
project(CollCacheTF LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakes)

find_package(CUDA REQUIRED)
find_package(TensorFlow REQUIRED)
find_package(Threads)

set(CUDA_SEPARABLE_COMPILATION ON)

# setting compiler flags
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Setting default CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

foreach(arch_name ${SM})
    if (arch_name STREQUAL 80 OR 
        arch_name STREQUAL 75 OR 
        arch_name STREQUAL 70 OR 
        arch_name STREQUAL 61 OR 
        arch_name STREQUAL 60)
        list(APPEND cuda_arch_list ${arch_name})
        message(STATUS "Assign GPU architecture (sm=${arch_name})")
    else()
        message(FATAL_ERROR "Unknown or unsupported GPU architecture (set sm=70)")
    endif()
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(${cuda_arch_list_length} EQUAL 0)
    list(APPEND cuda_arch_list "70")
    message(STATUS "Assign default GPU architecture sm=70")
endif()
list(REMOVE_DUPLICATES cuda_arch_list)

foreach(arch_name ${cuda_arch_list})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch_name},code=sm_${arch_name}")
endforeach()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DPSTL_USE_PARALLEL_POLICIES=0")
if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -Wall -Werror -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses -Xcudafe --display_error_number -Xcudafe --diag_suppress=177")
else()
    set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Werror -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable -Wno-maybe-uninitialized -Wno-format-truncation -Wno-sign-compare -Wno-error=stringop-overflow=")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall,-Werror,-Wno-error=cpp,-Wno-error=parentheses,-Wno-sign-compare")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# setting install folder
if(SKBUILD)
    # if we're building with skbuild, then we shouldn't be installing to /usr/local/lib
    # install the libraries alongside the python source in collcache_tf2_lib/lib instead
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION collcache_tf2)
else()
    install(DIRECTORY ${CMAKE_BINARY_DIR}/lib DESTINATION /usr/local)
endif()

# headers
include_directories(
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/collcache_tf2_cc_infra/include/
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/framework/kernels/
    ${PROJECT_SOURCE_DIR}/../../
    ${PROJECT_SOURCE_DIR}/../../3rdparty/json/single_include/
    ${CUDA_INCLUDE_DIRS}
)

# configure_file(${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/../HugeCTR/include/config.hpp)
# add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party third_party)
# building HPS C++ source code
# add_subdirectory(${PROJECT_SOURCE_DIR}/../gpu_cache/src gpu_cache)
# add_subdirectory(${PROJECT_SOURCE_DIR}/../HugeCTR/src/hps hps)

# setting tensorflow compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_COMPILE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGOOGLE_CUDA=1 ${TF_COMPILE_FLAGS}")

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess")
    endif()
else()
    if (${TF_VERSION_MAJOR} EQUAL 1)
        set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-error=class-memaccess")
    endif()
endif()

# libs
link_directories(
    ${TF_LINK_DIR}
    ${PROJECT_SOURCE_DIR}/../../build
)

add_subdirectory(collcache_tf2_cc/collcache_tf2_cc_infra/src)

# code sources related to framework
file(GLOB framework
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/framework/kernels/*.cpp
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/framework/kernels/*.cc
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/framework/kernels/*.cu
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/framework/ops/*.cpp
    ${PROJECT_SOURCE_DIR}/collcache_tf2_cc/framework/ops/*.cc
)

# build dynamic lib
add_library(collcache_tf2_lib SHARED ${framework})

target_link_libraries(collcache_tf2_lib PUBLIC ${TF_LINK_FLAGS} coll_cache collcache_tf2_cc_infra rt)
target_compile_features(collcache_tf2_lib PUBLIC cxx_std_17)
set_target_properties(collcache_tf2_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(collcache_tf2_lib PROPERTIES CUDA_ARCHITECTURES OFF)


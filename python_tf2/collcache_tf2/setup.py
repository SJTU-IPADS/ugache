"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import sys

from setuptools import find_packages
from skbuild import setup

# Package meta-data.
NAME = 'collcache_tf'
DESCRIPTION = 'A high-performance embedding cache'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '3.0.0'

def get_cmake_args():
    gpu_capabilities = ["70", "75", "80"]

    cmake_build_type = "Release"
    # cmake_build_type = "Debug"

    cmake_args = [
        "-DSM='{}'".format(";".join(gpu_capabilities)),
        "-DCMAKE_BUILD_TYPE={}".format(cmake_build_type),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    return cmake_args


setup(
    name=NAME,
    version=VERSION,
    # author="NVIDIA",
    # author_email="hugectr-dev@exchange.nvidia.com",
    # url="https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/hierarchical_parameter_server",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    extras_require={"tensorflow": "tensorflow>=1.15"},
    license="Apache 2.0",
    platforms=["Linux"],
    python_requires=">=3",
    packages=find_packages(),
    cmake_args=get_cmake_args(),
)

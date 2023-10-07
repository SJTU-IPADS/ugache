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
import ctypes
from tensorflow.python.framework import load_library
from tensorflow import __version__ as tf_version

if tf_version.startswith("2"):
    using_tf2 = True
elif tf_version.startswith("1"):
    using_tf2 = False
else:
    raise RuntimeError("Not supported TF version: {}".format(tf_version))


def in_tensorflow2():
    """
    This function will tell whether the installed TensorFlow is 2.x
    """
    return using_tf2


lib_name = r"libhierarchical_parameter_server.so"
install_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
paths = [r"/usr/local/lib", install_path]

lib_file = None
for path in paths:
    try:
        filename = os.path.join(path, lib_name)
        file = open(filename)
        file.close()
        lib_file = os.path.join(path, lib_name)
        break
    except FileNotFoundError:
        continue

if lib_file is None:
    raise FileNotFoundError("Could not find %s" % lib_name)
hps_ops = load_library.load_op_library(lib_file)
hps_clib = ctypes.CDLL(lib_file, mode=ctypes.RTLD_GLOBAL)
lookup = hps_ops.lookup
init = hps_ops.init
shutdown = hps_ops.shutdown
nop_dep = hps_ops.nop_dep
set_step_profile_value=hps_ops.set_step_profile_value
add_epoch_profile_value=hps_ops.add_epoch_profile_value
wait_one_child=hps_clib.wait_one_child
wait_one_child.restype = ctypes.c_int
# wait_one_child=hps_ops.wait_one_child
# wait_one_child.restype = ctypes.c_int

"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load all the necessary PyTorch C types.
import torch

from collcache.common import *
from collcache.common import _basics
from collcache.torch import c_lib

config = _basics.config
# num_local_step = _basics.num_local_step
# feat_dim = _basics.feat_dim
# num_epoch = _basics.num_epoch
# steps_per_epoch = _basics.steps_per_epoch
log_step_by_key = _basics.log_step_by_key
report_step = _basics.report_step
report_step_average = _basics.report_step_average
# train_barrier = _basics.train_barrier
wait_one_child = _basics.wait_one_child
print_memory_usage = _basics.print_memory_usage
coll_cache_record_init = _basics.coll_cache_record_init

def coll_torch_test(local_id, keys):
  batch_feat = c_lib.coll_torch_test(local_id, keys)
  return batch_feat
def coll_torch_lookup_key_t_val_ret(local_id, keys):
  batch_feat = c_lib.coll_torch_lookup_key_t_val_ret(local_id, keys)
  return batch_feat

def coll_torch_record(local_id, keys):
  c_lib.coll_torch_record(local_id, keys)

def coll_torch_init_t(replica_id, dev_id, emb, cache_percentage):
  c_lib.coll_torch_init_t(replica_id, dev_id, emb, cache_percentage)

def coll_torch_create_emb_shm(replica_id, n_keys, dim, dtype):
  return c_lib.coll_torch_create_emb_shm(replica_id, n_keys, dim, dtype)
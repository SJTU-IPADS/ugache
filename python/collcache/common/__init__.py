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

import ctypes
import os
import sysconfig


def _get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def _get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + _get_ext_suffix())
    return full_path


def _get_next_enum_val(next_val):
    res = next_val[0]
    next_val[0] += 1
    return res

kCacheByDegree          = 0
kCacheByHeuristic       = 1
kCacheByPreSample       = 2
kCacheByDegreeHop       = 3
kCacheByPreSampleStatic = 4
kCacheByFakeOptimal     = 5
kDynamicCache           = 6
kCacheByRandom          = 7
kCollCache              = 8
kCollCacheIntuitive     = 9
kPartitionCache         = 10
kPartRepCache           = 11
kRepCache               = 12
kCollCacheAsymmLink     = 13
kCliquePart             = 14
kCliquePartByDegree     = 15


def cpu(device_id=0):
    return 'cpu:{:}'.format(device_id)


def gpu(device_id=0):
    return 'cuda:{:}'.format(device_id)

cache_policies = {
    'degree'                : kCacheByDegree,
    'heuristic'             : kCacheByHeuristic,
    'pre_sample'            : kCacheByPreSample,
    'degree_hop'            : kCacheByDegreeHop,
    'presample_static'      : kCacheByPreSampleStatic,
    'fake_optimal'          : kCacheByFakeOptimal,
    'dynamic_cache'         : kDynamicCache,
    'random'                : kCacheByRandom,
    'coll_cache'            : kCollCache,
    'coll_intuitive'        : kCollCacheIntuitive,
    'partition'             : kPartitionCache,
    'part_rep'              : kPartRepCache,
    'rep'                   : kRepCache,
    'coll_cache_asymm_link' : kCollCacheAsymmLink,
    'clique_part'           : kCliquePart,
    'clique_part_by_degree' : kCliquePartByDegree,
}

_step_log_val = [0]

# Step L1 Log
kLogL1NumSample        = _get_next_enum_val(_step_log_val)
kLogL1NumNode          = _get_next_enum_val(_step_log_val)
kLogL1SampleTotalTime  = _get_next_enum_val(_step_log_val)
kLogL1SampleTime       = _get_next_enum_val(_step_log_val)
kLogL1SendTime         = _get_next_enum_val(_step_log_val)
kLogL1RecvTime         = _get_next_enum_val(_step_log_val)
kLogL1CopyTime         = _get_next_enum_val(_step_log_val)
kLogL1ConvertTime      = _get_next_enum_val(_step_log_val)
kLogL1TrainTime        = _get_next_enum_val(_step_log_val)
kLogL1FeatureBytes     = _get_next_enum_val(_step_log_val)
kLogL1LabelBytes       = _get_next_enum_val(_step_log_val)
kLogL1IdBytes          = _get_next_enum_val(_step_log_val)
kLogL1GraphBytes       = _get_next_enum_val(_step_log_val)
kLogL1MissBytes        = _get_next_enum_val(_step_log_val)
kLogL1RemoteBytes      = _get_next_enum_val(_step_log_val)
kLogL1PrefetchAdvanced = _get_next_enum_val(_step_log_val)
kLogL1GetNeighbourTime = _get_next_enum_val(_step_log_val)
kLogL1SamplerId        = _get_next_enum_val(_step_log_val)
# Step L2 Log
kLogL2ShuffleTime    = _get_next_enum_val(_step_log_val)
kLogL2LastLayerTime  = _get_next_enum_val(_step_log_val)
kLogL2LastLayerSize  = _get_next_enum_val(_step_log_val)
kLogL2CoreSampleTime = _get_next_enum_val(_step_log_val)
kLogL2IdRemapTime    = _get_next_enum_val(_step_log_val)
kLogL2GraphCopyTime  = _get_next_enum_val(_step_log_val)
kLogL2IdCopyTime     = _get_next_enum_val(_step_log_val)
kLogL2ExtractTime    = _get_next_enum_val(_step_log_val)
kLogL2FeatCopyTime   = _get_next_enum_val(_step_log_val)
kLogL2CacheCopyTime  = _get_next_enum_val(_step_log_val)
# Step L3 Log
kLogL3KHopSampleCooTime          = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleKernelTime       = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleSortCooTime      = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleCountEdgeTime    = _get_next_enum_val(_step_log_val)
kLogL3KHopSampleCompactEdgesTime = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkSampleCooTime    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKTime         = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep1Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep2Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep3Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep4Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep5Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep6Time    = _get_next_enum_val(_step_log_val)
kLogL3RandomWalkTopKStep7Time    = _get_next_enum_val(_step_log_val)
kLogL3RemapFillUniqueTime        = _get_next_enum_val(_step_log_val)
kLogL3RemapPopulateTime          = _get_next_enum_val(_step_log_val)
kLogL3RemapMapNodeTime           = _get_next_enum_val(_step_log_val)
kLogL3RemapMapEdgeTime           = _get_next_enum_val(_step_log_val)
kLogL3CacheGetIndexTime          = _get_next_enum_val(_step_log_val)
KLogL3CacheCopyIndexTime         = _get_next_enum_val(_step_log_val)
kLogL3CacheExtractMissTime       = _get_next_enum_val(_step_log_val)
kLogL3CacheCopyMissTime          = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineMissTime       = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineCacheTime      = _get_next_enum_val(_step_log_val)
kLogL3CacheCombineRemoteTime     = _get_next_enum_val(_step_log_val)
kLogL3LabelExtractTime           = _get_next_enum_val(_step_log_val)

# Epoch Log
_epoch_log_val = [0]
kLogEpochSampleTime                  = _get_next_enum_val(_epoch_log_val)
KLogEpochSampleGetCacheMissIndexTime = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleSendTime              = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleTotalTime             = _get_next_enum_val(_epoch_log_val)
kLogEpochCoreSampleTime              = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleCooTime               = _get_next_enum_val(_epoch_log_val)
kLogEpochIdRemapTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochShuffleTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochSampleKernelTime            = _get_next_enum_val(_epoch_log_val)
kLogEpochCopyTime                    = _get_next_enum_val(_epoch_log_val)
kLogEpochConvertTime                 = _get_next_enum_val(_epoch_log_val)
kLogEpochTrainTime                   = _get_next_enum_val(_epoch_log_val)
kLogEpochTotalTime                   = _get_next_enum_val(_epoch_log_val)
kLogEpochFeatureBytes                = _get_next_enum_val(_epoch_log_val)
kLogEpochMissBytes                   = _get_next_enum_val(_epoch_log_val)

class CollCacheBasics(object):
    def __init__(self, pkg_path, *args):
        full_path = _get_extension_full_path(pkg_path, *args)
        self.C_LIB_CTYPES = ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
        # coll_cache_config
        # coll_cache_config_from_map
        # coll_cache_num_epoch
        # coll_cache_feat_dim
        # coll_cache_log_step_by_key
        # coll_cache_report_step_by_key
        # coll_cache_report_step_average_by_key
        # coll_cache_train_barrier
        # coll_cache_wait_one_child
        # coll_cache_print_memory_usage
        # coll_cache_lookup

        self.C_LIB_CTYPES.coll_cache_log_step_by_key.argtypes = (
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_double
        )
        self.C_LIB_CTYPES.coll_cache_report_step_by_key.argtypes = (
            ctypes.c_uint64,)
        self.C_LIB_CTYPES.coll_cache_report_step_average_by_key.argtypes = (
            ctypes.c_uint64,)

        self.C_LIB_CTYPES.coll_cache_record_init.argtypes = (
            ctypes.c_int,)

        # self.C_LIB_CTYPES.coll_cache_feat_dim.restype = ctypes.c_size_t
        # self.C_LIB_CTYPES.coll_cache_num_epoch.restype = ctypes.c_size_t

        # self.C_LIB_CTYPES.samgraph_num_local_step.restype = ctypes.c_size_t
        self.C_LIB_CTYPES.coll_cache_wait_one_child.restype = ctypes.c_int

    def config(self, run_config : dict):
        num_configs_items = len(run_config)
        config_keys = [str.encode(str(key)) for key in run_config.keys()]
        config_values = []
        for value in run_config.values():
            if isinstance(value, list):
                config_values.append(str.encode(
                    ' '.join([str(v) for v in value])))
            else:
                config_values.append(str.encode(str(value)))


        return self.C_LIB_CTYPES.coll_cache_config(
            (ctypes.c_char_p * num_configs_items)(
                *config_keys
            ),
            (ctypes.c_char_p * num_configs_items) (
                *config_values
            ),
            ctypes.c_size_t(num_configs_items)
        )

    # def num_local_step(self):
    #     return self.C_LIB_CTYPES.samgraph_num_local_step()

    # def feat_dim(self):
    #     return self.C_LIB_CTYPES.samgraph_feat_dim()

    # def num_epoch(self):
    #     return self.C_LIB_CTYPES.samgraph_num_epoch()

    # def steps_per_epoch(self):
    #     return self.C_LIB_CTYPES.samgraph_steps_per_epoch()

    def log_step_by_key(self, key, item, val):
        return self.C_LIB_CTYPES.coll_cache_log_step_by_key(key, item, val)

    def report_step(self, key):
        return self.C_LIB_CTYPES.coll_cache_report_step_by_key(key)

    def report_step_average(self, key):
        return self.C_LIB_CTYPES.coll_cache_report_step_average_by_key(key)

    # def train_barrier(self):
    #     return self.C_LIB_CTYPES.samgraph_train_barrier()

    def wait_one_child(self):
        return self.C_LIB_CTYPES.coll_cache_wait_one_child()

    def print_memory_usage(self):
        return self.C_LIB_CTYPES.coll_cache_print_memory_usage()
    def coll_cache_record_init(self, replica_id):
        self.C_LIB_CTYPES.coll_cache_record_init(replica_id)

_basics = CollCacheBasics(__file__, "c_lib")
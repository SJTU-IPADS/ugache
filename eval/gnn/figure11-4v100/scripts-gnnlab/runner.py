import os, sys, copy
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0]))+'/../../common_gnnlab')
from runner_helper import Arch, RunConfig, ConfigList, App, Dataset, CachePolicy, TMP_LOG_DIR, SampleType, percent_gen, reverse_percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('root_path', ['/datasets_gnn/gnnlab/'])
  .override('amp', [True])
  .override('copy_job', [1])
  .override('sample_job', [1])
  .override('epoch', [3])
  .override('empty_feat', [25])
  .override('num_sampler', [1])
  .override('num_trainer', [3])
  .override('coll_cache_scale', [16])
  .override('logdir', [
    'run-logs',
  ])
  .override('profile_level', [3])
  .override('log_level', ['warn'])
  .override('multi_gpu', [True])
  .override('pipeline', [
    True,
    # False,
  ]))

cfg_list_collector = ConfigList.Empty()

'''
GraphSage
'''
# 1.1 unsup
cur_common_base = (cur_common_base.copy().override('app', [App.graphsage       ]).override('sample_type', [SampleType.kKHop2]))
cur_common_base = (cur_common_base.copy().override('batch_size', [2000]))
cur_common_base.override('custom_env', [f'SAMGRAPH_MQ_SIZE={70*1024*1024}'])
cur_common_base = (cur_common_base.copy().override('unsupervised', [True]).override('max_num_step', [1000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.19])) # should be 0.20
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.14])) # should be 0.16
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.02])) # should be 0.02

# 1.2 sup
cur_common_base = (cur_common_base.copy().override('batch_size', [8000]))
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]).override('max_num_step', [100000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.20])) # should be 0.21
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.14])) # should be 0.16
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.02])) # should be 0.02


cfg_list_collector.hyper_override(
  ['cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.rep_2, "DIRECT", ""],
  ])

if __name__ == '__main__':
  from sys import argv
  fail_only = False
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log)

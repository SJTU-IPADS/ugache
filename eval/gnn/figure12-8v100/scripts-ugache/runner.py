import os, sys, copy
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0]))+'/../../common')
from runner_helper import Model, Dataset, CachePolicy, ConfigList

do_mock = False
durable_log = True
fail_only = False

cur_common_base = (ConfigList()
  .override('root_path', ['/datasets_gnn/wholegraph'])
  .override('logdir', ['run-logs',])
  .override('num_workers', [8])
  .override('epoch', [4])
  .override('skip_epoch', [2])
  .override('presc_epoch', [2])
  .override('use_amp', [True])
  .override('empty_feat', [24])
  .override('omp_thread_num', [48])
  .override('num_intra_size', [4])
  .override('custom_env', [' COLL_BLOCK_SLICE_GRAIN_BY_CACHE=50 COLL_BLOCK_SLICE_BASE=2 COLL_GUROBI_EXP_PARAM=1 COLL_REMOTE_TIME_OVERRIDE=22.5'])
  )

cfg_list_1 = ConfigList.Empty()
cfg_list_2 = ConfigList.Empty()

'''
GraphSage
'''
# 1.1 unsup, large batch
cur_common_base = (cur_common_base.copy().override('model', [Model.sage]).override('unsupervised', [True]))
cur_common_base = (cur_common_base.copy().override('batchsize', [8000]).override('local_step', [125]))
cfg_list_1.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.25]).override('batchsize', [4000]))
cfg_list_1.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.25]).override('batchsize', [4000]))
cfg_list_2.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.04]).override('batchsize', [2000]))


# 1.1 sup, large batch
cur_common_base = (cur_common_base.copy().override('unsupervised', [False]))
cur_common_base = (cur_common_base.copy().override('batchsize', [8000]))
cfg_list_1.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', [0.25]))
cfg_list_1.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', [0.25]))
cfg_list_2.concat(cur_common_base.copy().override('dataset', [Dataset.mag240m_homo,     ]).override('cache_percent', [0.04]))

cfg_list_1.hyper_override(
  ['use_collcache', 'cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [False, CachePolicy.coll_cache_asymm_link, "", ""],
    [True, CachePolicy.coll_cache_asymm_link, "", ""],
])
cfg_list_2.hyper_override(
  ['use_collcache', 'cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [True, CachePolicy.clique_part, "", ""],
    [True, CachePolicy.coll_cache_asymm_link, "", ""],
])

cfg_list_collector = ConfigList.Empty()
cfg_list_collector.concat(cfg_list_1)
cfg_list_collector.concat(cfg_list_2)
cfg_list_collector.override('coll_cache_scale', [
  # 0,
  16,
])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log, fail_only=fail_only)
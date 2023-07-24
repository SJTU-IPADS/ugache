import os, sys, copy
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0]))+'/../common')
from runner_helper import Model, Dataset, CachePolicy, ConfigList, percent_gen as pg

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
  .override('use_amp', [False])
  .override('empty_feat', [24])
  .override('omp_thread_num', [56])
  .override('custom_env', [
    'COLL_MIN_FREQ=0.001 ',
  ])
  )

cfg_list_collector = ConfigList.Empty()

'''
GraphSage
'''
cur_common_base = (cur_common_base.copy().override('model', [Model.sage]).override('unsupervised', [True]))
cur_common_base = (cur_common_base.copy().override('batchsize', [2000]).override('local_step', [125]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', pg(1,2,1) + pg(4,20,4) + [0.25, 0.40]).override('use_amp', [False]).override('batchsize', [4000]))
# cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', pg(1,2,1) + pg(4,20,4) + [0.25, 0.40]).override('use_amp', [False]).override('batchsize', [4000]))

cur_common_base = (cur_common_base.copy().override('unsupervised', [False]))
cur_common_base = (cur_common_base.copy().override('batchsize', [8000]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.papers100M_undir, ]).override('cache_percent', pg(2,12,2)))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.friendster,       ]).override('cache_percent', pg(2,12,2)))

cfg_list_collector.hyper_override(
  ['use_collcache', 'cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [True, CachePolicy.clique_part, "", "MPSPhase"],
    [True, CachePolicy.rep, "", "MPSPhase"],
    [True, CachePolicy.coll_cache_asymm_link, "", ""],
])
cfg_list_collector.override('coll_cache_scale', [
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
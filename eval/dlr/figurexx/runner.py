import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('root_path', ['/datasets_dlr/'])
  .override('epoch', [3])
  # .override('epoch', [5])
  .override('gpu_num', [4])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('coll_cache_scale', [16])
  .override('model', [
    Model.dlrm, 
    Model.dcn,
  ])
  .override('empty_feat', [24])
  .override('custom_env', [
    # 'COLL_MIN_FREQ=0.02 ',
    'COLL_MIN_FREQ=0.1 ',
    # 'COLL_MIN_FREQ=0.5 ',
  #   # 'COLL_HASH_IMPL=RR  COLL_SKIP_HASH=1',
  #   # 'COLL_HASH_IMPL=RR  COLL_SKIP_HASH=0',
  #   # 'COLL_HASH_IMPL=CHUNK  COLL_SKIP_HASH=1',
  #   # 'COLL_HASH_IMPL=CHUNK  COLL_SKIP_HASH=0',
  ])
  )

cfg_list_collector = ConfigList.Empty()

'''
Coll Cache
'''
(cur_common_base
  .override('system', [System.collcache])
  .override('global_batch_size', [32768])
  .override('plain_dense_model', [True])
  .override('mock_embedding', [True])
)

cfg_list_collector.concat(cur_common_base.copy().hyper_override(
    ["random_request", "alpha", "dataset"],
    [
      # [True,  0.3,  RandomDataset("simple_power0.3_slot100_C800m", "SP_03_S100_C800m", 800000000, 100)],
      [True,  0.2,  RandomDataset("simple_power0.2_slot100_C800m", "SP_02_S100_C800m", 800000000, 100)],
      [False,  None,  Dataset.criteo_tb],
    ]
  )
  .override('cache_percent', 
    # [0.01] + 
    [0.02] + 
    []
  ).hyper_override(
  ['coll_cache_policy', "coll_cache_concurrent_link", "sok_use_hashtable"], 
  [
    [CachePolicy.clique_part,           "",         None],
    # [CachePolicy.clique_part,           "MPSPhase", None],
    [CachePolicy.rep_cache,             "",         None],
    # [CachePolicy.rep_cache,             "MPSPhase", None],
    [CachePolicy.coll_cache_asymm_link, "",         None],
    # [CachePolicy.coll_cache_asymm_link, "DIRECT",   None],
    [CachePolicy.hps,                   "",         None],
    [CachePolicy.sok,                   "",         True],
  ]))


# selector for fast validation
(cfg_list_collector
  # .select('cache_percent', [
  #   # 0.01,
  # ])
  # .select('coll_cache_policy', [
  #   # CachePolicy.coll_cache_asymm_link,
  #   # CachePolicy.clique_part,
  #   # CachePolicy.rep_cache,
  # ])
  # .select('coll_cache_no_group', [
  #   # 'DIRECT',
  #   # '',
  # ])
  # .select('model', [Model.dlrm
  # ])
  # .select('dataset', [Dataset.criteo_tb
  # ])
  # .override('custom_env', ["SAMGRAPH_EMPTY_FEAT=10"])
  )

if __name__ == '__main__':
  from sys import argv
  retry = False
  fail_only = False
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
    elif arg == '-r' or arg == '--retry':
      retry = True
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log, retry=retry, fail_only=fail_only)
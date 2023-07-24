import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, ConfigList

do_mock = False
durable_log = True
retry = False
fail_only = False

cur_common_base = (ConfigList()
  .override('root_path', ['/datasets_dlr/'])
  .override('epoch', [2])
  .override('gpu_num', [8])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('coll_cache_scale', [16])
  .override('model', [Model.dlrm, Model.dcn,])
  .override('system', [System.collcache])
  .override('global_batch_size', [65536])
  .override('plain_dense_model', [True])
  .override('mock_embedding', [True])
  .override('random_request', [False])
  .override('custom_env', ['SAMGRAPH_EMPTY_FEAT=24'])
)

cfg_list_collector = ConfigList.Empty()

'''
DLRM && DCN
'''
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_tb]).override('cache_percent', [0.04]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.syn]).override('cache_percent', [0.02]))

cfg_list_collector.hyper_override(
  ['coll_cache_policy', 'coll_cache_no_group', 'coll_cache_concurrent_link', 'sok_use_hashtable'], 
  [
    [CachePolicy.coll_cache_asymm_link, '', 'MPSPhase', None],
    [CachePolicy.hps, '', '', None],
    [CachePolicy.sok, '', '', True],
  ]
)

if __name__ == '__main__':
  from sys import argv
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

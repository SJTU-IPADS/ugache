import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, ConfigList

do_mock = False
durable_log = True
retry = False
fail_only = False

cfg_list_collector = (ConfigList()
  .override('dataset_root_path', ['/datasets_dlr/processed/'])
  .override('epoch', [50])
  .override('gpu_num', [8])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('coll_cache_scale', [16])
  .override('model', [Model.dlrm,])
  .override('system', [System.collcache])
  .override('global_batch_size', [65536])
  .override('plain_dense_model', [True])
  .override('mock_embedding', [True])
  .override('random_request', [False])
  .override('cache_percent', [0.01])
  .override('custom_env', ['SAMGRAPH_EMPTY_FEAT=24'])
  .override('coll_cache_enable_refresh', [False, True])
  .override('coll_cache_refresh_iter', [20000])
  .override('coll_cache_refresh_seq_bucket_sz', [8000])
  .override('dataset', [Dataset.criteo_tb])
  .override('coll_cache_policy', [CachePolicy.coll_cache_asymm_link])
  .override('coll_cache_no_group', [''])
  .override('coll_cache_concurrent_link', ['MPSPhase'])
  .override('log_level', ['info'])
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
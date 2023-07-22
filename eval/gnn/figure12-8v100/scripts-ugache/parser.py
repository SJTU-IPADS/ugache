import os, sys
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0]))+'/../../common')
from common_parser import *
from runner_helper import *
from runner import cfg_list_collector

selected_col = ['short_app']
selected_col += ['policy_impl']
selected_col += ['dataset_short']
selected_col += ['epoch_e2e_time']

if __name__ == '__main__':
  if len(sys.argv) > 1:
    logdir = sys.argv[1]
    cfg_list_collector.override('logdir', [logdir])
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)
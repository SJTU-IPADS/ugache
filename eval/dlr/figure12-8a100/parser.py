import os, sys, math
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner_helper import *
from runner import cfg_list_collector

selected_col = ['short_app']
selected_col += ['policy_impl']
selected_col += ['dataset_short']
# selected_col += ['Step(average) L1 train total']
selected_col += ['Step(average) L2 feat copy']

if __name__ == '__main__':
  if len(sys.argv) > 1:
    logdir = sys.argv[1]
    cfg_list_collector.override('logdir', [logdir])
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0]))+'/../../common_gnnlab')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['short_app']
selected_col += ['system_short']
selected_col += ['dataset_short']
selected_col += ['train_process_time']

if __name__ == '__main__':
  if len(sys.argv) > 1:
    logdir = sys.argv[1]
    cfg_list_collector.override('logdir', [logdir])
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  with open(f'data.dat', 'a') as f:
    BenchInstance.print_dat(bench_list, f, selected_col, skip_header=True)
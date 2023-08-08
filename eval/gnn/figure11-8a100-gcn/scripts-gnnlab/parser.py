import os, sys
sys.path.append(os.path.dirname(os.path.abspath(sys.argv[0]))+'/../../common_gnnlab')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['short_app']
selected_col += ['system_short']
selected_col += ['dataset_short']
selected_col += ['train_process_time']

# selected_col = ['short_app']
# selected_col += ['dataset_short']
# selected_col += ['batch_size']
# selected_col += ['cache_percentage']
# selected_col += ['system_short']
# selected_col += ['train_process_time']
# selected_col += ['Step(average) L1 sample']
# selected_col += ['Step(average) L2 feat copy']
# selected_col += ['Step(average) L1 train total']
# selected_col += ['cuda_usage']
# selected_col += ['theory_cache_percent']
# selected_col += ['log_fname']

if __name__ == '__main__':
  if len(sys.argv) > 1:
    logdir = sys.argv[1]
    cfg_list_collector.override('logdir', [logdir])
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  # for inst in bench_list:
  #   inst.vals['cuda_usage'] = inst.get_val('mem.trainer.total') / 1024 / 1024 / 1024
  #   inst.vals['theory_cache_percent'] = (79 - float(inst.get_val('cuda_usage'))) / inst.cfg.dataset.feat_GB * 100 + inst.get_val('cache_percentage')
  #   inst.vals['log_fname'] = inst.cfg.get_log_fname() + '.log'
  with open(f'data.dat', 'a') as f:
    BenchInstance.print_dat(bench_list, f, selected_col, skip_header=True)
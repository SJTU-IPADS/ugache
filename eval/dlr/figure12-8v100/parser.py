import os, sys, math
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner_helper import *
from runner import cfg_list_collector

selected_col = ['app_short']
selected_col += ['policy_impl']
selected_col += ['dataset_short']
selected_col += ['step.copy']

if __name__ == '__main__':
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  for inst in bench_list:
    inst : BenchInstance
    try:
        if inst.get_val('cache_policy_short') == 'CollAsymm':
          inst.vals['policy_impl'] = 'UGache'
        else:
          inst.vals['policy_impl'] = inst.get_val('cache_policy_short')
        inst.vals['step.copy'] = max_nan(inst.get_val('Step(average) L2 cache feat copy'), inst.get_val('Step(average) L2 extract'))
    except Exception as e:
      print("Error when " + inst.cfg.get_log_fname() + '.log')
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)
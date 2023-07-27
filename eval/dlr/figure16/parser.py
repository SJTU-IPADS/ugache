import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector

selected_col = ['short_app']
selected_col += ['policy_impl', 'cache_percentage', 'global_batch_size']
selected_col += ['dataset_short']
selected_col += ['Sequence']
selected_col += ['Sequence(Average) extract time']
selected_col += ['Sequence(Average) e2e time']
selected_col += ['Sequence(Average) seq duration']
selected_col += ['coll_cache_refresh_seq_bucket_sz']
selected_col += ['coll_cache_enable_refresh']
selected_col += ['Python seq e2e time']

if __name__ == '__main__':
  if len(sys.argv) > 1:
    logdir = sys.argv[1]
    cfg_list_collector.override('logdir', [logdir])
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  seq_bench_list = []
  for inst in bench_list:
    inst : BenchInstance
    try:
      inst.vals['short_app'] = inst.cfg.model.name
      inst.vals['policy_impl'] = inst.get_val('coll_cache_concurrent_link') + inst.get_val('cache_policy_short')
      # get bucket num
      profile_steps = inst.get_val('epoch') * inst.get_val('iteration_per_epoch') * inst.get_val('gpu_num')
      bucket_num = profile_steps / inst.get_val('coll_cache_refresh_seq_bucket_sz')
      python_profile_per_bucket = (inst.get_val('coll_cache_refresh_seq_bucket_sz') / inst.get_val('gpu_num')) / 100

      # example: [Step(Seq_23) Profiler Level 3 E2 S7999]
      for i in range(int(bucket_num)):
        inst.vals['Sequence'] = i
        inst.vals['Sequence(Average) convert time'] = inst.vals[f'Step(Seq_{i}) L1 convert time']
        inst.vals['Sequence(Average) e2e time'] = inst.vals[f'Step(Seq_{i}) L1 train']
        inst.vals['Sequence(Average) extract time'] = inst.vals[f'Step(Seq_{i}) L2 cache feat copy']
        inst.vals['Sequence(Average) seq duration'] = inst.vals[f'Step(Seq_{i}) L1 seq duration']
        python_train_time = 0
        for k in range(int(python_profile_per_bucket)):
          python_profile_iter = int(100 * (python_profile_per_bucket * i + k + 1))
          python_train_time += inst.vals[f'{python_profile_iter}_iter']
        inst.vals['Python seq e2e time'] = python_train_time / python_profile_per_bucket
        seq_bench_list.append(copy.deepcopy(inst))
        
      if inst.vals['coll_cache_enable_refresh']:
        refresh_start_bucket = 19
        refresh_stop_bucket = (int(inst.vals['refresh_stop']) - 1000) // (inst.get_val('coll_cache_refresh_seq_bucket_sz') / inst.get_val('gpu_num'))
        refresh_stop_bucket = int(refresh_stop_bucket)
        
        idx_start = int(len(seq_bench_list) - bucket_num)
        refresh_start_time = seq_bench_list[idx_start + refresh_start_bucket].vals['Sequence(Average) seq duration']
        refresh_stop_time = seq_bench_list[idx_start + refresh_stop_bucket].vals['Sequence(Average) seq duration']
        
        with open("data_refresh.dat", 'w') as file:
          file.write(f'{refresh_start_time} {refresh_stop_time}\n')
    except Exception as e:
      # print(e)
      print("Error when " + inst.cfg.get_log_fname() + '.log')
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(seq_bench_list, f, selected_col)
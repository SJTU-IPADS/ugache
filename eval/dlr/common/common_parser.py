"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University
  
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

from runner_helper import RunConfig, ConfigList, System, Model, Dataset, CachePolicy, percent_gen
import copy
import re
import sys
import glob
import math
import traceback

size_unit_to_coefficient = {
  'GB':1024*1024*1024,
  'MB':1024*1024,
  'KB':1024,
  'Bytes':1,
  'B':1,
}

def div_nan(a,b):
  if b == 0:
    return math.nan
  return a/b

def max_nan(a,b):
  if math.isnan(a):
    return b
  elif math.isnan(b):
    return a
  else:
    return max(a,b)

def handle_nan(a, default=0):
  if math.isnan(a):
    return default
  return a

def zero_nan(a):
  return handle_nan(a, 0)


def sample_method_str_short(cfg):
  if cfg.app is App.gcn:
    num_hop = 3
  elif cfg.app is App.graphsage:
    num_hop = 2
  if cfg.sample_type is SampleType.kRandomWalk:
    return 'RndWlk'
  elif cfg.sample_type is SampleType.kKHop2:
    return str(num_hop)+ 'HopRnd'
  elif cfg.sample_type is SampleType.kWeightedKHopPrefix:
    return str(num_hop)+ 'HopWgt'

def my_assert_eq(a, b):
  if a != b:
    print(a, "!=", b)
    assert(False)

def grep_from(fname, pattern, line_ctx=[0,0]):
  p = re.compile(pattern)
  try:
    with open(fname) as f:
      lines = f.readlines()
      ret_lines = []
      if len(lines) == 0:
        return ret_lines
      for i in range(len(lines)):
        if p.match(lines[i]):
          ret_lines += lines[max(0, i - line_ctx[0]):min(len(lines), i + line_ctx[1] + 1)]
    return ret_lines
  except FileNotFoundError as e:
    print("error when ", fname)
    # print(traceback.format_exc())
    return []
  except Exception as e:
    print("error when ", fname)
    print(traceback.format_exc())
    return []

def filter_from(line_list, pattern):
  ret_lines = []
  p = re.compile(pattern)
  for line in line_list:
    if p.match(line):
      ret_lines.append(line)
  return ret_lines
def exclude_from(line_list, pattern):
  ret_lines = []
  p = re.compile(pattern)
  for line in line_list:
    if not p.match(line):
      ret_lines.append(line)
  return ret_lines


def div_nan(a,b):
  if b == 0:
    return math.nan
  return a/b

def max_nan(a,b):
  if math.isnan(a):
    return b
  elif math.isnan(b):
    return a
  else:
    return max(a,b)

def handle_nan(a, default=0):
  if math.isnan(a):
    return default
  return a
def zero_nan(a):
  return handle_nan(a, 0)

default_meta_list = ['system', 'model', 'dataset', 'gpu_num', 'global_batch_size', 
                      'cache_policy', 'cache_percentage', 'mock_embedding', 'max_vocabulary_size'
                      'step_sample_time', 'step_copy_time', 'step_convert_time', 'step_train_time',
                      'step_feature_KB', 'step_label_KB', 'step_id_KB', 'step_graph_KB', 'step_miss_KB']

class BenchInstance:
  def __init__(self):
    self.vals = {}
    self.fname = ""
  def init_from_cfg(self, cfg):
    try:
      cfg.handle_mock_params()
      fname = cfg.get_log_fname() + '.log'
      self.vals = {}
      self.fname = fname
      self.cfg = cfg
      # self.prepare_torch_summary(cfg)
      # self.prepare_sam_workspace(cfg)
      self.get_refresh_end(cfg)
      self.prepare_init(cfg)
      self.prepare_epoch_eval(cfg)
      self.prepare_coll_cache(cfg)
      self.prepare_profiler_log(cfg)
      self.prepare_config(cfg)
      self.prepare_percentage_log(cfg)
      self.prepare_coll_cache_meta(cfg)
      self.prepare_cuda_eval(cfg)
      self.prepare_python_eval(cfg)
      self.vals['optimal_hit_percent'] = self.get_optimal()
    except Exception as e:
      print("error when ", fname)
      print(traceback.format_exc())
      traceback.print_exc()
      # raise e
    return self

  def prepare_config(self, cfg):
    fname = cfg.get_log_fname() + '.log'
    l = fname.split('/')[-1].split('.')[0].split('_')
    i = iter(l)
    config_str_list = grep_from(fname, "\(\'.*", [0, 0])
    if len(config_str_list) > 0:
      # old fashion:
      # ('key', 'val')
      for cur_str in config_str_list:
        cur_str : str
        if cur_str.startswith("('"):
          key = cur_str.split(',')[0][2:-1]
          val = cur_str.split(',')[1].strip()[:-1]
          if val[0] == "'":
            val = val[1:-1]
          self.vals[key] = val
    else:
      config_str_list = grep_from(fname, "^config:.*", [0, 0])
      for cur_str in config_str_list:
        cur_str : str
        key = cur_str.split('=')[0][7:]
        val = cur_str.split('=')[1].strip()
        self.vals[key] = val
    for k,v in cfg.__dict__.items():
      self.vals[k] = v
    self.vals['system'] = cfg.system.name
    self.vals['model'] = cfg.model.name
    # self.vals['gpu_num'] = cfg.gpu_num
    # self.vals['global_batch_size'] = cfg.global_batch_size
    # self.vals['mock_embedding'] = cfg.mock_embedding
    # self.vals['max_vocabulary_size'] = cfg.max_vocabulary_size
    self.vals['dataset'] = str(cfg.dataset)
    self.vals['dataset_short'] = cfg.dataset.short()
    self.vals['cache_policy'] = str(cfg.coll_cache_policy)
    self.vals['cache_policy_short'] = cfg.coll_cache_policy.short()
    self.vals['cache_percentage'] = 100 * cfg.cache_percent
    self.vals['short_app'] = cfg.model.name
    self.vals['policy_impl'] = self.get_val('coll_cache_concurrent_link') + self.get_val('cache_policy_short')
  
  def get_optimal(self):
    optimal_cfg = copy.deepcopy(self.cfg)
    optimal_cfg.cache_percent = 0
    optimal_cfg.report_optimal = 1
    optimal_cfg.multi_gpu = False
    optimal_cfg.cache_policy = CachePolicy.cache_by_degree
    optimal_file_list = glob.glob(optimal_cfg.get_log_fname() + '_optimal_cache_hit.txt')
    if len(optimal_file_list) == 0:
      return math.nan
    elif len(optimal_file_list) > 1:
      raise Exception("more than one optimal file to choose")
    with open(optimal_file_list[0]) as f:
      for line in f.readlines():
        percent = int(line.split("\t")[0])
        hit_rate = float(line.split("\t")[1])
        if abs(percent - self.vals['cache_percentage']) < 1e-4:
          return hit_rate*100
    raise Exception("can not find optimal cache hit rate")

  def get_refresh_end(self, cfg):
    file = open(cfg.get_log_fname() + '.log')
    content = file.readlines()
    for lno, line in enumerate(content):
      if line.endswith('refresh done\n'): 
        m = re.match(r'^\[GPU0\] ([0-9]+) time ([0-9e\-\.]+) ([0-9\.]+)', content[lno - 1])
        if m: self.vals['refresh_stop'] = m.group(1)
        break

  def prepare_init(self, cfg):
    self.vals['init:presample'] = math.nan
    self.vals['init:load_dataset:mmap'] = math.nan
    self.vals['init:load_dataset:copy'] = math.nan
    self.vals['init:dist_queue'] = math.nan
    self.vals['init:internal'] = math.nan
    self.vals['init:cache'] = math.nan
    self.vals['init:other'] = math.nan
    self.vals['init:copy'] = math.nan
    fname = cfg.get_log_fname() + '.log'
    init_str_list = grep_from(fname, "^test_result:init:.*", [0, 0])
    for line in init_str_list:
      m2 = re.match(r'test_result:(.+)=(.+)\n', line)
      if m2:
        key = m2.group(1)
        value = m2.group(2)
        self.vals[key] = float(value)
    if not math.isnan(self.vals['init:presample']):
      self.vals['init:load_dataset:copy'] = self.vals['init:load_dataset:copy:sampler'] + self.vals['init:load_dataset:copy:trainer']
      self.vals['init:dist_queue']        = self.vals['init:dist_queue:alloc+push']     + self.vals['init:dist_queue:pin:sampler']   + self.vals['init:dist_queue:pin:trainer']
      self.vals['init:internal']          = self.vals['init:internal:sampler']          + self.vals['init:internal:trainer']
      self.vals['init:cache']             = self.vals['init:cache:sampler']             + self.vals['init:cache:trainer']
      self.vals['init:other']             = self.vals['init:dist_queue']                + self.vals['init:internal']
      self.vals['init:copy'] = self.vals['init:load_dataset:copy'] + self.vals['init:cache']
  
  def prepare_coll_cache(self, cfg):
    fname = cfg.get_log_fname() + '.log'
    coll_rst_list = grep_from(fname, "^coll_cache:.*", [0, 0])
    for line in coll_rst_list:
      m2 = re.match(r'coll_cache:(.+)=(.+)\n', line)
      if m2:
        key = m2.group(1)
        value = m2.group(2)
        self.vals[key] = float(value.split(',')[0])

    solve_time_list = grep_from(fname, "^Explored [0-9]+ nodes.*", [0,0])
    for line in solve_time_list:
      m = re.match(r'Explored [0-9]+ nodes \([0-9]+ simplex iterations\) in ([0-9\.]+) seconds.*\n', line)
      if m:
        value = m.group(1)
        self.vals["coll_cache:solve_time"] = float(value)

    fname = cfg.get_log_fname() + '.err.log'
    coll_rst_list = grep_from(fname, r".*remote ([0-9]+) / ([0-9]+) nodes.*", [0, 0])
    if len(coll_rst_list) == 0:
      self.vals["coll_cache:local_cache_rate"] = cfg.cache_percent
      self.vals["coll_cache:global_cache_rate"] = cfg.cache_percent
      self.vals["coll_cache:remote_cache_rate"] = 0
    else:
      m = re.match(r".*remote ([0-9]+) / ([0-9]+) nodes.*", coll_rst_list[0])
      self.vals["coll_cache:remote_cache_rate"] = float(m.group(1)) / float(m.group(2))
      coll_rst_list = grep_from(fname, r".*local ([0-9]+) / ([0-9]+) nodes.*", [0, 0])
      m = re.match(r".*local ([0-9]+) / ([0-9]+) nodes.*", coll_rst_list[0])
      self.vals["coll_cache:local_cache_rate"] = float(m.group(1)) / float(m.group(2))
      self.vals["coll_cache:global_cache_rate"] = self.vals["coll_cache:remote_cache_rate"] + self.vals["coll_cache:local_cache_rate"]
  def prepare_coll_cache_meta(self, cfg):
    try:
      self.vals['Step(average) L1 train total'] = self.get_val('Step(average) L1 convert time') + self.get_val('Step(average) L1 train')
      # when cache rate = 0, extract time has different log name...
      self.vals['Step(average) L2 feat copy'] = max_nan(self.get_val('Step(average) L2 cache feat copy'), self.get_val('Step(average) L2 extract'))

      # per-step feature nbytes (Remote, Cpu, Local)
      self.vals['Size.A'] = self.get_val('Step(average) L1 feature nbytes')
      self.vals['Size.R'] = handle_nan(self.get_val('Step(average) L1 remote nbytes'), 0)
      self.vals['Size.C'] = handle_nan(self.get_val('Step(average) L1 miss nbytes'), self.vals['Size.A'])
      self.vals['Size.L'] = self.get_val('Size.A') - self.get_val('Size.C') - self.get_val('Size.R')

      self.vals['SizeGB.R'] = self.get_val('Size.R') / 1024 / 1024 / 1024
      self.vals['SizeGB.C'] = self.get_val('Size.C') / 1024 / 1024 / 1024
      self.vals['SizeGB.L'] = self.get_val('Size.L') / 1024 / 1024 / 1024

      # per-step extraction time
      self.vals['Time.R'] = handle_nan(self.get_val('Step(average) L3 cache combine remote'))
      self.vals['Time.C'] = handle_nan(self.get_val('Step(average) L3 cache combine_miss'), self.get_val('Step(average) L2 extract'))
      self.vals['Time.L'] = handle_nan(self.get_val('Step(average) L3 cache combine cache'))

      # per-step extraction throughput (GB/s)
      self.vals['Thpt.R'] = div_nan(self.get_val('Size.R'), self.get_val('Time.R')) / 1024 / 1024 / 1024
      self.vals['Thpt.C'] = div_nan(self.get_val('Size.C'), self.get_val('Time.C')) / 1024 / 1024 / 1024
      self.vals['Thpt.L'] = div_nan(self.get_val('Size.L'), self.get_val('Time.L')) / 1024 / 1024 / 1024

      # per-step extraction portion from different source
      self.vals['Wght.R'] = div_nan(self.get_val('Size.R'), self.get_val('Size.A')) * 100
      self.vals['Wght.C'] = div_nan(self.get_val('Size.C'), self.get_val('Size.A')) * 100
      self.vals['Wght.L'] = 100 - self.get_val('Wght.R') - self.get_val('Wght.C')
    except Exception as e:
      print("Error when " + self.cfg.get_log_fname() + '.log')
      traceback.print_exc()

  def prepare_epoch_eval(self, cfg):
    self.vals['epoch_time'] = math.nan
    fname = cfg.get_log_fname() + '.log'
    epoch_rst_list = exclude_from(grep_from(fname, "^test_result:.*", [0, 0]), "^test_result:init:.*")
    self.vals['pipeline_train_epoch_time'] = math.nan
    self.vals['pipeline_train_epoch_time'] = math.nan
    self.vals['epoch_time:sample_total']   = math.nan
    self.vals['epoch_time:copy_time']      = math.nan
    self.vals['epoch_time:train_total']    = math.nan
    for line in epoch_rst_list:
      m2 = re.match(r'test_result:(.+)=(.+)\n', line)
      if m2:
        key = m2.group(1)
        value = m2.group(2)
        if key != 'cache_percentage':
          self.vals[key] = float(value)
    if 'cache_hit_rate' not in self.vals:
      self.vals['hit_percent'] = math.nan
    if 'hit_percent' not in self.vals:
      self.vals['hit_percent'] = float(self.vals['cache_hit_rate'])*100
    # if cfg.pipeline:
    #   self.vals['epoch_time'] = self.vals['pipeline_train_epoch_time']
    #   self.vals['train_process_time'] = self.vals['pipeline_train_epoch_time']
    # else:
    if math.isnan(self.vals['epoch_time:sample_total']) and 'epoch_time:sample_time' in self.vals:
      self.vals['epoch_time:sample_total'] = self.vals['epoch_time:sample_time']
    self.vals['epoch_time'] = self.vals['epoch_time:sample_total'] + self.vals['epoch_time:copy_time'] + self.vals['epoch_time:train_total']
    self.vals['train_process_time'] = self.vals['epoch_time:copy_time'] + self.vals['epoch_time:train_total']

    self.vals['epoch_time'] = '{:.4f}'.format(self.vals['epoch_time'])
    self.vals['train_process_time'] = '{:.4f}'.format(self.vals['train_process_time'])

  @staticmethod
  def prepare_profiler_log_merge_groups(result_map_list, cfg):
    rst = {}
    for result_map in result_map_list:
      for key,val in result_map.items():
        if key in rst:
          val = max(rst[key], val)
        rst[key] = val
    # print(rst)
    return rst
  @staticmethod
  def prepare_profiler_log_one_group(line_list):
    result_map = {}
    if len(line_list) == 0:
      return None
    assert(line_list[0].startswith('    ['))
    global_prefix = re.match(r'    \[([^ ]+) .*', line_list[0]).group(1)
    for i in range(1, len(line_list)):
      line = line_list[i].strip()
      if line.find(':') != -1:
        prefix = line[:line.find(':')]
        line = line[len(prefix)+1:]
      else:
        prefix = line[:2]
        line = line[len(prefix):]
      item_list = line.split('|')
      item_list = [global_prefix + ' ' + prefix + ' ' + item.strip() for item in item_list]
      for item in item_list:
        m=re.match(r'([^\.]*) +([0-9\.]*)( (MB|Bytes|KB|GB))?', item)
        # print(item, m)
        key,val = m.group(1).strip(),float(m.group(2))
        if m.group(3):
          val *= size_unit_to_coefficient[m.group(4)]
        assert(key not in result_map)
        result_map[key] = val
    return result_map
  
  def prepare_percentage_log(self, cfg):
    # percentage_pattern = r'^p([0-9\.]+)_(.*)'
    percentage_pattern = r'^        p([0-9\.]+)_([^=]*)=([0-9\.]+)'
    line_list = grep_from(cfg.get_log_fname() + '.log', percentage_pattern)
    for i in range(0, len(line_list)):
      line = line_list[i]
      m = re.match(percentage_pattern, line)
      key = m.group(2) + '_' + m.group(1)
      val = m.group(3)
      self.vals[key] = val

  def prepare_profiler_log(self, cfg):
    line_list = grep_from(cfg.get_log_fname() + '.log', r'^(    \[Step|        L|    \[Init).*')
    line_list.append('    [END]')
    result_map_list = []
    cur_begin = -1
    for i in range(0, len(line_list)):
      line = line_list[i]
      if line.startswith('    ['):
        if cur_begin != -1:
          result_map_list.append(self.prepare_profiler_log_one_group(line_list[cur_begin:i]))
        cur_begin = i
    result_map = self.prepare_profiler_log_merge_groups(result_map_list, cfg)
    self.vals.update(result_map)

    num_step_pattern = r'^    \[Step.* E[0-9]+ S([0-9]+)\]\n'
    line_list = grep_from(cfg.get_log_fname() + '.log', num_step_pattern)
    num_step = 0
    for line in line_list:
      num_step = max(num_step, int(re.match(num_step_pattern, line).group(1)))
    self.vals['num_step'] = num_step + 1
    # print(result_map_list)

  def prepare_cuda_eval(self, cfg):
    cuda_eval_pattern = r'^\[CUDA\] worker0 cuda mem usage: ([0-9\.]+) GB'
    line_list = grep_from(cfg.get_log_fname() + '.log', cuda_eval_pattern)
    for i in range(0, len(line_list)):
      line = line_list[i]
      m = re.match(cuda_eval_pattern, line)
      val = m.group(1)
      self.vals['cuda_usage'] = val

  def prepare_python_eval(self, cfg):
    # 35700 time 1.06001e-05 0.00170291
    python_eval_pattern = r'^\[GPU0\] ([0-9]+) time ([0-9e\-\.]+) ([0-9\.]+)'
    line_list = grep_from(cfg.get_log_fname() + '.log', python_eval_pattern)
    for i in range(0, len(line_list)):
      line = line_list[i]
      m = re.match(python_eval_pattern, line)
      key = m.group(1) + '_iter'
      val = float(m.group(3))
      self.vals[key] = val

  def to_formated_str(self):
    pass
    # self.vals['dataset_short'] = self.cfg.dataset.short()
  def get_val(self, key):
    if key in self.vals:
      return self.vals[key]
    else:
      return math.nan

  @staticmethod
  def print_dat(inst_list: list, outf, meta_list = default_meta_list, custom_col_title_list=None, sep='\t'):
    if custom_col_title_list is None:
      custom_col_title_list = meta_list
    print(sep.join(custom_col_title_list), file=outf)
    for inst in inst_list:
      try:
        inst.to_formated_str()
        # '{:.2f}'.format(inst.vals[meta]) if isinstance(inst.vals[meta], float) else str(inst.vals[meta])
        def to_print_val(_dict, _key):
          if _key not in _dict:
            return "None"
          if isinstance(_dict[_key], float):
            return '{:.6f}'.format(_dict[_key])
          return str(_dict[_key])
        print(sep.join([to_print_val(inst.vals, meta) for meta in meta_list]), file=outf)
      except KeyError:
        print("error when ", inst.fname)
        # print(sys.exc_info())
        traceback.print_exc()
        print(sep.join([str(inst.vals[meta]) if meta in inst.vals else "None" for meta in meta_list]), file=outf)
        # sys.exit(1)
    pass

def assign_sequence_number(cfg_list : list, start=0):
  for i in range(len(cfg_list)):
    cfg_list[i].seq_num = i + start
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

import os
import datetime
from enum import Enum
import copy
import json

def percent_gen(lb, ub, gap=1):
  ret = []
  i = lb
  while i <= ub:
    ret.append(i/100)
    i += gap
  return ret

def reverse_percent_gen(lb, ub, gap=1):
  ret = percent_gen(lb, ub, gap)
  return list(reversed(ret))

datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_DIR='run-logs/logs_samgraph_' + datetime_str
CONFIG_DIR='run-configs/config_hps_' + datetime_str

class System(Enum):
  hps = 0
  collcache = 1

  def __str__(self):
    if self is System.hps:
      return "HPS"
    else:
      return "COLL"

class Model(Enum):
  dlrm = 0
  dcn  = 1

class Dataset(Enum):
  def __new__(cls, *args, **kwds):
    value = len(cls.__members__) + 1
    obj = object.__new__(cls)
    obj._value_ = value
    return obj
  def __init__(self, path, log_name, datfile_name, vocabulary, slot_num):
    self.path = path if path else self.name
    self.log_name = log_name if log_name else self.name
    self.datfile_name = datfile_name if datfile_name else self.log_name
    self.vocabulary = vocabulary
    self.slot_num = slot_num
  def __str__(self):
    return self.path
  criteo_like_uniform       = None,                      "CRU",         "CRU",        187767399, 26
  criteo_like_uniform_small = None,                      "CRU_S",       "CRU_S",      187767399, 26
  dlrm_datasets             = None,                      "DLRM",        "DLRM",       None,      None
  simple_power02            = "simple_power0.2",         "SP_02",       "SP_02",      100000000, 25
  simple_power02_slot100    = "simple_power0.2_slot100", "SP_02_S100",  "SP_02_S100", 100000000, 100
  simple_power1             = None,                      "SP_1",        "SP_1",       100000000, 25
  simple_power1_slot100     = None,                      "SP_1_S100",   "SP_1_S100",  100000000, 100
  simple_uniform            = None,                      None,          None,         100000000, 25
  criteo_tb                 = "criteo_tb",               "CR",          "CR",         882774592, 26
  syn                       = "syn_a12_s100_c800m",      "SYN",         "SYN-A",      800000000, 100
  syn_14                    = "syn_a14_s100_c800m",      "SYN_14",      "SYN-B",      800000000, 100
  syn_14_50                 = "syn_a14_s50_c800m",       "SYN_14_50",   "SYN_14_50",  800000000, 50
  criteo_kaggle             = "criteo_kaggle",           "CK",          "CK",         33762604, 26

class RandomDataset:
  def __init__(self, path, short_name, vocabulary, slot_num):
    self.path = path
    self.short_name = short_name
    self.vocabulary = vocabulary
    self.slot_num = slot_num
    self.log_name = short_name
    self.datfile_name = short_name
  def short(self):
    return self.short_name
  def __str__(self):
    return self.path

class CachePolicy(Enum):
  cache_by_degree=0
  cache_by_heuristic=1
  cache_by_pre_sample=2
  cache_by_degree_hop=3
  cache_by_presample_static=4
  cache_by_fake_optimal=5
  dynamic_cache=6
  cache_by_random=7
  coll_cache=8
  coll_cache_intuitive=9
  partition_cache=10
  part_rep_cache=11
  rep_cache=12
  coll_cache_asymm_link=13
  clique_part=14
  clique_part_by_degree=15
  sok = 16
  hps = 17
  coll_fine_grain = 18

  def __str__(self):
    name_list = [
      'degree',
      'heuristic',
      'pre_sample',
      'degree_hop',
      'presample_static',
      'fake_optimal',
      'dynamic_cache',
      'random',
      'coll_cache',
      'coll_intuitive',
      'partition',
      'part_rep',
      'rep',
      'coll_asymm',
      'cliq_part',
      'cliq_part_degree',
      'sok',
      'hps',
      'coll_fine_grain',
    ]
    return name_list[self.value]
  
  def short(self):
    policy_str_short = [
      "Deg",
      "Heuristic",
      "PreS",
      "DegH",
      "PreSS",
      "FakeOpt",
      "DynCache",
      "Rand",
      "Coll",
      "CollIntui",
      "Part",
      "PartRep",
      "Rep",
      "CollAsymm",
      "CliqPart",
      "CliqPartDeg",
      "SOK",
      "HPS",
      "CollFine"
    ]
    return policy_str_short[self.value]

class RunConfig:
  def __init__(self, system:System, model:Model, dataset:Dataset, 
               gpu_num: int=8,
               global_batch_size: int=65536,
               coll_cache_policy:CachePolicy=CachePolicy.coll_cache_asymm_link, 
               cache_percent:float=0.1, 
               logdir:str=LOG_DIR,
               confdir:str=CONFIG_DIR):
    # arguments
    self.system         = system
    self.model          = model
    self.dataset        = dataset
    self.logdir         = logdir
    self.confdir        = confdir
    self.gpu_num        = gpu_num
    self.epoch          = 5
    # self.iter_num       = 6000
    self.slot_num       = None
    self.dense_dim      = 13
    self.embed_vec_size = 128
    self.combiner       = "mean"
    self.optimizer      = "plugin_adam"
    self.global_batch_size      = global_batch_size
    self.dataset_root_path      = "/datasets_dlr/processed/"
    self.model_root_path        = "/nvme/songxiaoniu/hps-model/dlrm_criteo/"
    # hps json
    self.cache_percent          = cache_percent
    self.coll_cache_policy      = coll_cache_policy
    self.mock_embedding         = False    # if true, mock embedding table by emb_vec_sz and max_voc_sz
    self.plain_dense_model      = False
    self.random_request         = False
    self.alpha                  = 0.2
    self.max_vocabulary_size    = None
    self.coll_cache_enable_iter = 1000
    self.coll_cache_refresh_iter = 1000
    self.coll_cache_refresh_seq_bucket_sz = 0
    self.coll_cache_enable_refresh = False
    self.iteration_per_epoch    = 1000
    # env variables
    self.coll_cache_no_group    = ""
    self.coll_cache_concurrent_link   = ""
    self.log_level              = "warn"
    self.profile_level          = 3
    self.custom_env             = ""
    self.empty_feat             = 25
    self.scalability_test       = False
    self.hps_cache_statistic    = False
    self.coll_cache_scale       = 0
    self.sok_use_hashtable      = False
    self.coll_hash_impl = ""
    self.coll_skip_hash = ""

    self.skip_model = None
    self.nsys_prof_metric = False
    self.coll_intuitive_min_freq = ""

  def get_mock_sparse_name(self):
    if self.mock_embedding:
      return '_'.join(['mock', f'{self.max_vocabulary_size}', f'{self.embed_vec_size}'])
    else:
      return 'nomock'

  def get_output_fname_base(self):
    std_out_fname = '_'.join(
      [str(self.system), self.model.name, self.dataset.log_name] + 
      [f'policy_{self.coll_cache_policy.short()}', f'cache_rate_{self.cache_percent}'] +
      [f'batch_size_{self.global_batch_size}'])
    if self.mock_embedding:
      std_out_fname += '_' + self.get_mock_sparse_name()
    if self.coll_cache_no_group != "":
      std_out_fname += '_nogroup_' + self.coll_cache_no_group
    if self.coll_cache_concurrent_link != "":
      std_out_fname += '_concurrent_impl_' + self.coll_cache_concurrent_link
    if self.scalability_test == True:
      std_out_fname += f'_gpu_num_{self.gpu_num}'
    if self.coll_cache_scale != 0:
      std_out_fname += f'_scale_nb_{self.coll_cache_scale}'
    if self.coll_cache_refresh_seq_bucket_sz != 0:
      std_out_fname += f'_bucket_{self.coll_cache_refresh_seq_bucket_sz}'
    if self.coll_cache_enable_refresh:
      std_out_fname += f'_refresh_{self.coll_cache_refresh_iter}'
    if self.coll_hash_impl != "":
      std_out_fname += f'_hash_impl_{self.coll_hash_impl}'
    if self.coll_skip_hash != "":
      std_out_fname += f'_skip_hash_{self.coll_skip_hash}'
    if self.coll_intuitive_min_freq != "" and self.coll_cache_policy == CachePolicy.coll_cache_intuitive:
      std_out_fname += f'_min_freq_{self.coll_intuitive_min_freq}'
    return std_out_fname

  def get_conf_fname(self):
    std_out_conf = f'{self.confdir}/'
    std_out_conf += self.get_output_fname_base()
    std_out_conf += '.json'
    return std_out_conf

  def get_log_fname(self):
    std_out_log = f'{self.logdir}/'
    std_out_log += self.get_output_fname_base()
    return std_out_log

  def beauty(self):
    msg = ' '.join(
      ['Running', str(self.system), self.model.name, str(self.dataset)] +
      [str(self.coll_cache_policy), f'cache_rate {self.cache_percent}', f'batch_size {self.global_batch_size}'])
    if self.mock_embedding:
      msg += f' mock({self.max_vocabulary_size} vocabs, {self.embed_vec_size} emb_vec_sz)'
    if self.coll_cache_no_group != "":
      msg += f' nogroup={self.coll_cache_no_group}'
    if self.coll_cache_concurrent_link != "":
      msg += f' concurrent_link={self.coll_cache_concurrent_link}'
    if self.scalability_test == True:
      msg += f' gpu_num={self.gpu_num}'
    if self.coll_cache_scale != "":
      msg += f' scale_nb={self.coll_cache_scale}'
    if self.coll_cache_refresh_seq_bucket_sz != 0:
      msg += f' seq_bucket={self.coll_cache_refresh_seq_bucket_sz}'
    if self.coll_cache_enable_refresh:
      msg += f' refresh'
    if self.coll_hash_impl != "":
      msg += f' hash_impl={self.coll_hash_impl}'
    if self.coll_skip_hash != "":
      msg += f' skip_hash={self.coll_skip_hash}'
    return msg + '.'

  def form_cmd(self, durable_log=True):
    assert((self.epoch * self.iteration_per_epoch + self.coll_cache_enable_iter) == self.iter_num)
    cmd_line = f'{self.custom_env} '
    cmd_line += f'HUGECTR_LOG_LEVEL=0 '
    cmd_line += f'TF_CPP_MIN_LOG_LEVEL=2 '
    cmd_line += f'COLL_NUM_REPLICA={self.gpu_num} '
    if self.coll_cache_scale != 0:
      cmd_line += f'COLL_CACHE_SCALE={self.coll_cache_scale} '
    if self.coll_cache_no_group != "":
      cmd_line += f'SAMGRAPH_COLL_CACHE_NO_GROUP={self.coll_cache_no_group} '
    if self.coll_cache_concurrent_link != "":
      cmd_line += f' SAMGRAPH_COLL_CACHE_CONCURRENT_LINK_IMPL={self.coll_cache_concurrent_link} '
    cmd_line += f'SAMGRAPH_LOG_LEVEL={self.log_level} '
    cmd_line += f'SAMGRAPH_PROFILE_LEVEL={self.profile_level} '
    if self.coll_cache_policy == CachePolicy.sok:
      cmd_line += f'ITERATION_PER_EPOCH={self.iteration_per_epoch} '
      cmd_line += f'EPOCH={self.epoch} '
    if self.coll_cache_refresh_seq_bucket_sz != 0:
      cmd_line += f'PROFILE_SEQ_BUCKET_SZ={self.coll_cache_refresh_seq_bucket_sz} '

    if self.coll_hash_impl != "":
      cmd_line += f' COLL_HASH_IMPL={self.coll_hash_impl} '
    if self.coll_skip_hash != "":
      cmd_line += f' COLL_SKIP_HASH={self.coll_skip_hash} '

    if self.coll_intuitive_min_freq != "":
      cmd_line += f'COLL_INTUITIVE_MIN_FREQ={self.coll_intuitive_min_freq} '
    if self.nsys_prof_metric:
      cmd_line += f' nsys profile --trace-fork-before-exec=true -t cuda --gpu-metrics-device=all --gpu-metrics-set=ga100 -o {self.get_log_fname()}.nsys-rep --force-overwrite true '
    cmd_line += f'python ../common/inference.py'
    cmd_line += f' --gpu_num {self.gpu_num} '
    
    cmd_line += f' --iter_num {self.iter_num} '
    cmd_line += f' --slot_num {self.slot_num} '
    cmd_line += f' --dense_dim {self.dense_dim} '
    cmd_line += f' --embed_vec_size {self.embed_vec_size} '
    cmd_line += f' --global_batch_size {self.global_batch_size} '
    cmd_line += f' --combiner {self.combiner} '
    cmd_line += f' --optimizer {self.optimizer} '
    cmd_line += f' --model {self.model.name} '
    if self.plain_dense_model:
      cmd_line += f' --dense_model_path plain'
    else:
      cmd_line += f' --dense_model_path {self.model_root_path}dense.model'

    cmd_line += f' --max_vocabulary_size {self.max_vocabulary_size}'
    cmd_line += f' --cache_percent {self.cache_percent}'
    if self.random_request:
      cmd_line += f' --random_request '
      cmd_line += f' --alpha {self.alpha} '
    else:
      cmd_line += f' --dataset_path {self.dataset_root_path + str(self.dataset)}'
    if self.coll_cache_policy != CachePolicy.sok:
      cmd_line += f' --ps_config_file {self.get_conf_fname()}'
    else:
      if self.sok_use_hashtable:
        cmd_line += f' --sok_use_hashtable '

    cmd_line += f' --iteration_per_epoch {self.iteration_per_epoch}'
    cmd_line += f' --coll_cache_enable_iter {self.coll_cache_enable_iter}'
    cmd_line += f' --coll_cache_refresh_iter {self.coll_cache_refresh_iter}'
    if self.coll_cache_enable_refresh:
      cmd_line += f' --coll_cache_enable_refresh '
    
    cmd_line += f' --coll_cache_policy {str(self.coll_cache_policy)}'
    cmd_line += f' --empty-feat {self.empty_feat}'

    if self.skip_model:
      cmd_line += f' --skip_model '

    if durable_log:
      std_out_log = self.get_log_fname() + '.log'
      std_err_log = self.get_log_fname() + '.err.log'
      cmd_line += f' > \"{std_out_log}\"'
      cmd_line += f' 2> \"{std_err_log}\"'
      cmd_line += ';'
      cmd_line = "COLL_LOG_BASE=\"" + self.get_log_fname() + "\" " + cmd_line;
    return cmd_line

  # some members are lazy initialized
  def handle_mock_params(self):
    # if self.system == System.hps: self.coll_cache_enable_iter = 0
    if self.coll_cache_policy in [CachePolicy.hps, CachePolicy.sok]: self.coll_cache_enable_iter = 0
    self.iter_num = self.epoch * self.iteration_per_epoch + self.coll_cache_enable_iter
    self.max_vocabulary_size = self.dataset.vocabulary
    self.slot_num = self.dataset.slot_num

  def generate_ps_config(self, succeed=False):
    # self.handle_mock_params()
    assert((self.global_batch_size % self.gpu_num) == 0)
    if self.coll_cache_policy == CachePolicy.sok: return
    conf = {
      "supportlonglong": True,
      "models": [{
          "num_of_worker_buffer_in_pool": 1,
          "num_of_refresher_buffer_in_pool": 0,
          "embedding_table_names":["sparse_embedding0"],
          "default_value_for_each_table": [1.0],
          "i64_input_key": False,
          "cache_refresh_percentage_per_iteration": 0,
          "hit_rate_threshold": 1.1,
          "gpucache": True,
          }
      ],
      "volatile_db": {
          "type": "direct_map",
          "num_partitions": 56
      },
      "use_multi_worker": True,
    }
    conf['models'][0]['model'] = self.model.name
    conf['models'][0]['sparse_files'] = [self.model_root_path + 'sparse_cont.model']
    if self.mock_embedding: conf['models'][0]['sparse_files'] = [self.get_mock_sparse_name()]
    conf['models'][0]['embedding_vecsize_per_table'] = [self.embed_vec_size]
    conf['models'][0]['maxnum_catfeature_query_per_table_per_sample'] = [self.slot_num]
    conf['models'][0]['deployed_device_list'] = list(range(self.gpu_num))
    conf['models'][0]['max_batch_size'] = self.global_batch_size // self.gpu_num
    conf['models'][0]['gpucacheper'] = self.cache_percent
    if self.cache_percent == 0:
      conf['models'][0]['gpucache'] = False

    conf['models'][0]['max_vocabulary_size'] = [self.max_vocabulary_size]
    if self.coll_cache_policy == CachePolicy.hps: conf['use_coll_cache'] = False
    else: conf['use_coll_cache'] = True
    conf['coll_cache_enable_iter'] = self.coll_cache_enable_iter
    conf['coll_cache_refresh_iter'] = self.coll_cache_refresh_iter
    conf['coll_cache_enable_refresh'] = self.coll_cache_enable_refresh
    conf['iteration_per_epoch'] = self.iteration_per_epoch
    conf['epoch'] = self.epoch
    conf['coll_cache_policy'] = self.coll_cache_policy.value
    conf['hps_cache_statistic'] = self.hps_cache_statistic
    conf['succeed'] = succeed

    result = json.dumps(conf, indent=4)
    with open(self.get_conf_fname(), "w") as outfile:
      outfile.write(result)

  def run(self, mock=False, durable_log=True, callback = None, retry=False, fail_only=False):
    '''
    retry: immediately retry on each failed job
    fail_only: only run previously failed job. fail status is recorded in json file
    '''
    self.handle_mock_params()
    os.system('mkdir -p {}'.format(self.confdir))
    previous_succeed = False
    try:
      with open(self.get_conf_fname(), "r") as conf:
        js = json.load(conf)
      if 'succeed' in js and js['succeed']:
        previous_succeed = True
    except Exception as e:
      pass
    if fail_only:
      if previous_succeed:
        if callback != None:
          callback(self)
        return 0

    self.generate_ps_config(previous_succeed)

    if mock:
      print(self.form_cmd(durable_log))
    else:
      print(self.beauty())

      if durable_log:
        os.system('mkdir -p {}'.format(self.logdir))
      while True:
        # os.system('echo quit | nvidia-cuda-mps-control 2> /dev/null; sleep 0.5')
        # os.system('nvidia-cuda-mps-control -d; sleep 0.5')
        status = os.system(self.form_cmd(durable_log))
        # os.system('echo quit | nvidia-cuda-mps-control 2> /dev/null')
        if os.WEXITSTATUS(status) != 0:
          if retry:
            print("FAILED and Retry!")
            continue
          print("FAILED!")
          self.generate_ps_config(False)
        else:
          self.generate_ps_config(True)
        if callback != None:
          callback(self)
        break
    return 0

def run_in_list(conf_list : list, mock=False, durable_log=True, callback = None):
  for conf in conf_list:
    conf : RunConfig
    conf.run(mock, durable_log, callback)

class ConfigList:
  def __init__(self):
    self.conf_list = [
      RunConfig(System.collcache, Model.dlrm, Dataset.criteo_like_uniform)]

  def select(self, key, val_indicator):
    '''
    filter config list by key and list of value
    available key: model, dataset, cache_policy, pipeline
    '''
    newlist = []
    for cfg in self.conf_list:
      if getattr(cfg, key) in val_indicator:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

  def override(self, key, val_list):
    '''
    override config list by key and value.
    if len(val_list)>1, then config list is extended, example:
       [cfg1(batch_size=4000)].override('batch_size',[1000,8000]) 
    => [cfg1(batch_size=1000),cfg1(batch_size=8000)]
    available key: arch, logdir, cache_percent, cache_policy, batch_size
    '''
    if len(val_list) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for val in val_list:
      new_list = copy.deepcopy(orig_list)
      for cfg in new_list:
        setattr(cfg, key, val)
      self.conf_list += new_list
    return self

  def override_T(self, key, val_list):
    if len(val_list) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for cfg in orig_list:
      for val in val_list:
        cfg = copy.deepcopy(cfg)
        setattr(cfg, key, val)
        self.conf_list.append(cfg)
    return self

  def part_override(self, filter_key, filter_val_list, override_key, override_val_list):
    newlist = []
    for cfg in self.conf_list:
      # print(cfg.cache_impl, cfg.logdir, filter_key, filter_val_list)
      if getattr(cfg, filter_key) in filter_val_list:
        # print(cfg.cache_impl, cfg.logdir)
        for val in override_val_list:
          # print(cfg.cache_impl, cfg.logdir)
          cfg = copy.deepcopy(cfg)
          setattr(cfg, override_key, val)
          newlist.append(cfg)
      else:
        newlist.append(cfg)
    self.conf_list = newlist
    return self

  def hyper_override(self, key_array, val_matrix):
    if len(key_array) == 0 or len(val_matrix) == 0:
      return self
    orig_list = self.conf_list
    self.conf_list = []
    for cfg in orig_list:
      for val_list in val_matrix:
        cfg = copy.deepcopy(cfg)
        for idx in range(len(key_array)):
          setattr(cfg, key_array[idx], val_list[idx])
        self.conf_list.append(cfg)
    return self

  def concat(self, another_list):
    self.conf_list += copy.deepcopy(another_list.conf_list)
    return self
  def copy(self):
    return copy.deepcopy(self)
  @staticmethod
  def Empty():
    ret = ConfigList()
    ret.conf_list = []
    return ret
  @staticmethod
  def MakeList(conf):
    ret = ConfigList()
    if isinstance(conf, list):
      ret.conf_list = conf
    elif isinstance(conf, RunConfig):
      ret.conf_list = [conf]
    else:
      raise Exception("Please construct fron runconfig or list of it")
    return ret

  def run(self, mock=False, durable_log=True, callback = None, retry=False, fail_only=False):
    for conf in self.conf_list:
      conf : RunConfig
      conf.run(mock, durable_log, callback, retry, fail_only)

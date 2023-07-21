import time
import os
from enum import Enum
import numpy as np
import tensorflow as tf

__all__ = ['get_dataset_list', 'get_default_common_config', 'add_common_arguments', 'process_common_config', 'print_run_config']

def get_dataset_list():
    return ['criteo_like_uniform', 'criteo_like_uniform_small',
            'dlrm_datasets', 'simple_power02', 'simple_power02_slot100',
            'simple_power1', 'simple_power1_slot100', 'simple_uniform',
            'criteo_tb']

def get_default_common_config(**kwargs):
    default_common_config = {}

    default_common_config["gpu_num"] = 8                               # the number of available GPUs
    default_common_config["iter_num"] = 6000                           # the number of training iteration
    default_common_config["slot_num"] = 26                             # the number of feature fields in this embedding layer
    default_common_config["embed_vec_size"] = 128                      # the dimension of embedding vectors
    default_common_config["dense_dim"] = 13                            # the dimension of dense features
    default_common_config["global_batch_size"] = 65536                 # the globally batchsize for all GPUs
    default_common_config["iteration_per_epoch"] = 1000
    default_common_config["coll_cache_enable_iter"] = 1000
    default_common_config["coll_cache_refresh_iter"] = 2147483648
    default_common_config["coll_cache_enable_refresh"] = False
    default_common_config["coll_cache_policy"] = "coll_asymm"
    default_common_config["model"] = "DLRM"
    default_common_config["combiner"] = "mean"
    default_common_config["optimizer"] = "plugin_adam"
    default_common_config["np_key_type"] = np.int32
    default_common_config["np_vector_type"] = np.float32
    default_common_config["tf_key_type"] = tf.int32
    default_common_config["tf_vector_type"] = tf.float32
    default_common_config["ps_config_file"] = ""
    default_common_config["cache_percent"] = 0
    default_common_config["dense_model_path"] = "/nvme/songxiaoniu/hps-model/dlrm_criteo/dense.model"
    default_common_config["dataset_path"] = "/nvme/songxiaoniu/hps-dataset/criteo_like_uniform"

    default_common_config.update(kwargs)

    return default_common_config

def add_common_arguments(argparser, run_config):
    argparser.add_argument('--model', type=str,
                            default=run_config['model'])
    argparser.add_argument('--gpu_num', type=int,
                            default=run_config['gpu_num'])
    argparser.add_argument('--iter_num', type=int,
                            default=run_config['iter_num'])
    argparser.add_argument('--slot_num', type=int,
                            default=run_config['slot_num'])
    argparser.add_argument('--dense_dim', type=int,
                            default=run_config['dense_dim'])
    argparser.add_argument('--embed_vec_size', type=int,
                            default=run_config['embed_vec_size'])
    argparser.add_argument('--global_batch_size', type=int,
                            default=run_config['global_batch_size'])
    argparser.add_argument('--iteration_per_epoch', type=int,
                            default=run_config['iteration_per_epoch'])   
    argparser.add_argument('--cache_percent', type=float,
                            default=run_config['cache_percent'])                        
    argparser.add_argument('--coll_cache_enable_iter', type=int,
                            default=run_config['coll_cache_enable_iter'])                        
    argparser.add_argument('--coll_cache_refresh_iter', type=int,
                            default=run_config['coll_cache_refresh_iter'])
    argparser.add_argument('--coll_cache_enable_refresh', action='store_true',
                            default=run_config['coll_cache_enable_refresh'])
    argparser.add_argument('--coll_cache_policy', type=str,
                            default=run_config['coll_cache_policy'])
    argparser.add_argument('--combiner', type=str,
                            default=run_config['combiner'])
    argparser.add_argument('--optimizer', type=str,
                            default=run_config['optimizer'])
    argparser.add_argument('--dense_model_path', type=str,
                            default=run_config['dense_model_path'])
    argparser.add_argument('--dataset_path', type=str,
                            default=run_config['dataset_path'])
    argparser.add_argument('--ps_config_file', type=str,
                            default=run_config['ps_config_file'])
    argparser.add_argument('--empty-feat', type=str,
                            dest='_empty_feat', default='')

def process_common_config(run_config):
    run_config["dataset_path"] += '/saved_dataset'
    if run_config["coll_cache_policy"] == "sok":
        run_config["tf_key_type"] = tf.uint32
    # os.environ['SAMGRAPH_LOG_LEVEL'] = run_config['_log_level']
    # os.environ["SAMGRAPH_PROFILE_LEVEL"] = run_config['_profile_level']
    os.environ['SAMGRAPH_EMPTY_FEAT'] = run_config['_empty_feat']

def print_run_config(run_config):
    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        if not k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

    for k, v in run_config.items():
        if k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

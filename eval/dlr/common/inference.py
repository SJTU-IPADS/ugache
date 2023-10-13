import argparse
import time
import os
proxy = None
def handle_proxy(proxy_env):
    if proxy_env in os.environ:
        global proxy
        proxy = os.environ[proxy_env]
        del os.environ[proxy_env]
handle_proxy("http_proxy")
handle_proxy("https_proxy")
handle_proxy("all_proxy")
handle_proxy("HTTP_PROXY")
handle_proxy("HTTPS_PROXY")
handle_proxy("ALL_PROXY")
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import atexit
import multiprocessing
from common_config import *
import json

from model_zoo.hps import DLRM

from ds_generator import generate_random_samples as generate_random_samples

def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("RM INFERENCE")
    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--sok_use_hashtable', dest='sok_use_hashtable', action='store_true')
    argparser.add_argument('--random_request', dest='random_request', action='store_true')
    argparser.add_argument('--skip_model', dest='skip_model', action='store_true')
    argparser.add_argument('--alpha', type=float, default=None)
    argparser.add_argument('--max_vocabulary_size', type=int, default=100000000)

    return vars(argparser.parse_args())

def get_run_config():
    run_config = {}
    run_config.update(get_default_common_config())
    run_config.update(parse_args(run_config))
    process_common_config(run_config)
    if run_config["random_request"]:
        if run_config["alpha"] == None:
            raise Exception("when random request is used, alpha must be provided")
        slot_cardinality = args["max_vocabulary_size"] // args["slot_num"]
        # specify vocabulary range
        ranges = [[0, 0] for i in range(run_config["slot_num"])]
        max_range = 0
        for i in range(run_config["slot_num"]):
            ranges[i][0] = max_range
            ranges[i][1] = max_range + slot_cardinality
            max_range += slot_cardinality
        run_config["vocabulary_range_per_slot"] = ranges
        assert(max_range == run_config["max_vocabulary_size"])
    print_run_config(run_config)
    return run_config

def prepare_model(args):
    if args['skip_model']:
        from model_zoo.coll import EmbOnly
        model = EmbOnly(args["embed_vec_size"], args["slot_num"], args["dense_dim"])
        return model
    assert(args["dense_model_path"] == "plain")
    assert(args["model"] in ['dlrm', 'dcn'])
    kwargs = {}
    if args["coll_cache_policy"] == "sok":
        kwargs={
            'max_vocabulary_size_per_gpu' : int(args["max_vocabulary_size"] * args['cache_percent']) if args['sok_use_hashtable'] else args["max_vocabulary_size"] // args["gpu_num"],
            'use_hashtable' : args['sok_use_hashtable']
        }
    if args["model"] == "dcn":
        if args["coll_cache_policy"] == "sok":
            from model_zoo.sok import DCN as Model
        elif args["coll_cache_policy"] == "hps":
            from model_zoo.hps import DCN as Model
        else:
            from model_zoo.coll import DCN as Model
    if args["model"] == "dlrm":
        if args["coll_cache_policy"] == "sok":
            from model_zoo.sok import DLRM as Model
        elif args["coll_cache_policy"] == "hps":
            from model_zoo.hps import DLRM as Model
        else:
            from model_zoo.coll import DLRM as Model
    model = Model(args["embed_vec_size"], args["slot_num"], args["dense_dim"], **kwargs)
    return model

def worker_func(args, worker_id):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        if args["coll_cache_policy"] == "sok":
            sok.Init(global_batch_size = args["global_batch_size"])
        elif args["coll_cache_policy"] == "hps":
            hps.Init(global_batch_size = args["global_batch_size"],
                ps_config_file = args["ps_config_file"])
        else:
            collcache_tf2.Init(global_batch_size = args["global_batch_size"],
                ps_config_file = args["ps_config_file"])
        barrier.wait()
        model = prepare_model(args)
    barrier.wait()
    # from https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    atexit.register(strategy._extended._cross_device_ops._pool.close) # type: ignore
    atexit.register(strategy._extended._host_cross_device_ops._pool.close) #type: ignore
    time.sleep(5)
    barrier.wait()

    @tf.function
    def _infer_step(inputs,):
        logit = model(inputs, training=False)
        return logit
    @tf.function
    def _whole_infer_step(in_param):
        return strategy.run(_infer_step, args=in_param)
    @tf.function
    def _warmup_step(inputs,):
        return inputs,

    def _dataset_fn(num_replica, local_id):
        assert(args["global_batch_size"] % num_replica == 0)
        replica_batch_size = args["global_batch_size"] // num_replica
        if args["dataset_path"].endswith("criteo_tb"):
            print("Loading Criteo TB")
            from ds_generator import criteo_tb
            dataset = criteo_tb(args["dataset_path"], replica_batch_size, args["iter_num"], num_replica, args["tf_key_type"])
            dataset = dataset.shard(num_replica, local_id)
        elif os.path.split(args['dataset_path'])[-1].startswith('syn'):
            print("Loading SYN")
            from ds_generator import syn
            dataset = syn(args["dataset_path"], replica_batch_size, args["iter_num"], num_replica, args["tf_key_type"], args["slot_num"], args["dense_dim"])
            dataset = dataset.shard(num_replica, local_id)
        else:
            print("Generating random dataset")
            sparse_keys, dense_features, labels = generate_random_samples(replica_batch_size * args["iter_num"], args["vocabulary_range_per_slot"], args["dense_dim"], np.int32, args["alpha"])
            def sequential_batch_gen():
                for i in range(0, replica_batch_size * args["iter_num"], replica_batch_size):
                    sparse_keys, dense_features, labels
                    yield sparse_keys[i:i+replica_batch_size],dense_features[i:i+replica_batch_size],labels[i:i+replica_batch_size]
            print("creating tf dataset")
            dataset = tf.data.Dataset.from_generator(sequential_batch_gen, 
                output_signature=(
                    tf.TensorSpec(shape=(replica_batch_size, args["slot_num"]), dtype=args["tf_key_type"]), 
                    tf.TensorSpec(shape=(replica_batch_size, args["dense_dim"]), dtype=args["tf_vector_type"]),
                    tf.TensorSpec(shape=(replica_batch_size, 1), dtype=tf.int32)))
            print("creating tf dataset - done")
        dataset = dataset.cache()
        dataset = dataset.prefetch(1000)
        return dataset
    dataset = _dataset_fn(args["gpu_num"], worker_id)

    ret_list = []
    for sparse_keys, dense_features, labels in tqdm(dataset, "warmup run"):
        inputs = [sparse_keys, dense_features]
        ret = strategy.run(_warmup_step, args=(inputs,))
        ret_list.append(ret)
    barrier.wait()

    ds_time = 0
    md_time = 0
    global proxy
    if proxy:
      print("restoring proxy")
      os.environ["http_proxy"] = proxy
      os.environ["HTTPS_PROXY"] = proxy
      os.environ["HTTP_PROXY"] = proxy
    else:
      print("no proxy in env")
    for i in range(args["iter_num"]):
        t0 = tf.timestamp()
        t1 = tf.timestamp()
        if args['skip_model']:
            _whole_infer_step(ret_list[i])
        else:
            _ = _whole_infer_step(ret_list[i])[0].numpy()
        t2 = tf.timestamp()
        ds_time += t1 - t0
        md_time += t2 - t1
        # profile
        if i >= args["coll_cache_enable_iter"]:
            if args["coll_cache_policy"] == "sok":
                sok.SetStepProfileValue(profile_type=sok.kLogL1TrainTime, value=(t2 - t1))
            else:
                hps.SetStepProfileValue(profile_type=hps.kLogL1TrainTime, value=(t2 - t1))
        if (i + 1) % 100 == 0:
            print("[GPU{}] {} time {:.6} {:.6}".format(worker_id, i + 1, ds_time / 100, md_time / 100), flush=True)
            ds_time = 0
            md_time = 0
            barrier.wait()

def proc_func(id):
    print(f"worker {id} at process {os.getpid()}")
    with open(f"/tmp/infer_{id}.pid", 'w') as f:
        print(f"{os.getpid()}", file=f, flush=True)
    time.sleep(5)
    tf_config = {"task": {"type": "worker", "index": id}, "cluster": {"worker": []}}
    for i in range(args["gpu_num"]): tf_config['cluster']['worker'].append("localhost:" + str(12340+i))
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[id], 'GPU')
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    _ = worker_func(args, id)
    Shutdown()

print(os.environ, flush=True)
args = get_run_config()
proc_list = [None for _ in range(args["gpu_num"])]
barrier = multiprocessing.Barrier(args["gpu_num"])

if args["coll_cache_policy"] == "sok":
    import sparse_operation_kit as sok
    Shutdown = sok.Shutdown
    SetStepProfileValue = sok.SetStepProfileValue
    kLogL1TrainTime = sok.kLogL1TrainTime
    def init():
        sok.Init(global_batch_size = args["global_batch_size"])
elif args["coll_cache_policy"] == "hps":
    import hierarchical_parameter_server as hps
    Shutdown = hps.Report
    SetStepProfileValue = hps.SetStepProfileValue
    kLogL1TrainTime = hps.kLogL1TrainTime
    def init():
        hps.Init(global_batch_size = args["global_batch_size"],
            ps_config_file = args["ps_config_file"])
else:
    import collcache_tf2
    Shutdown = collcache_tf2.Report
    SetStepProfileValue = collcache_tf2.SetStepProfileValue
    kLogL1TrainTime = collcache_tf2.kLogL1TrainTime
    def init():
        collcache_tf2.Init(global_batch_size = args["global_batch_size"],
            ps_config_file = args["ps_config_file"])
for i in range(args["gpu_num"]):
    proc_list[i] = multiprocessing.Process(target=proc_func, args=(i,))
    proc_list[i].start()
# from collcache_tf2 import wait_one_child
# ret_code = wait_one_child()
# if ret_code != 0:
#     for i in range(args["gpu_num"]):
#         proc_list[i].kill()
for i in range(args["gpu_num"]):
    proc_list[i].join()
import sys
# sys.exit(ret_code)

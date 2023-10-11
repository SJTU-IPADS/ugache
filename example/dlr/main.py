import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from numba import njit
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import atexit
import json
import multiprocessing

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                arch,
                activation='relu',
                out_activation=None,
                **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        index = 0
        for units in arch[:-1]:
            self.layers.append(tf.keras.layers.Dense(units, activation=activation, name="{}_{}".format(kwargs['name'], index)))
            index+=1
        self.layers.append(tf.keras.layers.Dense(arch[-1], activation=out_activation, name="{}_{}".format(kwargs['name'], index)))

            
    def call(self, inputs, training=False):
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x

class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, self_interaction=False):
        super(SecondOrderFeatureInteraction, self).__init__()
        self.self_interaction = self_interaction

    def call(self, inputs, training = False):
        batch_size = tf.shape(inputs)[0]
        num_feas = tf.shape(inputs)[1]

        dot_products = tf.matmul(inputs, inputs, transpose_b=True)

        ones = tf.ones_like(dot_products, dtype=tf.float32)
        mask = tf.linalg.band_part(ones, 0, -1)
        out_dim = num_feas * (num_feas + 1) // 2

        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = num_feas * (num_feas - 1) // 2
        flat_interactions = tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
        return flat_interactions

class DLRM(tf.keras.models.Model):
    def __init__(self,
                 combiner,
                 embed_vec_size,
                 slot_num,
                 dense_dim,
                 arch_bot,
                 arch_top,
                 self_interaction,
                 tf_key_type,
                 tf_vector_type,
                 **kwargs):
        super(DLRM, self).__init__(**kwargs)
        
        self.combiner = combiner
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.tf_key_type = tf_key_type
        self.tf_vector_type = tf_vector_type
 
        self.lookup_layer = collcache_tf2.LookupLayer(
                model_name = "dlrm", 
                table_id = 0,
                emb_vec_size = self.embed_vec_size,
                emb_vec_dtype = self.tf_vector_type)
        self.bot_nn = MLP(arch_bot, name = "bottom", out_activation='relu')
        self.top_nn = MLP(arch_top, name = "top", out_activation='sigmoid')
        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)
        if self_interaction:
            self.interaction_out_dim = (self.slot_num+1) * (self.slot_num+2) // 2
        else:
            self.interaction_out_dim = self.slot_num * (self.slot_num+1) // 2
        self.reshape_layer0 = tf.keras.layers.Reshape((slot_num, arch_bot[-1]), name="reshape0")
        self.reshape_layer1 = tf.keras.layers.Reshape((1, arch_bot[-1]), name = "reshape1")
        self.reshape_layer_final = tf.keras.layers.Reshape((), name = "reshape_final")
        self.concat1 = tf.keras.layers.Concatenate(axis=1, name = "concat1")
        self.concat2 = tf.keras.layers.Concatenate(axis=1, name = "concat2")
            
    def call(self, inputs, training=False):
        global collcache_tf2
        input_cat = inputs[0]
        input_dense = inputs[1]
        
        embedding_vector = self.lookup_layer(input_cat)
        embedding_vector = self.reshape_layer0(embedding_vector)
        dense_x = self.bot_nn(input_dense)
        concat_features = self.concat1([embedding_vector, self.reshape_layer1(dense_x)])
        
        Z = self.interaction_op(concat_features)
        z = self.concat2([dense_x, Z])
        logit = self.top_nn(z)
        return self.reshape_layer_final(logit)

    def summary(self):
        inputs = [tf.keras.Input(shape=(self.slot_num, ), sparse=False, dtype=self.tf_key_type), 
                  tf.keras.Input(shape=(self.dense_dim, ), dtype=tf.float32)]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

def get_run_config():
    argparser = argparse.ArgumentParser("RM INFERENCE")
    argparser.add_argument('--gpu_num',                   type=int,   default=4)
    argparser.add_argument('--iter_num',                  type=int,   default=3000)
    argparser.add_argument('--slot_num',                  type=int,   default=100)
    argparser.add_argument('--dense_dim',                 type=int,   default=13)
    argparser.add_argument('--embed_vec_size',            type=int,   default=128)
    argparser.add_argument('--global_batch_size',         type=int,   default=32768)
    argparser.add_argument('--ps_config_file',            type=str,   default="./config.json")
    argparser.add_argument('--max_vocabulary_size',       type=int,   default=800000000)
    argparser.add_argument('--coll_hotness_profile_iter', type=int,   default=1000)
    argparser.add_argument('--alpha',                     type=float, default=1.2)

    args = vars(argparser.parse_args())
    assert(args["global_batch_size"] % args["gpu_num"] == 0)

    for k, v in args.items():
        print('config:{:}={:}'.format(k, v))
    return args

def random_dataset(args):
    slot_cardinality = args["max_vocabulary_size"] // args["slot_num"]
    # specify vocabulary range
    vocab_ranges = [[0, 0] for i in range(args["slot_num"])]
    max_range = 0
    for i in range(args["slot_num"]):
        vocab_ranges[i][0] = max_range
        vocab_ranges[i][1] = max_range + slot_cardinality
        max_range += slot_cardinality
    assert(max_range == args["max_vocabulary_size"])

    @njit(parallel=True)
    def generate_cat_keys_zipf(num_samples, num_slots, vocab_ranges, key_dtype, alpha):
        dense_keys = np.empty((num_samples, num_slots), key_dtype)
        for i in range(num_slots):
            vocab_range = vocab_ranges[i]
            H = vocab_range[1] - vocab_range[0] + 1
            L = 1
            rnd_rst = np.random.uniform(0.0, 1.0, size=num_samples)
            rnd_rst = ((-H**alpha * L**alpha) / (rnd_rst * (H**alpha - L**alpha) - H**alpha)) ** (1 / alpha)
            keys_per_slot = rnd_rst.astype(key_dtype) + vocab_range[0] - 1
            dense_keys[:, i] = keys_per_slot
        return dense_keys

    @njit(parallel=True)
    def generate_cont_feats(num_samples, dense_dim):
        dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
        labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
        return dense_features, labels

    vocab_ranges = np.array(vocab_ranges)
    replica_batch_size = args["global_batch_size"] // args["gpu_num"]
    print("generating random dataset samples")
    sparse_keys = generate_cat_keys_zipf(replica_batch_size * args["iter_num"], args["slot_num"], vocab_ranges, np.int32, args["alpha"] - 1)
    dense_features, labels = generate_cont_feats(replica_batch_size * args["iter_num"], args["dense_dim"])

    print("creating tf dataset")
    def sequential_batch_gen():
        for i in range(0, replica_batch_size * args["iter_num"], replica_batch_size):
            yield sparse_keys[i:i+replica_batch_size],dense_features[i:i+replica_batch_size],labels[i:i+replica_batch_size]
    dataset = tf.data.Dataset.from_generator(sequential_batch_gen, 
        output_signature=(
            tf.TensorSpec(shape=(replica_batch_size, args["slot_num"]), dtype=tf.int32), 
            tf.TensorSpec(shape=(replica_batch_size, args["dense_dim"]), dtype=tf.float32),
            tf.TensorSpec(shape=(replica_batch_size, 1), dtype=tf.int32))).cache().prefetch(1000)
    return dataset

def worker_func(args, worker_id):
    tf_config = {"task": {"type": "worker", "index": worker_id}, "cluster": {"worker": []}}
    for i in range(args["gpu_num"]): tf_config['cluster']['worker'].append("localhost:" + str(12340+i))
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[worker_id], 'GPU')

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # from https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-997304668
    atexit.register(strategy._extended._cross_device_ops._pool.close)
    atexit.register(strategy._extended._host_cross_device_ops._pool.close)

    with strategy.scope():
        collcache_tf2.Config(global_batch_size = args["global_batch_size"], ps_config_file = args["ps_config_file"])
        barrier.wait()
        # for quick example, the parameters are plain initialized rather than load from disk
        model = DLRM("mean", args["embed_vec_size"], args["slot_num"], args["dense_dim"], 
            arch_bot = [256, 128, args["embed_vec_size"]],
            arch_top = [256, 128, 1],
            tf_key_type = tf.int32, tf_vector_type = tf.float32,
            self_interaction=False)
    barrier.wait()

    dataset = random_dataset(args)

    def _warmup_step(inputs):
        return inputs
    for sparse_keys, dense_features, labels in tqdm(dataset, "warmup run"):
        inputs = [sparse_keys, dense_features]
        _ = strategy.run(_warmup_step, args=(inputs,))
    barrier.wait()

    def _record_hotness(sparse_keys):
        return collcache_tf2.record_hotness(sparse_keys)
    dataset_iter = iter(dataset)
    for i in range(args["coll_hotness_profile_iter"]):
        sparse_keys, _, _ = next(dataset_iter)
        _ = strategy.run(_record_hotness, args=(sparse_keys, ))

    with strategy.scope():
        collcache_tf2.Init(global_batch_size = args["global_batch_size"],
            ps_config_file = args["ps_config_file"])
        barrier.wait()

    dataset_iter = iter(dataset)
    time_sum = 0
    @tf.function
    def _infer_step(inputs):
        logit = model(inputs, training=False)
        return logit
    for i in range(args["iter_num"]):
        sparse_keys, dense_features, labels = next(dataset_iter)
        t1 = tf.timestamp()
        _ = strategy.run(_infer_step, args=([sparse_keys, dense_features],))[0].numpy()
        t2 = tf.timestamp()
        collcache_tf2.SetStepProfileValue(profile_type=collcache_tf2.kLogL1TrainTime, value=(t2 - t1))
        time_sum += t2 - t1
        if (i + 1) % 100 == 0:
            print("[GPU{}] {} time {:.6} ".format(worker_id, i + 1, time_sum / 100), flush=True)
            time_sum = 0
            barrier.wait()

    # collcache internally shares profiled values across different processes, so only report results on master process
    if worker_id == 0:
        collcache_tf2.Report()

if __name__ == "__main__":
    args = get_run_config()
    # set required env before importing collcache module
    os.environ['COLL_NUM_REPLICA'] = str(args["gpu_num"])
    import collcache_tf2
    proc_list = [None for _ in range(args["gpu_num"])]
    barrier = multiprocessing.Barrier(args["gpu_num"])

    for i in range(args["gpu_num"]):
        proc_list[i] = multiprocessing.Process(target=worker_func, args=(args, i))
        proc_list[i].start()
    for i in range(args["gpu_num"]):
        proc_list[i].join()

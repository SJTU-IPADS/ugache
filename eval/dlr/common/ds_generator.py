from numba import njit
import numpy as np
import tensorflow as tf
import os

@njit(parallel=True)
def generate_dense_keys_zipf(num_samples, num_slots, vocabulary_range_per_slot, key_dtype, alpha):
    dense_keys = np.empty((num_samples, num_slots), key_dtype)
    for i in range(num_slots):
        vocab_range = vocabulary_range_per_slot[i]
        H = vocab_range[1] - vocab_range[0] + 1
        L = 1
        rnd_rst = np.random.uniform(0.0, 1.0, size=num_samples)
        rnd_rst = ((-H**alpha * L**alpha) / (rnd_rst * (H**alpha - L**alpha) - H**alpha)) ** (1 / alpha)
        keys_per_slot = rnd_rst.astype(key_dtype) + vocab_range[0] - 1
        dense_keys[:, i] = keys_per_slot
    return dense_keys

@njit(parallel=True)
def generate_dense_keys_uniform(num_samples, num_slots, vocabulary_range_per_slot, key_dtype):
    dense_keys = np.empty((num_samples, num_slots), key_dtype)
    for i in range(num_slots):
        vocab_range = vocabulary_range_per_slot[i]
        keys_per_slot = np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(num_samples)).astype(key_dtype)
        dense_keys[:, i] = keys_per_slot
    return dense_keys

@njit(parallel=True)
def generate_cont_feats(num_samples, dense_dim):
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return dense_features, labels

def generate_random_samples(num_samples, vocabulary_range_per_slot, dense_dim, key_dtype, alpha):
    vocabulary_range_per_slot = np.array(vocabulary_range_per_slot)
    num_slots = vocabulary_range_per_slot.shape[0]
    if alpha == 0:
        cat_keys = generate_dense_keys_uniform(num_samples, num_slots, vocabulary_range_per_slot, key_dtype)
    else:
        cat_keys = generate_dense_keys_zipf(num_samples, num_slots, vocabulary_range_per_slot, key_dtype, alpha)
    dense_features, labels = generate_cont_feats(num_samples, dense_dim)
    return cat_keys, dense_features, labels

def criteo_tb(fnames, replica_batch_size, iter_num, num_replica, key_type):
    assert(len(fnames) == 1)
    fname = fnames[0]
    print("load criteo from ", fname + ".sparse")
    file_num_samples = os.stat(fname+".label").st_size
    assert(file_num_samples % 4 == 0)
    file_num_samples = file_num_samples // 4

    print(file_num_samples)
    if file_num_samples < iter_num * replica_batch_size * num_replica:
        print(f"ds contains {file_num_samples} samples, not enough for ", iter_num * replica_batch_size * num_replica)
        assert(False)
    file_num_samples = iter_num * replica_batch_size * num_replica

    sparse_keys = np.memmap(fname+".sparse", dtype='int32', mode='r', shape=(file_num_samples, 26))
    dense_features = np.memmap(fname+".dense", dtype='float32', mode='r', shape=(file_num_samples, 13))
    labels = np.memmap(fname+".label", dtype='int32', mode='r', shape=(file_num_samples, 1))

    sparse_keys = sparse_keys[:iter_num * replica_batch_size * num_replica, :]
    dense_features = dense_features[:iter_num * replica_batch_size * num_replica, :]
    labels = labels[:iter_num * replica_batch_size * num_replica, :]

    def sequential_batch_gen():
        for i in range(0, replica_batch_size * iter_num * num_replica, replica_batch_size):
            sparse_keys, dense_features, labels
            yield sparse_keys[i:i+replica_batch_size],dense_features[i:i+replica_batch_size],labels[i:i+replica_batch_size]
    dataset = tf.data.Dataset.from_generator(sequential_batch_gen, 
        output_signature=(
            tf.TensorSpec(shape=(replica_batch_size, 26), dtype=key_type), 
            tf.TensorSpec(shape=(replica_batch_size, 13), dtype=tf.float32),
            tf.TensorSpec(shape=(replica_batch_size, 1), dtype=tf.int32)))

    return dataset

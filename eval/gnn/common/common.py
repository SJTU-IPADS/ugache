import torch
import time
kCacheByDegree          = 0
kCacheByHeuristic       = 1
kCacheByPreSample       = 2
kCacheByDegreeHop       = 3
kCacheByPreSampleStatic = 4
kCacheByFakeOptimal     = 5
kDynamicCache           = 6
kCacheByRandom          = 7
kCollCache              = 8
kCollCacheIntuitive     = 9
kPartitionCache         = 10
kPartRepCache           = 11
kRepCache               = 12
kCollCacheAsymmLink     = 13
kCliquePart             = 14
kCliquePartByDegree     = 15

cache_policy_map = {
    'coll_cache'            : kCollCache,
    'coll_intuitive'        : kCollCacheIntuitive,
    'partition'             : kPartitionCache,
    'part_rep'              : kPartRepCache,
    'rep'                   : kRepCache,
    'coll_cache_asymm_link' : kCollCacheAsymmLink,
    'clique_part'           : kCliquePart,
    'clique_part_by_degree' : kCliquePartByDegree,
}
    

def generate_config(run_config):
    config = {}
    config["cache_percentage"] = run_config['cache_percentage']
    config["_cache_policy"] = cache_policy_map[run_config['cache_policy']]
    config["num_device"] = run_config['num_worker']
    config["num_global_step_per_epoch"] = run_config['num_worker'] * run_config['local_step']
    config["num_epoch"] = run_config['epochs']
    config["omp_thread_num"] = run_config['omp_thread_num']
    return config

def parse_max_neighbors(num_layer, neighbor_str):
    neighbor_str_vec = neighbor_str.split(",")
    max_neighbors = []
    for ns in neighbor_str_vec:
        max_neighbors.append(int(ns))
    assert len(max_neighbors) == 1 or len(max_neighbors) == num_layer
    if len(max_neighbors) != num_layer:
        for i in range(1, num_layer):
            max_neighbors.append(max_neighbors[0])
    return max_neighbors

def print_run_config(run_config):
    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        if not k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

    for k, v in run_config.items():
        if k.startswith('_'):
            print('config:{:}={:}'.format(k, v))

num_class_dict = {
    'papers100M'      : 172,
    'ogbn-papers100M' : 172,
    'uk-2006-05'      : 150,
    'com-friendster'  : 100,
    'mag240m-homo'    : 153,
}

feat_dtype_dict = {
    'papers100M'      : torch.float32,
    'ogbn-papers100M' : torch.float32,
    'uk-2006-05'      : torch.float32,
    'com-friendster'  : torch.float32,
    'mag240m-homo'    : torch.float16,
}
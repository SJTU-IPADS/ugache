import argparse
import datetime
import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time
import numpy as np
import math
import sys
import os
import collcache.torch as co

class MetaReader(object):
    def __init__(self):
        pass

    def read(self, folder):
        meta = {'FEAT_DATA_TYPE' : 'F32'}
        with open(os.path.join(folder, 'meta.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                assert len(line) == 2
                if line[0] == 'FEAT_DATA_TYPE':
                    meta[line[0]] = line[1]
                else:
                    meta[line[0]] = int(line[1])

        meta_keys = meta.keys()

        assert('NUM_NODE' in meta_keys)
        assert('NUM_EDGE' in meta_keys)
        assert('FEAT_DIM' in meta_keys)
        assert('NUM_CLASS' in meta_keys)
        assert('NUM_TRAIN_SET' in meta_keys)
        assert('NUM_VALID_SET' in meta_keys)
        assert('NUM_TEST_SET' in meta_keys)

        return meta

class DatasetLoader:
    _data_type_name_map = {
        'F32': ('float32', torch.float32),
        'F16': ('float16', torch.float16),
    }
    def __init__(self, dataset_path):
        tic = time.time()

        meta_reader = MetaReader()
        meta = meta_reader.read(dataset_path)

        self.num_node = meta['NUM_NODE']
        self.num_edge = meta['NUM_EDGE']
        self.feat_dim = meta['FEAT_DIM']
        self.num_class = meta['NUM_CLASS']
        self.num_train_set = meta['NUM_TRAIN_SET']
        self.num_valid_set = meta['NUM_VALID_SET']
        self.num_test_set = meta['NUM_TEST_SET']

        if self.num_edge >= 2**31:
            raise Exception("num edge exceeds int max")
        self.load32(dataset_path)

        if meta['FEAT_DATA_TYPE'] not in DatasetLoader._data_type_name_map:
            raise Exception(f"unsupported feature datatype {meta['FEAT_DATA_TYPE']}")
        if os.path.isfile(os.path.join(dataset_path, 'feat.bin')):
            dtype = DatasetLoader._data_type_name_map[meta['FEAT_DATA_TYPE']][0]
            self.feat = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'feat.bin'), dtype=dtype, mode='r', shape=(self.num_node, self.feat_dim)))
        else:
            dtype = DatasetLoader._data_type_name_map[meta['FEAT_DATA_TYPE']][1]
            self.feat = torch.empty((self.num_node, self.feat_dim), dtype=dtype)
        if os.path.isfile(os.path.join(dataset_path, 'label.bin')):
            self.label = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'label.bin'), dtype='long', mode='r', shape=(self.num_node,)))
        else:
            self.label = torch.empty((self.num_node, ), dtype=torch.long)

        toc = time.time()
        print('Loading {:s} uses {:4f} secs.'.format(dataset_path, toc-tic))

    def load32(self, dataset_path):
        self.indptr  = torch.from_numpy(np.memmap(os.path.join(dataset_path,  'indptr.bin'), dtype='int32', mode='r', shape=(self.num_node + 1,)))
        self.indices = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'indices.bin'), dtype='int32', mode='r', shape=(self.num_edge,)))

        self.train_set = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'train_set.bin'), dtype='int32', mode='r', shape=(self.num_train_set,)))
        self.valid_set = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'valid_set.bin'), dtype='int32', mode='r', shape=(self.num_valid_set,)))
        self.test_set  = torch.from_numpy(np.memmap(os.path.join(dataset_path,  'test_set.bin'), dtype='int32', mode='r', shape=(self.num_test_set,)))

    def to_dgl_graph(self, g_format='csc'):
        import dgl

        if g_format == 'csc':
            g = dgl.graph(data = ('csc', (self.indptr, self.indices, torch.empty(0))), num_nodes = self.num_node)
        elif g_format == 'csr':
            g = dgl.graph(data = ('csr', (self.indptr, self.indices, torch.empty(0))), num_nodes = self.num_node)
        else:
            assert(False)

        return g

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GraphSage Training")
    argparser.add_argument('--devices', nargs='+',type=int, default=default_run_config['devices'])
    argparser.add_argument('--dataset', type=str,default=default_run_config['dataset'])
    argparser.add_argument('--root-path', type=str,default=default_run_config['root_path'])

    argparser.add_argument('--fanout', nargs='+',type=int, default=default_run_config['fanout'])
    argparser.add_argument('--num-epoch', type=int,default=default_run_config['num_epoch'])
    argparser.add_argument('--num-hidden', type=int,default=default_run_config['num_hidden'])
    argparser.add_argument('--batch-size', type=int,default=default_run_config['batch_size'])
    argparser.add_argument('--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,default=default_run_config['dropout'])
    argparser.add_argument('--cache_percentage', type=float,default=default_run_config['cache_percentage'])

    return vars(argparser.parse_args())

def get_run_config():
    default_run_config = {}
    default_run_config['cache_percentage'] = 0.1
    default_run_config['devices'] = [0, 1]
    default_run_config['devices'] = [0, 1, 2, 3]
    default_run_config['dataset'] = 'products-undir'
    default_run_config['root_path'] = '/datasets_gnn/gnnlab'

    # DGL fanouts from front to back are from leaf to root
    default_run_config['fanout'] = [25, 10]
    default_run_config['num_epoch'] = 10
    default_run_config['num_hidden'] = 256
    default_run_config['batch_size'] = 8000
    default_run_config['lr'] = 0.003
    default_run_config['dropout'] = 0.5

    run_config = parse_args(default_run_config)

    assert(len(run_config['devices']) > 0)

    # the first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_fanout'] = run_config['num_layer'] = len(run_config['fanout'])
    run_config['num_worker'] = len(run_config['devices'])

    dataset = DatasetLoader(os.path.join(run_config['root_path'], run_config['dataset']))

    run_config['prefetch_factor'] = 2

    run_config['sample_devices'] = run_config['devices']
    run_config['train_devices'] = run_config['devices']

    run_config['num_thread'] = torch.get_num_threads() // run_config['num_worker']

    print('config:eval_tsp="{:}"'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        print('config:{:}={:}'.format(k, v))

    run_config['dataset'] = dataset
    run_config['g'] = dataset.to_dgl_graph()

    return run_config

def generate_coll_config(run_config):
    config = {}
    config["cache_percentage"] = run_config['cache_percentage']
    config["_cache_policy"] = 13
    config["num_device"] = run_config['num_worker']
    config["num_global_step_per_epoch"] = run_config['num_worker'] * run_config['local_step']
    config["num_epoch"] = run_config['num_epoch']
    config["omp_thread_num"] = 40
    return config

def run(worker_id, run_config):
    torch.set_num_threads(run_config['num_thread'])
    sample_device = torch.device(run_config['sample_devices'][worker_id])
    train_device = torch.device(run_config['train_devices'][worker_id])
    num_worker = run_config['num_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=600))

    dataset = run_config['dataset']
    g = run_config['g'].to(sample_device)
    label = dataset.label

    train_nids = dataset.train_set.to(sample_device)
    in_feats = dataset.feat_dim
    n_classes = dataset.num_class

    sampler = dgl.dataloading.MultiLayerNeighborSampler(run_config['fanout'])
    dataloader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler, use_ddp=num_worker > 1, batch_size=run_config['batch_size'], shuffle=True, drop_last=False)
    run_config['local_step'] = len(dataloader)

    model = SAGE(in_feats, run_config['num_hidden'], n_classes, run_config['num_layer'], F.relu, run_config['dropout'])
    model = model.to(train_device)
    if num_worker > 1:
        model = DistributedDataParallel(model, device_ids=[train_device], output_device=train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(model.parameters(), lr=run_config['lr'])
    num_epoch = run_config['num_epoch']

    model.train()
    
    config = generate_coll_config(run_config)
    config["num_total_item"] = dataset.num_node
    co.config(config)
    co.coll_cache_record_init(worker_id)

    for presc_epoch in range(2):
        for step, (input_nodes, _, _) in enumerate(iter(dataloader)):
            co.coll_torch_record(worker_id, input_nodes.to('cpu'))
    node_feat = co.coll_torch_create_emb_shm(worker_id, dataset.num_node, in_feats, torch.float32)
    co.coll_torch_init_t(worker_id, worker_id, node_feat, run_config["cache_percentage"])

    step_key = worker_id
    for epoch in range(num_epoch):
        tic = time.time()
        ds_iter = iter(dataloader)
        for step in range(len(dataloader)):
            t0 = time.time()
            input_nodes, output_nodes, blocks = next(ds_iter)
            blocks = [block.int().to(train_device) for block in blocks]
            batch_labels = torch.index_select(label, 0, output_nodes.to(label.device)).to(train_device)
            torch.cuda.synchronize()
            t1 = time.time()

            batch_inputs = co.coll_torch_lookup_key_t_val_ret(worker_id, input_nodes, pad_to_8=False)
            torch.cuda.synchronize()
            t2 = time.time()

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t3 = time.time()

            co.log_step_by_key(step_key, co.kLogL1NumNode,    blocks[0].num_src_nodes())
            co.log_step_by_key(step_key, co.kLogL1NumSample,  sum([block.num_edges() for block in blocks]))
            co.log_step_by_key(step_key, co.kLogL1SampleTime, t1 - t0)
            co.log_step_by_key(step_key, co.kLogL1CopyTime,   t2 - t1)
            co.log_step_by_key(step_key, co.kLogL1TrainTime,  t3 - t2)
            step_key += num_worker

        torch.cuda.synchronize()

        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()
        if worker_id == 0:
            print(f"[Epoch {epoch}], time={toc - tic}")

    if num_worker > 1:
        torch.distributed.barrier()
    if worker_id == 0:
        co.report_step_average(0)

if __name__ == '__main__':
    run_config = get_run_config()
    num_worker = run_config['num_worker']

    if num_worker == 1:
        run(0, run_config)
    else:
        workers = []
        for worker_id in range(num_worker):
            p = mp.Process(target=run, args=(worker_id, run_config))
            p.start()
            workers.append(p)
        
        ret = os.waitpid(-1, 0)
        if os.WEXITSTATUS(ret[1]) != 0:
            print("Detect pid {:} error exit".format(ret[0]))
            for p in workers:
                p.kill()
            
        for p in workers:
                p.join()
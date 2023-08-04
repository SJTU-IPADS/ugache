# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import time
from optparse import OptionParser

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from dgl.heterograph import DGLBlock, DGLHeteroGraph
import dgl.multiprocessing as mp
import dgl
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from wg_torch import comm as comm
from wg_torch import embedding_ops as embedding_ops
from wg_torch import graph_ops as graph_ops
from wg_torch.wm_tensor import *

from wholegraph.torch import wholegraph_pytorch as wg
import collcache.torch as co
from common import generate_config, parse_max_neighbors, print_run_config, feat_dtype_dict, num_class_dict
from wg_util import BatchDataPack

def wg_params(parser):
    parser.add_option(
        "-c", "--num_workers", type="int", dest="num_worker", default=8, help="number of workers"
    )
    parser.add_option(
        "--num_intra_size", type="int", dest="num_intra_size", default=8, help="number of local workers"
    )
    parser.add_option(
        "-r",
        "--root_dir",
        dest="root_dir",
        default="/nvme/songxiaoniu/graph-learning/wholegraph",
        # default="/dev/shm/dataset",
        help="dataset root directory.",
    )
    parser.add_option(
        "-g",
        "--graph_name",
        dest="graph_name",
        default="ogbn-papers100M",
        # default="papers100M",
        help="graph name",
    )
    parser.add_option(
        "-e", "--epochs", type="int", dest="epochs", default=4, help="number of epochs"
    )
    parser.add_option(
        "-b", "--batchsize", type="int", dest="batchsize", default=8000, help="batch size"
    )
    parser.add_option("--skip_epoch", type="int", dest="skip_epoch", default=3, help="num of skip epoch for profile")
    parser.add_option("--local_step", type="int", dest="local_step", default=19, help="num of steps on a GPU in an epoch")
    parser.add_option("--presc_epoch", type="int", dest="presc_epoch", default=2, help="epochs to pre-sample")
    parser.add_option(
        "-n",
        "--neighbors",
        dest="neighbors",
        # default="5,5",
        default="15,10,5",
        help="train neighboor sample count",
    )
    parser.add_option(
        "--hiddensize", type="int", dest="hiddensize", default=256, help="hidden size"
    )
    parser.add_option(
        "-l", "--layernum", type="int", dest="num_layer", default=3, help="layer number"
    )
    parser.add_option(
        "-m",
        "--model",
        dest="model",
        default="sage",
        help="model type, valid values are: sage, gcn, gat",
    )
    parser.add_option(
        "-f",
        "--framework",
        dest="framework",
        default="dgl",
        help="framework type, valid values are: dgl, pyg, wg",
    )
    parser.add_option(
        "-w",
        "--dataloaderworkers",
        type="int",
        dest="dataloaderworkers",
        default=0,
        help="number of workers for dataloader",
    )
    parser.add_option(
        "-d", "--dropout", type="float", dest="dropout", default=0.5, help="dropout"
    )
    parser.add_option(
        "--weight_decay", type="float", dest="weight_decay", default=0.0005, help="weight_decay"
    )
    parser.add_option("--lr", type="float", dest="lr", default=0.003, help="learning rate")
    parser.add_option(
        "--use_nccl",
        action="store_true",
        dest="use_nccl",
        default=False,
        help="whether use nccl for embeddings, default False",
    )
    parser.add_option(
        "--amp",
        action="store_true",
        dest="use_amp",
        default=False,
        help="whether use amp for training, default True",
    )
    parser.add_option("--use_collcache", action="store_true", dest="use_collcache", default=False, help="use collcache lib")
    parser.add_option("--cache_percentage", type="float", dest="cache_percentage", default=0.25, help="cache percent of collcache")
    parser.add_option("--cache_policy", type="str", dest="cache_policy", default="coll_cache_asymm_link", help="cache policy of collcache")
    parser.add_option("--omp_thread_num", type="int", dest="omp_thread_num", default=40, help="omp thread num of collcache")


def create_gnn_layers(in_feat_dim, hidden_feat_dim, class_count, num_layer, activation):
    if run_config["framework"] == "dgl":
        from dgl.nn.pytorch.conv import SAGEConv, GATConv
        from dgl.nn.pytorch import GraphConv
    elif run_config["framework"] == "wg":
        from wg_torch.gnn.SAGEConv import SAGEConv
        from wg_torch.gnn.GATConv import GATConv
    gnn_layers = torch.nn.ModuleList()
    for i in range(num_layer):
        layer_output_dim = (
            hidden_feat_dim if i != num_layer - 1 else class_count
        )
        layer_input_dim = in_feat_dim if i == 0 else hidden_feat_dim
        if run_config["framework"] == "dgl":
            if run_config["model"] == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "mean"))
            else:
                assert run_config["model"] == "gcn"
                gnn_layers.append(GraphConv(layer_input_dim, layer_output_dim, activation=activation if i != num_layer - 1 else None, allow_zero_in_degree=True))
        elif run_config["framework"] == "wg":
            if run_config["model"] == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            else:
                assert run_config["model"] == "gcn"
                assert False
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, aggregator="gcn"))
    return gnn_layers

def layer_forward(layer, x_feat, sub_graph, target_gid_cnt):
    if run_config["framework"] == "dgl":
        x_feat = layer(sub_graph, x_feat)
    elif run_config["framework"] == "wg":
        x_target_feat = x_feat[: target_gid_cnt]
        x_feat = layer(sub_graph[0], sub_graph[1], sub_graph[2], x_feat, x_target_feat)
    return x_feat

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = create_gnn_layers(in_feats, n_hidden, n_classes, n_layers, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, target_gid_cnt, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != 0:
                h = self.dropout(h)
            h = layer_forward(layer, h, block, target_gid_cnt[l])
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        if use_amp:
            u, v = g.edges()
            u=u.long()
            v=v.long()
            return (h[u] * h[v]).sum(1)
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

def create_train_dataset(data_tensor_dict, rank, size):
    return DataLoader(
        dataset=graph_ops.NodeClassificationDataset(data_tensor_dict, rank, size),
        batch_size=run_config["batchsize"],
        shuffle=True,
        num_workers=run_config["dataloaderworkers"],
        pin_memory=True,
        drop_last=True
    )

def do_unsup_sample(iter_id, dist_homo_graph):
    src_nid, pos_dst_nid = dist_homo_graph.get_train_edge_batch(iter_id)
    bs = src_nid.size(0)
    neg_dst_nid = dist_homo_graph.per_source_negative_sample(src_nid)
    assert src_nid.shape == pos_dst_nid.shape and src_nid.shape == neg_dst_nid.shape
    orig_g_ids = torch.cat([src_nid, pos_dst_nid, neg_dst_nid])
    orig_g_ids_unique, reverse_map = torch.unique(orig_g_ids, return_inverse=True)

    # sample
    orig_g_ids_unique = orig_g_ids_unique.to(dist_homo_graph.id_type()).cuda()
    graph_block = dist_homo_graph.unweighted_sample_without_replacement(orig_g_ids_unique, run_config['max_neighbors'])

    ret = BatchDataPack()
    ret.wg_graph_block = graph_block
    ret.pair_graph_num_node = orig_g_ids_unique.size(0)
    remapped_src = reverse_map[:bs]
    ret.pair_graph_src = remapped_src.repeat(2)
    ret.pair_graph_dst = reverse_map[bs:]
    return ret

def ds_load(worker_id, run_config):
    wm_comm = create_intra_node_communicator(run_config["worker_id"], run_config["num_worker"], run_config["num_intra_size"])
    wm_embedding_comm = None
    if run_config["use_nccl"]:
        wm_embedding_comm = create_global_communicator(run_config["worker_id"], run_config["num_worker"])
    ignore_embeddings = None
    if run_config['use_collcache']:
        ignore_embeddings=['paper', 'node']

    # graph loading
    dist_homo_graph = graph_ops.HomoGraph()
    use_chunked = True
    use_host_memory = False
    dist_homo_graph.load(
        run_config["root_dir"],
        run_config["graph_name"],
        wm_comm,
        use_chunked,
        use_host_memory,
        wm_embedding_comm,
        link_pred_task=run_config['unsupervised'],
        ignore_embeddings=ignore_embeddings,
    )
    max_total_local_steps = dist_homo_graph.start_iter(run_config['batchsize'])
    print(f"Rank={worker_id}, Graph loaded.")
    
    iter_start_time = time.time()
    for iter_id in range(options.local_step):
        src_nid, pos_dst_nid = dist_homo_graph.get_train_edge_batch(iter_id)
    torch.cuda.synchronize()
    test_end_time = time.time()
    print(
        "!!!!dist_homo_graph enumerate latency per epoch: %f, per step: %f"
        % ((test_end_time - iter_start_time), ((test_end_time - iter_start_time) / options.local_step))
    )
    return dist_homo_graph, None

def main(worker_id, run_config):
    with open(f'/tmp/infer_{worker_id}.pid', 'w') as f:
        print(os.getpid(), file=f)
    print(f"worker {worker_id} running with pid={os.getpid()}")
    if worker_id == 0: print_run_config(run_config)

    num_worker = run_config['num_worker']
    global_barrier = run_config['global_barrier']

    ctx = f"cuda:{worker_id}"
    device = torch.device(ctx)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    except:
        pass
    try:
        torch.set_float32_matmul_precision("medium")
    except:
        pass

    wg.init_lib()

    ###################
    # torch multi process
    # fixme
    torch.set_num_threads(run_config['omp_thread_num'] // run_config['num_worker'])
    os.environ["RANK"] = str(worker_id)
    os.environ["WORLD_SIZE"] = str(run_config["num_worker"])
    # slurm in Selene has MASTER_ADDR env
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12335"
    assert num_worker <= torch.cuda.device_count()

    torch.cuda.set_device(worker_id)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    ###################
    # dataset loading
    dist_homo_graph, _ = ds_load(worker_id, run_config)
    # run_config["local_step"] = min(run_config["local_step"], len(train_data_list))

    ###################
    # get config for collcache
    # if run_config["use_collcache"]:
    config = generate_config(run_config)
    config["num_total_item"] = dist_homo_graph.node_count
    co.config(config)
    co.coll_cache_record_init(worker_id)


    def wg_gather_fn_builder():
        gather_fn = embedding_ops.EmbeddingLookUpModule(need_backward=False).cuda()
        def gather(keys):
            return gather_fn(keys, dist_homo_graph.node_feat)
        return gather
    def co_gather_fn_builder():
        def gather(keys):
            return co.coll_torch_lookup_key_t_val_ret(worker_id, keys)
        return gather
    if run_config["use_collcache"]:
        gather_fn = co_gather_fn_builder()
    else:
        gather_fn = wg_gather_fn_builder()
    


    # model initialize
    in_feat = dist_homo_graph.node_info['emb_dim']
    num_class = run_config["classnum"]
    num_layer = run_config['num_layer']
    run_config["max_neighbors"] = parse_max_neighbors(run_config['num_layer'], run_config["neighbors"])

    model = GCN(in_feat, run_config["hiddensize"], run_config["hiddensize"],
                 num_layer, F.relu, run_config['dropout'])
    model = model.to(device)
    predictor = DotPredictor().to(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    if len(list(predictor.parameters())) > 0:
        predictor = DistributedDataParallel(predictor, device_ids=[device], output_device=device)

    # loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=run_config['lr'], weight_decay=run_config['weight_decay'])

    scaler = GradScaler()

    num_epoch = run_config['epochs']
    total_steps = run_config["epochs"]* run_config["local_step"]
    if worker_id == 0:
        print(f"epoch={num_epoch} total_steps={total_steps}")
    model.train()

    global_barrier.wait()

    # presample
    if run_config["use_collcache"]:
        presc_start = time.time()
        print("presamping")
        for presc_epoch in range(run_config['presc_epoch']):
            for iter_id in range(run_config['local_step']):
                batch_data_pack = do_unsup_sample(iter_id, dist_homo_graph)
                block_input_nodes = batch_data_pack.input_keys.to('cpu')
                torch.cuda.synchronize()
                co.coll_torch_record(worker_id, block_input_nodes)
        presc_stop = time.time()
        print(f"presamping takes {presc_stop - presc_start}")

        if worker_id == 0:
            co.print_memory_usage()

        node_feat = co.coll_torch_create_emb_shm(worker_id, dist_homo_graph.node_count, dist_homo_graph.node_info['emb_dim'], feat_dtype_dict[run_config['graph_name']])
        co.coll_torch_init_t(worker_id, worker_id, node_feat, run_config["cache_percentage"])

    if worker_id == 0:
        print("start training...")
    torch.cuda.synchronize()
    train_start_time = time.time()
    skip_epoch_time = time.time()
    latency_s = 0
    latency_e = 0
    latency_t = 0
    latency_total = 0
    step_key = worker_id
    labels_reuse = torch.cat([torch.ones(run_config['batchsize']), torch.zeros(run_config['batchsize'])]).cuda()
    for epoch in range(num_epoch):
        if epoch == run_config["skip_epoch"]:
            torch.cuda.synchronize()
            skip_epoch_time = time.time()
            latency_s = 0
            latency_e = 0
            latency_t = 0
        global_barrier.wait()
        epoch_start_time = time.time()
        for iter_id in range(run_config['local_step']):

            optimizer.zero_grad(set_to_none=True)
            step_start_time = time.time()
            # sample
            batch_data_pack = do_unsup_sample(iter_id, dist_homo_graph)
            sub_graphs, target_gid_cnt = batch_data_pack.get_graph(run_config['framework'], run_config['use_collcache'])
            gather_keys = batch_data_pack.input_keys
            batch_label = labels_reuse
            pair_graph = batch_data_pack.to_dgl_pair_graph(run_config['use_collcache'])
            torch.cuda.synchronize()
            sample_end_time = time.time()

            # extract
            x_feat = gather_fn(gather_keys)
            torch.cuda.synchronize()
            extract_end_time = time.time()

            # train
            if run_config["use_amp"]:
                with autocast(enabled=run_config["use_amp"]):
                    batch_pred = model(sub_graphs, target_gid_cnt, x_feat)
                    score = predictor(pair_graph, batch_pred)
                    loss = F.binary_cross_entropy_with_logits(score, batch_label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_pred = model(sub_graphs, target_gid_cnt, x_feat)
                score = predictor(pair_graph, batch_pred)
                loss = F.binary_cross_entropy_with_logits(score, batch_label)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            step_end_time = time.time()

            # SET time profile
            co.log_step_by_key(step_key, co.kLogL1SampleTime, sample_end_time - step_start_time)
            co.log_step_by_key(step_key, co.kLogL1CopyTime,   extract_end_time - sample_end_time)
            co.log_step_by_key(step_key, co.kLogL2CacheCopyTime,   extract_end_time - sample_end_time)
            co.log_step_by_key(step_key, co.kLogL1TrainTime,  step_end_time - extract_end_time)
            latency_s += (sample_end_time - step_start_time)
            latency_e += (extract_end_time - sample_end_time)
            latency_t += (step_end_time - extract_end_time)
            latency_total += (step_end_time - step_start_time)
            step_key += num_worker
            if worker_id == 0:
                print(f"[Epoch {epoch}][Step {iter_id}], time={step_end_time - step_start_time}, ext_time={extract_end_time - sample_end_time}, train_time={step_end_time - extract_end_time}")
        global_barrier.wait()
        epoch_end_time = time.time()
        if worker_id == 0:
            print(f"[Epoch {epoch}], time={epoch_end_time - epoch_start_time}, loss={loss.cpu().item()}")
    torch.cuda.synchronize()
    train_end_time = time.time()
    
    global_barrier.wait()
    if worker_id == 0:
        print(torch.cuda.memory_summary())
        print(
            "[TRAIN_TIME] train time is %.6f seconds"
            % (train_end_time - train_start_time)
        )
        print(
            "[EPOCH_TIME] %.6f seconds, maybe large due to not enough epoch skipped."
            % ((train_end_time - train_start_time) / run_config["epochs"])
        )
        print(
            "[EPOCH_TIME] %.6f seconds"
            % ((train_end_time - skip_epoch_time) / (run_config["epochs"] - run_config["skip_epoch"]))
        )
        co.report_step_average(0)
        co.print_memory_usage()
    wg.finalize_lib()


if __name__ == "__main__":

    parser = OptionParser()
    wg_params(parser)
    (options, args) = parser.parse_args()
    run_config = vars(options)
    run_config['unsupervised'] = True
    run_config["classnum"] = num_class_dict[run_config["graph_name"]]
    use_amp = run_config['use_amp']

    num_worker = run_config["num_worker"]
    # global barrier is used to sync all the sample workers and train workers
    run_config["global_barrier"] = mp.Barrier(num_worker, timeout=300.0)

    # fork child processes
    workers = []
    for worker_id in range(num_worker):
        run_config["worker_id"] = worker_id
        p = mp.Process(target=main, args=(worker_id, run_config, ))
        p.start()
        workers.append(p)

    ret = wg.wait_one_child()
    if ret != 0:
        for p in workers:
            p.kill()
    for p in workers:
        p.join()

    if ret != 0:
        sys.exit(1)

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

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import pickle, json

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip'

RAW_DATA_DIR = '/datasets_gnn/data-raw'
PAPERS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/papers100M-bin'

GNNLAB_OUTPUT_DATA_DIR = '/datasets_gnn/gnnlab/papers100M-undir'
WHOLEGRAPH_OUTPUT_DATA_DIR = '/datasets_gnn/wholegraph/ogbn_papers100M/converted'

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/papers100M-bin.zip'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/papers100M-bin.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{PAPERS_RAW_DATA_DIR}/unzipped'):
        print('Start unziping...')
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/papers100M-bin.zip') == 0)
        assert(os.system(f'touch {PAPERS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_split_feat_label_gnnlab():
    print('Reading split raw data...')
    train_idx = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/train.csv.gz', compression='gzip', header=None).values.T[0]
    valid_idx = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/valid.csv.gz', compression='gzip', header=None).values.T[0]
    test_idx  = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/test.csv.gz',  compression='gzip', header=None).values.T[0]

    # file0 = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/data.npz')
    file1 = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/node-label.npz')

    # features = file0['node_feat']
    label = file1['node_label']

    print('Writing split files for gnnlab...')
    train_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/test_set.bin')
    # features.astype('float32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/feat.bin')
    label.astype('uint64').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/label.bin')

def write_split_feat_label_wholegraph():
    print('Reading split raw data...')
    train_idx = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/train.csv.gz', compression='gzip', header=None).values.T[0]

    file1 = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/node-label.npz')
    label = file1['node_label']

    data_and_label = {
        "train_idx"   : None,
        "train_label" : None,
        "valid_idx"   : None,
        "valid_label" : None,
        "test_idx"    : None,
        "test_label"  : None,
    }

    data_and_label['train_idx'] = train_idx.astype('uint32')
    data_and_label['train_label'] = label.astype('float32')[data_and_label['train_idx']]
    output_fname = f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_papers100M_data_and_label.pkl"

    print('Writing split files for wholegraph...')
    with open(output_fname, "wb") as f:
        pickle.dump(data_and_label, f)
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_papers100M_edge_index_paper_cites_paper_part_0_of_1')
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_papers100M_node_feat_paper_part_0_of_1')

def soft_link_graph_topo_gnnlab():
    indptr = np.memmap(f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_row_ptr", dtype='uint32', mode='r')
    indptr.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/indptr.bin')
    os.system(f'ln -s {os.path.relpath(WHOLEGRAPH_OUTPUT_DATA_DIR, GNNLAB_OUTPUT_DATA_DIR)}/homograph_csr_col_idx {GNNLAB_OUTPUT_DATA_DIR}/indices.bin')

def gen_undir_graph_wholegraph_cpu():
    print('Reading raw graph topo...')
    file0 = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/data.npz', mmap_mode='r')
    num_nodes = file0['num_nodes_list'][0]

    edge_index = file0['edge_index']
    src = edge_index[0]
    dst = edge_index[1]
    data = np.zeros(src.shape)
    coo = coo_matrix((data, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint32)
    csr = coo.tocsr()
    del coo, src, dst, edge_index

    indptr = csr.indptr
    indices = csr.indices

    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    data = np.ones(num_edges)

    print('Converting topo...')
    csr = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    csr += csr.transpose(copy=True)

    indptr = csr.indptr
    indices = csr.indices

    print('Writing topo...')
    indptr.astype('uint64').tofile(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_row_ptr')
    indices.astype('uint32').tofile(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_col_idx')

def write_gnnlab_meta():
    print('Writing meta file...')
    with open(f'{GNNLAB_OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 111059956))
        f.write('{}\t{}\n'.format('NUM_EDGE', 3228124712))
        f.write('{}\t{}\n'.format('FEAT_DIM', 128))
        f.write('{}\t{}\n'.format('NUM_CLASS', 172))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 1207179))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 125265))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 214338))
def write_wholegraph_meta():
    print('Writing meta file...')
    meta = {
        "nodes": [{
            "name": "paper",
            "has_emb": True,
            "emb_file_prefix": "ogbn_papers100M_node_feat_paper",
            "num_nodes": 111059956,
            "emb_dim": 128,
            "dtype": "float32"
        }],
        "edges": [{
            "src": "paper",
            "dst": "paper",
            "rel": "cites",
            "has_emb": False,
            "edge_list_prefix": "ogbn_papers100M_edge_index_paper_cites_paper",
            "num_edges": 3228124712,
            "dtype": "int32",
            "directed": True
        }]
    }
    with open(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_papers100M_meta.json', 'w') as f:
        json.dump(meta, f)

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {PAPERS_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {GNNLAB_OUTPUT_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {WHOLEGRAPH_OUTPUT_DATA_DIR}') == 0)

    download_data()
    write_split_feat_label_gnnlab()
    write_split_feat_label_wholegraph()
    gen_undir_graph_wholegraph_cpu()
    soft_link_graph_topo_gnnlab()
    write_gnnlab_meta()
    write_wholegraph_meta()
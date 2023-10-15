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
import pickle, json, torch

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip'

RAW_DATA_DIR = '/datasets_gnn/data-raw'
PRODUCTS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/products'

GNNLAB_OUTPUT_DATA_DIR = '/datasets_gnn/gnnlab/products-undir'
WHOLEGRAPH_OUTPUT_DATA_DIR = '/datasets_gnn/wholegraph/ogbn_products/converted'

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/products.zip'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/products.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{PRODUCTS_RAW_DATA_DIR}/unzipped'):
        print('Start unziping...')
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/products.zip') == 0)
        assert(os.system(f'touch {PRODUCTS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_split_feat_label_gnnlab():
    print('Reading split raw data...')
    
    train_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/train.csv.gz', compression='gzip', header=None).values.T[0]
    valid_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/valid.csv.gz', compression='gzip', header=None).values.T[0]
    test_idx  = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/test.csv.gz',  compression='gzip', header=None).values.T[0]

    # feature = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-feat.csv.gz',  compression='gzip', header=None).values
    label   = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-label.csv.gz', compression='gzip', header=None).values.T[0]

    print('Writing split files for gnnlab...')
    train_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/test_set.bin')
    # feature.astype('float32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/feat.bin')
    label.astype('uint64').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/label.bin')

def write_split_feat_label_wholegraph():
    print('Reading split raw data...')
    train_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/train.csv.gz', compression='gzip', header=None).values.T[0]

    label = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-label.csv.gz', compression='gzip', header=None).values.T[0]

    data_and_label = {
        "train_idx"   : None,
        "train_label" : None,
        "valid_idx"   : None,
        "valid_label" : None,
        "test_idx"    : None,
        "test_label"  : None,
    }

    data_and_label['train_idx'] = train_idx.astype('int32')
    data_and_label['train_label'] = label.astype('float32')[data_and_label['train_idx']]
    output_fname = f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_products_data_and_label.pkl"

    print('Writing split files for wholegraph...')
    with open(output_fname, "wb") as f:
        pickle.dump(data_and_label, f)
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_products_edge_index_nt_et_nt_part_0_of_1')
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_products_node_feat_nt_part_0_of_1')

    train_edge_idx_list = (torch.randperm(3228124712, dtype=torch.int64, device="cpu"))
    torch.save(train_edge_idx_list, f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/global_train_edge_idx_rand")

def soft_link_graph_topo_gnnlab():
    indptr = np.memmap(f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_row_ptr", dtype='uint64', mode='r')
    indptr.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/indptr.bin')
    os.system(f'ln -s {os.path.relpath(WHOLEGRAPH_OUTPUT_DATA_DIR, GNNLAB_OUTPUT_DATA_DIR)}/homograph_csr_col_idx {GNNLAB_OUTPUT_DATA_DIR}/indices.bin')

def gen_undir_graph_wholegraph_cpu():
    print('Reading raw graph topo...')
    edges = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/edge.csv.gz', compression='gzip', header=None).values.T
    num_nodes = 2449029

    src = edges[0]
    dst = edges[1]
    data = np.zeros(src.shape)
    coo = coo_matrix((data, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint32)
    csr = coo.tocsr()
    del coo, src, dst, edges

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
        f.write('{}\t{}\n'.format('NUM_NODE', 2449029))
        f.write('{}\t{}\n'.format('NUM_EDGE', 123718152))
        f.write('{}\t{}\n'.format('FEAT_DIM', 100))
        f.write('{}\t{}\n'.format('NUM_CLASS', 47))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 196615))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 39323))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 2213091))
def write_wholegraph_meta():
    print('Writing meta file...')
    meta = {
        "nodes": [{
            "name": "nt",
            "has_emb": True,
            "emb_file_prefix": "ogbn_products_node_feat_nt",
            "num_nodes": 2449029,
            "emb_dim": 100,
            "dtype": "float32"
        }],
        "edges": [{
            "src": "nt",
            "dst": "nt",
            "rel": "et",
            "has_emb": False,
            "edge_list_prefix": "ogbn_products_edge_index_nt_et_nt",
            "num_edges": 123718152,
            "dtype": "int32",
            "directed": True
        }]
    }
    with open(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/ogbn_products_meta.json', 'w') as f:
        json.dump(meta, f)

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {PRODUCTS_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {GNNLAB_OUTPUT_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {WHOLEGRAPH_OUTPUT_DATA_DIR}') == 0)

    download_data()
    write_split_feat_label_gnnlab()
    write_split_feat_label_wholegraph()
    gen_undir_graph_wholegraph_cpu()
    soft_link_graph_topo_gnnlab()
    write_gnnlab_meta()
    write_wholegraph_meta()
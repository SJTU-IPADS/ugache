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
import pickle, json, torch

DOWNLOAD_URL = 'https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/com-friendster.tar.zst'

RAW_DATA_DIR = '/datasets_gnn/data-raw'
CF_RAW_DATA_DIR = f'{RAW_DATA_DIR}/com-friendster'

GNNLAB_OUTPUT_DATA_DIR = '/datasets_gnn/gnnlab/com-friendster'
WHOLEGRAPH_OUTPUT_DATA_DIR = '/datasets_gnn/wholegraph/com_friendster/converted'

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/com-friendster.tar.zst'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/com-friendster.tar.zst') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{CF_RAW_DATA_DIR}/unzipped'):
        print('Start unziping...')
        assert(os.system(f'cd {RAW_DATA_DIR}; tar --use-compress-program=zstd -xf {RAW_DATA_DIR}/com-friendster.tar.zst -C {CF_RAW_DATA_DIR}') == 0)
        assert(os.system(f'touch {CF_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_split_feat_label_gnnlab():
    os.system(f'cp {CF_RAW_DATA_DIR}/train_set.bin {GNNLAB_OUTPUT_DATA_DIR}/')
    os.system(f'cp {CF_RAW_DATA_DIR}/valid_set.bin {GNNLAB_OUTPUT_DATA_DIR}/')
    os.system(f'cp {CF_RAW_DATA_DIR}/test_set.bin {GNNLAB_OUTPUT_DATA_DIR}/')

    # file0 = np.load(f'{CF_RAW_DATA_DIR}/raw/data.npz')
    # file1 = np.load(f'{CF_RAW_DATA_DIR}/raw/node-label.npz')

    # features = file0['node_feat']
    # label = file1['node_label']

    # features.astype('float32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/feat.bin')
    # label.astype('uint64').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/label.bin')

def write_split_feat_label_wholegraph():
    data_and_label = {
        "train_idx"   : None,
        "train_label" : None,
        "valid_idx"   : None,
        "valid_label" : None,
        "test_idx"    : None,
        "test_label"  : None,
    }
    data_and_label['train_idx'] = np.memmap(f"{CF_RAW_DATA_DIR}/train_set.bin", dtype='uint32', mode='r')
    data_and_label['train_label'] = np.zeros_like(data_and_label['train_label'], dtype='float32')

    output_fname = f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/com_friendster_data_and_label.pkl"

    print('Writing split files for wholegraph...')
    with open(output_fname, "wb") as f:
        pickle.dump(data_and_label, f)
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/com_friendster_edge_index_paper_cites_paper_part_0_of_1')
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/com_friendster_node_feat_paper_part_0_of_1')

    train_edge_idx_list = (torch.randperm(3612134270, dtype=torch.int64, device="cpu"))
    torch.save(train_edge_idx_list, f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/global_train_edge_idx_rand")

def soft_link_graph_topo_gnnlab():
    os.system(f'cp {CF_RAW_DATA_DIR}/indptr.bin {GNNLAB_OUTPUT_DATA_DIR}/indptr.bin')
    os.system(f'ln -s {os.path.relpath(WHOLEGRAPH_OUTPUT_DATA_DIR, GNNLAB_OUTPUT_DATA_DIR)}/homograph_csr_col_idx {GNNLAB_OUTPUT_DATA_DIR}/indices.bin')

def gen_undir_graph_wholegraph_cpu():
    os.system('g++ -std=c++11 -pthread -fopenmp -O2 comfriendster_csr_generator.cc -o comfriendster_csr_generator.out')
    os.system('./comfriendster_csr_generator.out')
    indptr = np.memmap(f"{CF_RAW_DATA_DIR}/indptr.bin", dtype='uint32', mode='r')
    indptr.astype('uint64').tofile(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_row_ptr')
    os.system(f'mv {CF_RAW_DATA_DIR}/indices.bin {WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_col_idx')

def write_gnnlab_meta():
    print('Writing meta file...')
    with open(f'{GNNLAB_OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 65608366))
        f.write('{}\t{}\n'.format('NUM_EDGE', 3612134270))
        f.write('{}\t{}\n'.format('FEAT_DIM', 256))
        f.write('{}\t{}\n'.format('NUM_CLASS', 100))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 1000000))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 200000))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 100000))
def write_wholegraph_meta():
    print('Writing meta file...')
    meta = {
        "nodes": [{
            "name": "node",
            "has_emb": True,
            "emb_file_prefix": "com_friendster_node_feat_node",
            "num_nodes": 65608366,
            "emb_dim": 256,
            "dtype": "float32"
        }],
        "edges": [{
            "src": "node",
            "dst": "node",
            "rel": "edge",
            "has_emb": False,
            "edge_list_prefix": "com_friendster_edge_index_node_edge_node",
            "num_edges": 3612134270,
            "dtype": "int32",
            "directed": True
        }]
    }
    with open(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/com_friendster_meta.json', 'w') as f:
        json.dump(meta, f)

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {CF_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {GNNLAB_OUTPUT_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {WHOLEGRAPH_OUTPUT_DATA_DIR}') == 0)

    download_data()
    gen_undir_graph_wholegraph_cpu()
    soft_link_graph_topo_gnnlab()
    write_split_feat_label_gnnlab()
    write_split_feat_label_wholegraph()
    write_gnnlab_meta()
    write_wholegraph_meta()
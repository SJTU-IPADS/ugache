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
import torch

import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import pickle, json

DOWNLOAD_URL = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/mag240m_kddcup2021.zip'

RAW_DATA_DIR = '/datasets_gnn/data-raw'
MAG240M_RAW_DATA_DIR = f'{RAW_DATA_DIR}/mag240m_kddcup2021'

GNNLAB_OUTPUT_DATA_DIR = '/datasets_gnn/gnnlab/mag240m-homo'
WHOLEGRAPH_OUTPUT_DATA_DIR = '/datasets_gnn/wholegraph/mag240m_homo/converted'

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/mag240m_kddcup2021.zip'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/mag240m_kddcup2021.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{MAG240M_RAW_DATA_DIR}/unzipped'):
        print('Start unziping...')
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/mag240m_kddcup2021.zip') == 0)
        assert(os.system(f'touch {MAG240M_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_split_feat_label_gnnlab():
    print('Reading split raw data...')
    meta = torch.load(f'{MAG240M_RAW_DATA_DIR}/meta.pt')
    print(meta)
    num_paper = meta['paper']
    num_author = meta['author']
    num_institution = meta['institution']

    split_dict = torch.load(f'{MAG240M_RAW_DATA_DIR}/split_dict.pt')
    train_idx = split_dict['train']
    valid_idx = split_dict['valid']
    test_idx = split_dict['test']

    label = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/paper/node_label.npy')
    label = np.pad(label, (0, num_author + num_institution), 'constant', constant_values=(0,))

    print('Writing split files for gnnlab...')
    train_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/test_set.bin')
    label.astype('uint64').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/label.bin')

def write_split_feat_label_wholegraph():
    print('Reading split raw data...')
    split_dict = torch.load(f'{MAG240M_RAW_DATA_DIR}/split_dict.pt')
    train_idx = split_dict['train']

    paper_label = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/paper/node_label.npy')
    # file1 = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/node-label.npz')
    # label = file1['node_label']

    data_and_label = {
        "train_idx"   : None,
        "train_label" : None,
        "valid_idx"   : None,
        "valid_label" : None,
        "test_idx"    : None,
        "test_label"  : None,
    }

    data_and_label['train_idx'] = train_idx.astype('int32')
    data_and_label['train_label'] = paper_label.astype('float32')[data_and_label['train_idx']]
    output_fname = f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/mag240m_homo_data_and_label.pkl"

    print('Writing split files for wholegraph...')
    with open(output_fname, "wb") as f:
        pickle.dump(data_and_label, f)
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/mag240m_homo_edge_index_paper_cites_paper_part_0_of_1')
    os.system(f'touch {WHOLEGRAPH_OUTPUT_DATA_DIR}/mag240m_homo_node_feat_paper_part_0_of_1')

    train_edge_idx_list = (torch.randperm(3454471824, dtype=torch.int64, device="cpu"))
    torch.save(train_edge_idx_list, f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/global_train_edge_idx_rand")

def soft_link_graph_topo_gnnlab():
    indptr = np.memmap(f"{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_row_ptr", dtype='uint32', mode='r')
    indptr.astype('uint32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/indptr.bin')
    os.system(f'ln -s {os.path.relpath(WHOLEGRAPH_OUTPUT_DATA_DIR, GNNLAB_OUTPUT_DATA_DIR)}/homograph_csr_col_idx {GNNLAB_OUTPUT_DATA_DIR}/indices.bin')

def gen_undir_graph_wholegraph_cpu():
    meta = torch.load(f'{MAG240M_RAW_DATA_DIR}/meta.pt')
    print(meta)
    num_paper = meta['paper']
    num_author = meta['author']
    num_institution = meta['institution']

    print('Reading raw data...')
    paper_edge_index = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/paper___cites___paper/edge_index.npy').astype('uint32')
    write_edge_index = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/author___writes___paper/edge_index.npy').astype('uint32')
    affil_edge_index = np.load(f'{MAG240M_RAW_DATA_DIR}/processed/author___affiliated_with___institution/edge_index.npy').astype('uint32')

    p_c_p_src = paper_edge_index[0]
    p_c_p_dst = paper_edge_index[1]

    a_w_p_src = write_edge_index[0] + num_paper
    a_w_p_dst = write_edge_index[1]

    a_a_i_src = affil_edge_index[0] + num_paper
    a_a_i_dst = affil_edge_index[1] + num_paper + num_author

    total_num_nodes = num_paper + num_author + num_institution
    print('Converting topo...')
    final_coo = coo_matrix((np.ones(p_c_p_src.shape), (p_c_p_src, p_c_p_dst)), shape=(total_num_nodes, total_num_nodes), dtype=np.uint32)
    final_coo += coo_matrix((np.ones(p_c_p_src.shape), (p_c_p_dst, p_c_p_src)), shape=(total_num_nodes, total_num_nodes), dtype=np.uint32)
    final_coo += coo_matrix((np.ones(a_w_p_src.shape), (a_w_p_src, a_w_p_dst)), shape=(total_num_nodes, total_num_nodes), dtype=np.uint32)
    final_coo += coo_matrix((np.ones(a_w_p_src.shape), (a_w_p_dst, a_w_p_src)), shape=(total_num_nodes, total_num_nodes), dtype=np.uint32)
    final_coo += coo_matrix((np.ones(a_a_i_src.shape), (a_a_i_src, a_a_i_dst)), shape=(total_num_nodes, total_num_nodes), dtype=np.uint32)
    final_coo += coo_matrix((np.ones(a_a_i_src.shape), (a_a_i_dst, a_a_i_src)), shape=(total_num_nodes, total_num_nodes), dtype=np.uint32)

    csr = final_coo.tocsr()

    indptr = csr.indptr
    indices = csr.indices

    print('Writing topo...')
    indptr.astype('uint64').tofile(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_row_ptr')
    indices.astype('uint32').tofile(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/homograph_csr_col_idx')

def write_gnnlab_meta():
    print('Writing meta file...')
    with open(f'{GNNLAB_OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE',       244160499))
        f.write('{}\t{}\n'.format('NUM_EDGE',       3454471824))
        f.write('{}\t{}\n'.format('FEAT_DIM',       768))
        f.write('{}\t{}\n'.format('NUM_CLASS',      153))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET',  1112392))
        f.write('{}\t{}\n'.format('NUM_VALID_SET',  138949))
        f.write('{}\t{}\n'.format('NUM_TEST_SET',   146818))
        f.write('{}\t{}\n'.format('FEAT_DATA_TYPE', 'F16'))
def write_wholegraph_meta():
    print('Writing meta file...')
    meta = {
        "nodes": [{
            "name": "paper",
            "has_emb": True,
            "emb_file_prefix": "mag240m_homo_node_feat_paper",
            "num_nodes": 244160499,
            "emb_dim": 768,
            "dtype": "float16"
        }],
        "edges": [{
            "src": "paper",
            "dst": "paper",
            "rel": "cites",
            "has_emb": False,
            "edge_list_prefix": "mag240m_homo_edge_index_paper_cites_paper",
            "num_edges": 1727235912,
            "dtype": "int32",
            "directed": True
        }]
    }
    with open(f'{WHOLEGRAPH_OUTPUT_DATA_DIR}/mag240m_homo_meta.json', 'w') as f:
        json.dump(meta, f)

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {MAG240M_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {GNNLAB_OUTPUT_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {WHOLEGRAPH_OUTPUT_DATA_DIR}') == 0)

    download_data()
    write_split_feat_label_gnnlab()
    write_split_feat_label_wholegraph()
    gen_undir_graph_wholegraph_cpu()
    soft_link_graph_topo_gnnlab()
    write_gnnlab_meta()
    write_wholegraph_meta()

import numpy as np
import os
import time
import torch
INT_MAX = 2**31

def dataset(name, root_path, force_load64=False):
    assert(name in ['papers100M', 'com-friendster', 'reddit', 'products-undir', 'twitter', 'uk-2006-05'])
    dataset_path = os.path.join(root_path, name)
    dataset_loader = DatasetLoader(dataset_path, force_load64)
    return dataset_loader

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
    def __init__(self, dataset_path, force_load64=False):
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

        if self.num_edge >= INT_MAX:
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
        # self.eids = torch.from_numpy(
        #     np.arange(0, self.num_edge, dtype='int32'))

        self.train_set = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'train_set.bin'), dtype='int32', mode='r', shape=(self.num_train_set,)))
        self.valid_set = torch.from_numpy(np.memmap(os.path.join(dataset_path, 'valid_set.bin'), dtype='int32', mode='r', shape=(self.num_valid_set,)))
        self.test_set  = torch.from_numpy(np.memmap(os.path.join(dataset_path,  'test_set.bin'), dtype='int32', mode='r', shape=(self.num_test_set,)))

    def to_dgl_graph(self, g_format='csc'):
        import dgl

        if g_format == 'csc':
            g = dgl.graph(data = ('csc', (self.indptr, self.indices, torch.empty(0))), num_nodes = self.num_node, idtype=torch.int32)
        elif g_format == 'csr':
            g = dgl.graph(data = ('csr', (self.indptr, self.indices, torch.empty(0))), num_nodes = self.num_node)
        else:
            assert(False)

        return g
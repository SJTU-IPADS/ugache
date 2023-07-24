import dgl
from dgl.heterograph import DGLBlock, DGLHeteroGraph

def pad(target, unit):
    return ((target + unit - 1) // unit) * unit

def pad_on_demand(target, unit, demand=True):
    if not demand:
        return target
    return pad(target, unit)

def get_input_keys(graph_block):
    (target_gids, _, _, _, _, ) = graph_block
    return target_gids[0]

class BatchDataPack:
    def __init__(self):
        self.wg_graph_block = None
        self.pair_graph_num_node = None
        self.pair_graph_src = None
        self.pair_graph_dst = None
    @property
    def input_keys(self):
        (target_gids, _, _, _, _, ) = self.wg_graph_block
        return target_gids[0]
    def get_graph(self, framework, do_pad=False):
        if framework == "dgl": return self.to_dgl_graph(do_pad)
        if framework == "wg": return self.to_wg_graph()
    def to_dgl_graph(self, do_pad=False):
        (target_gids, edge_indice, _, _, _,) = self.wg_graph_block
        num_layer = len(target_gids) - 1
        sub_graphs = []
        target_gid_cnt = []
        for l in range(num_layer):
            gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, 
                pad_on_demand(target_gids[l].size(0), 8, do_pad),
                pad_on_demand(target_gids[l + 1].size(0), 8, do_pad),
                edge_indice[l][0],
                edge_indice[l][1],
                ['coo', 'csr', 'csc']
            )
            sub_graph = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
            sub_graphs.append(sub_graph)
            target_gid_cnt.append(target_gids[l + 1].numel())
        return sub_graphs, target_gid_cnt
    def to_wg_graph(self):
        (target_gids, _, csr_row_ptrs, csr_col_inds,sample_dup_counts,) = self.wg_graph_block
        num_layer = len(target_gids) - 1
        sub_graphs = []
        target_gid_cnt = []
        for l in range(num_layer):
            sub_graph = [csr_row_ptrs[l], csr_col_inds[l], sample_dup_counts[l]]
            sub_graphs.append(sub_graph)
            target_gid_cnt.append(target_gids[l + 1].numel())
        return sub_graphs, target_gid_cnt
    def to_dgl_pair_graph(self, do_pad=False):
        num_node = pad_on_demand(self.pair_graph_num_node, 8, do_pad)
        gidx = dgl.heterograph_index.create_unitgraph_from_coo(1, num_node, num_node, self.pair_graph_src, self.pair_graph_dst, ['coo', 'csr', 'csc'])
        pair_graph = DGLHeteroGraph(gidx)
        return pair_graph
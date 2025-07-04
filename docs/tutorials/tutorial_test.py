import pandas as pd
import torch

from tgm.graph import DGraph


def test_pass():
    edge_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_feat': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }  # edge events

    node_dict = {
        'node': [7, 8],
        't': [3, 6],
        'node_feat': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }  # node events, optional

    our_dgraph = DGraph(
        edge_df=pd.DataFrame(edge_dict),
        edge_src_col='src',
        edge_dst_col='dst',
        edge_time_col='t',
        edge_feats_col='edge_feat',
        node_df=pd.DataFrame(node_dict),
        node_id_col='node',
        node_time_col='t',
        dynamic_node_feats_col='node_feat',
    )

    print('=== Graph Properties ===')
    print(f'start time : {our_dgraph.start_time}')
    print(f'end time : {our_dgraph.end_time}')
    print(f'number of nodes : {our_dgraph.num_nodes}')
    print(f'number of edge events : {our_dgraph.num_edges}')
    print(f'number of timestamps : {our_dgraph.num_timestamps}')
    print(f'number of edge and node events : {our_dgraph.num_events}')
    print(f'edge feature dim : {our_dgraph.edge_feats_dim}')
    print(f'static node feature dim : {our_dgraph.static_node_feats_dim}')
    print(f'dynamic node feature dim : {our_dgraph.dynamic_node_feats_dim}')
    print('======================')


test_pass()

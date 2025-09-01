import argparse

import torch

from tgm.graph import DGData, DGraph
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TPNet Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument(
    '--time_gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)

    data = DGData.from_tgb(args.dataset)
    dgraph = DGraph(data)

    num_nodes = dgraph.num_nodes
    edge_feats_dim = dgraph.edge_feats_dim

    if dgraph.static_node_feats is not None:
        STATIC_NODE_FEAT = dgraph.static_node_feats
    else:
        STATIC_NODE_FEAT = torch.randn((num_nodes, args.node_dim), device=args.device)

    node_dim = STATIC_NODE_FEAT.shape[1]

    train_data, val_data, test_data = data.split()
    train_data.discretize(args.time_gran)
    val_data.discretize(args.time_gran)
    test_data.discretize(args.time_gran)

    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

    # hm = HookManager(hooks=['train','val','test'])

    # train_loader = DGDataLoader(train_dg,batch_size=args.bsize,hook=hm)

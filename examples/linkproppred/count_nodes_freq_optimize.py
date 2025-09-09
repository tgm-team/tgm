import argparse
import timeit

import torch
from tqdm import tqdm

from tgm.graph import DGData, DGraph
from tgm.hooks import DeduplicationHook, HookManager, RecencyNeighborHook
from tgm.loader import DGDataLoader
from tgm.nn.model.dygformer import NeighborCooccurrenceEncoder
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='DyGFormers NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--max_sequence_length',
    type=int,
    default=32,
    help='maximal length of the input sequence of each node',
)
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--time_dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed_dim', type=int, default=172, help='attention dimension')
parser.add_argument('--node_dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--channel-embedding-dim',
    type=int,
    default=50,
    help='dimension of each channel embedding',
)
parser.add_argument('--patch-size', type=int, default=1, help='patch size')
parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
parser.add_argument(
    '--num_heads', type=int, default=2, help='number of heads used in attention layer'
)
parser.add_argument(
    '--num-channels',
    type=int,
    default=4,
    help='number of channels used in attention layer',
)

parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)

parser.add_argument('--bsize', type=int, default=200, help='batch size')


args = parser.parse_args()
seed_everything(args.seed)

full_data = DGData.from_tgb(args.dataset)
full_graph = DGraph(full_data)
num_nodes = full_graph.num_nodes
edge_feats_dim = full_graph.edge_feats_dim

train_data, val_data, test_data = full_data.split()

train_data = train_data.discretize(args.time_gran)
val_data = val_data.discretize(args.time_gran)
test_data = test_data.discretize(args.time_gran)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


nbr_hook = RecencyNeighborHook(
    num_nbrs=[31],  # Keep 1 slot for seed node itself
    num_nodes=num_nodes,
    edge_feats_dim=edge_feats_dim,
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register_shared(DeduplicationHook())
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, batch_size=args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, batch_size=args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

if train_dg.static_node_feats is not None:
    STATIC_NODE_FEAT = train_dg.static_node_feats
else:
    STATIC_NODE_FEAT = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )


def _count_optimize(src: torch.Tensor, dst: torch.Tensor):
    cross_mask = src.unsqueeze(dim=1) == dst.unsqueeze(dim=-1)
    src_mask = src.unsqueeze(dim=1) == src.unsqueeze(dim=-1)
    dst_mask = dst.unsqueeze(dim=1) == dst.unsqueeze(dim=-1)

    src_freq = torch.stack([src_mask.sum(1), cross_mask.sum(1)], dim=2).float()
    dst_freq = torch.stack([dst_mask.sum(1), cross_mask.sum(2)], dim=2).float()
    src_freq[src == -1] = 0.0
    dst_freq[dst == -1] = 0.0
    return src_freq, dst_freq


neighbour_module = NeighborCooccurrenceEncoder(args.embed_dim, args.device)
old_version = []
new_version = []
with hm.activate('test'):
    for batch in tqdm(test_loader):
        batch_size = batch.src.shape[0]
        src_neighbour = batch.nbr_nids[0][:batch_size]
        dst_neighbour = batch.nbr_nids[0][batch_size : batch_size * 2]
        start = timeit.default_timer()
        source_freq_tensor_exp, dst_freq_tensor_exp = (
            neighbour_module._count_nodes_freq(
                src_neighbour.cpu().numpy(), dst_neighbour.cpu().numpy()
            )
        )
        end = timeit.default_timer()
        old_version.append(end - start)

        start = timeit.default_timer()
        source_freq_tensor, dst_freq_tensor = _count_optimize(
            src_neighbour, dst_neighbour
        )
        end = timeit.default_timer()
        new_version.append(end - start)

        assert torch.equal(source_freq_tensor_exp, source_freq_tensor)
        assert torch.equal(dst_freq_tensor_exp, dst_freq_tensor)

print('Old:', sum(old_version) / len(old_version))
print('New:', sum(new_version) / len(new_version))

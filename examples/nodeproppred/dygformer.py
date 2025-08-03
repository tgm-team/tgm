r"""python -u tgcn.py --dataset tgbn-trade --time-gran Y --batch-time-gran Y
python -u tgcn.py --dataset tgbn-genre --time-gran s --batch-time-gran D\
example commands to run this script.
"""

import argparse

import torch
from tgb.nodeproppred.evaluate import Evaluator

from tgm.graph import DGraph
from tgm.hooks import NeighborSamplerHook, RecencyNeighborHook
from tgm.loader import DGDataLoader
from tgm.nn.model import DyGFormer
from tgm.timedelta import TimeDeltaDG
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGCN Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--max_sequence_length', type=int, default=512, help='learning rate'
)
parser.add_argument('--node_dim', type=int, default=128, help='embedding dimension')


parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--batch-time-gran',
    type=str,
    default='D',
    help='time granularity to operate on for snapshots',
)


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(
    args.dataset,
    time_delta=TimeDeltaDG(args.time_gran),
    split='train',
    device=args.device,
)
val_dg = DGraph(
    args.dataset,
    time_delta=TimeDeltaDG(args.time_gran),
    split='val',
    device=args.device,
)
test_dg = DGraph(
    args.dataset,
    time_delta=TimeDeltaDG(args.time_gran),
    split='test',
    device=args.device,
)

num_nodes = DGraph(args.dataset).num_nodes
label_dim = train_dg.dynamic_node_feats_dim
evaluator = Evaluator(name=args.dataset)

train_loader = DGDataLoader(
    train_dg,
    batch_unit=args.batch_time_gran,
    hook=[
        RecencyNeighborHook(
            num_nodes=num_nodes, num_nbrs=[args.max_sequence_length], edge_feats_dim=1
        )
    ],
)
val_loader = DGDataLoader(
    val_dg, batch_unit=args.batch_time_gran, hook=NeighborSamplerHook(num_nbrs=[1])
)
test_loader = DGDataLoader(
    test_dg, batch_unit=args.batch_time_gran, hook=NeighborSamplerHook(num_nbrs=[1])
)

# TODO: add static node features to DGraph
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)

model = DyGFormer(
    node_feat_dim=args.node_dim,
    edge_feat_dim=1,
    time_feat_dim=100,
    channel_embedding_dim=50,
)

for batch in train_loader:
    # print(batch.nbr_nids.shape)
    # descriptor = "DGBatch("
    # for attr, value in batch.__dict__.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"{attr}: {value.shape}")
    print(batch.__dict__)

    break

# TODO: need to feed delta time: as TGAT do (DyGFormer):edge_time.unsqueeze(1) - src_neighbours_time
# TODO: RecencySample = max_sequence -1 (keep 1 for target node itself)

# for epoch in range(1, args.epochs + 1):
#     start_time = time.perf_counter()
#     loss, h_0 = train(train_loader, model, opt, static_node_feats)
#     end_time = time.perf_counter()
#     latency = end_time - start_time

#     val_results, h_0 = eval(val_loader, model, static_node_feats, h_0)
#     print(
#         f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
#         + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
#     )

# test_results, h_0 = eval(test_loader, model, static_node_feats, h_0)
# print('Test:', ' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

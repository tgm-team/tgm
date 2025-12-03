import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.nn import EvolveGCNH, EvolveGCNO, NodePredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser(
    description='EvolveGCN NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--embed-dim', type=int, default=256, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='Y',
    help='time granularity to operate on for snapshots',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)
parser.add_argument(
    '--mode',
    type=str,
    default='o',
    choices=['o', 'h'],
    help='To use EvolveGCNO or EvolveGCNH',
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class RecurrentGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        mode: str,
        hidden_dim: int,
        num_nodes: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ['o', 'h']:
            raise ValueError(
                'EvolveGCN mode must be either o or h , currently received,', mode
            )
        if mode == 'o':
            self.recurrent = EvolveGCNO(
                in_channels=in_channels,
                improved=improved,
                cached=cached,
                normalize=normalize,
                add_self_loops=add_self_loops,
            )

        if mode == 'h':
            self.recurrent = EvolveGCNH(
                num_nodes=num_nodes,
                in_channels=in_channels,
                improved=improved,
                cached=cached,
                normalize=normalize,
                add_self_loops=add_self_loops,
            )
        self.linear = torch.nn.Linear(in_channels, hidden_dim)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.tensor,
    ) -> torch.Tensor:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = torch.ones(edge_index.size(1), 1).to(args.device)
        h = self.recurrent(node_feat.to(torch.float), edge_index, edge_weight)
        z = F.relu(h)
        z = self.linear(z)
        return z


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()

    opt.zero_grad()
    loss = 0
    for batch in tqdm(loader):
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z = encoder(batch, static_node_feats)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        loss += F.cross_entropy(y_pred, y_true)
    loss.backward()
    opt.step()

    return loss.item()


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z = encoder(batch, static_node_feats)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        input_dict = {
            'y_true': y_true,
            'y_pred': y_pred,
            'eval_metric': [METRIC_TGB_NODEPROPPRED],
        }
        perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

    encoder.recurrent.weight = encoder.recurrent.weight.detach()

    return float(np.mean(perf_list))


seed_everything(args.seed)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(train_dg, batch_unit=args.snapshot_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.snapshot_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.snapshot_time_gran)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

evaluator = Evaluator(name=args.dataset)
num_classes = train_dg.dynamic_node_feats_dim

encoder = RecurrentGCN(
    in_channels=static_node_feats.shape[1],
    mode=args.mode,
    hidden_dim=args.embed_dim,
    num_nodes=static_node_feats.shape[0],  # required for EvolveGCNH
).to(args.device)
decoder = NodePredictor(
    in_dim=args.embed_dim, out_dim=num_classes, hidden_dim=args.embed_dim
).to(args.device)

opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

best_ndcg = 0

for epoch in range(1, args.epochs + 1):
    loss = train(train_loader, static_node_feats, encoder, decoder, opt)
    val_ndcg = eval(val_loader, static_node_feats, encoder, decoder, evaluator)
    if val_ndcg > best_ndcg:
        best_ndcg = val_ndcg
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_ndcg, epoch=epoch)
print(f'Best {METRIC_TGB_NODEPROPPRED} on validation set: {best_ndcg}')
test_ndcg = eval(test_loader, static_node_feats, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_ndcg, epoch=args.epochs)

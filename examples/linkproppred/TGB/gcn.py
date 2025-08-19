r"""python -u gcn.py --dataset tgbl-wiki --time-gran s --snapshot-time-gran h
example commands to run this script.
"""

import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import NegativeEdgeSamplerHook, TGBNegativeEdgeSamplerHook
from tgm.loader import DGDataLoader
from tgm.timedelta import TimeDeltaDG
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN TGB Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
parser.add_argument(
    '--bsize', type=int, default=200, help='batch size for TGN CTDG iteration'
)
parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='h',
    help='time granularity to operate on for snapshots',
)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = GCNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_channels=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self, batch: DGBatch, node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        z = self.encoder(node_feat, edge_index)
        return z


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid().view(-1)  # Ensure output is a 1D tensor


def train_in_batches(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    static_node_feat: torch.Tensor,
    conversion_rate: int,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    embeddings = encoder(snapshot_batch, static_node_feat)
    for batch in tqdm(loader):
        opt.zero_grad()
        embeddings = encoder(snapshot_batch, static_node_feat)
        pos_out = decoder(embeddings[batch.src], embeddings[batch.dst])
        neg_out = decoder(embeddings[batch.src], embeddings[batch.neg])
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        total_loss += float(loss) / batch.src.shape[0]
        loss.backward()
        opt.step()

        # update the model if the prediction batch has moved to next snapshot.
        while (
            batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate
        ):  # if batch timestamps greater than snapshot, process the snapshot
            try:
                snapshot_batch = next(snapshots_iterator)
            except StopIteration:
                pass
    return total_loss, embeddings.detach()


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    eval_metric: str,
    evaluator: Evaluator,
    static_node_feat: torch.Tensor,
    z: torch.Tensor,
    conversion_rate: int,
) -> dict:
    encoder.eval()
    decoder.eval()
    perf_list = []
    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    for batch in tqdm(loader):
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.tensor(
                [batch.src[idx] for _ in range(len(neg_batch) + 1)]
            )
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            y_pred = decoder(z[query_src], z[query_dst])

            # compute MRR
            input_dict = {
                'y_pred_pos': y_pred[0].detach().cpu().numpy(),
                'y_pred_neg': y_pred[1:].detach().cpu().numpy(),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])

        # update the model if the prediction batch has moved to next snapshot.
        while (
            batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate
        ):  # if batch timestamps greater than snapshot, process the snapshot
            z = encoder(snapshot_batch, static_node_feat).detach()
            try:
                snapshot_batch = next(snapshots_iterator)
            except StopIteration:
                pass

    metric_dict = {}
    metric_dict[eval_metric] = float(np.mean(perf_list))
    return metric_dict


args = parser.parse_args()
seed_everything(args.seed)
snapshot_td = TimeDeltaDG(args.snapshot_time_gran, 1)
tgb_td = TimeDeltaDG(args.time_gran, 1)
conversion_rate = int(snapshot_td.convert(tgb_td))

dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

train_data = DGData.from_tgb(args.dataset, split='train')
train_data_discretized = train_data.discretize(args.snapshot_time_gran)
train_dg = DGraph(train_data, train_data.time_delta, device=args.device)
train_snapshots = DGraph(
    train_data_discretized, train_data_discretized.time_delta, device=args.device
)

val_data = DGData.from_tgb(args.dataset, split='val')
val_data_discretized = val_data.discretize(args.snapshot_time_gran)
val_dg = DGraph(val_data, val_data.time_delta, device=args.device)
val_snapshots = DGraph(
    val_data_discretized, val_data_discretized.time_delta, device=args.device
)

test_data = DGData.from_tgb(args.dataset, split='test')
test_data_discretized = test_data.discretize(args.snapshot_time_gran)
test_dg = DGraph(test_data, test_data.time_delta, device=args.device)
test_snapshots = DGraph(
    test_data_discretized, test_data_discretized.time_delta, device=args.device
)

train_loader = DGDataLoader(
    train_dg,
    hook=[NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes)],
    batch_size=args.bsize,
)
train_snapshots_loader = DGDataLoader(
    train_snapshots,
    hook=[NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes)],
    batch_unit=args.snapshot_time_gran,
)

val_loader = DGDataLoader(
    val_dg,
    hook=[TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val')],
    batch_size=args.bsize,
)
val_snapshots_loader = DGDataLoader(val_snapshots, batch_unit=args.snapshot_time_gran)

test_loader = DGDataLoader(
    test_dg,
    hook=[TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='test')],
    batch_size=args.bsize,
)
test_snapshots_loader = DGDataLoader(test_snapshots, batch_unit=args.snapshot_time_gran)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

node_dim = static_node_feats.shape[1]
encoder = GCN(
    in_channels=node_dim,
    embed_dim=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
).to(args.device)

decoder = LinkPredictor(
    args.embed_dim,
).to(args.device)

opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss, z = train_in_batches(
        train_loader,
        train_snapshots_loader,
        encoder,
        decoder,
        opt,
        static_node_feats,
        conversion_rate,
    )
    end_time = time.perf_counter()
    latency = end_time - start_time
    print(f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f}')

    val_results = eval(
        val_loader,
        val_snapshots_loader,
        encoder,
        decoder,
        eval_metric,
        evaluator,
        static_node_feats,
        z,
        conversion_rate,
    )
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(
    test_loader,
    test_snapshots_loader,
    encoder,
    decoder,
    eval_metric,
    evaluator,
    static_node_feats,
    z,
    conversion_rate,
)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

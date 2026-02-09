import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGraph, TimeDeltaDG
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecipeRegistry
from tgm.nn import LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='h',
    help='time granularity to operate on for snapshots',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


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

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_index = torch.stack([batch.edge_src, batch.edge_dst], dim=0)
        x = node_feat
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    conversion_rate: int,
) -> Tuple[float, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_x = loader.dgraph.static_node_x

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    z = encoder(snapshot_batch, static_node_x)
    z = z.detach()

    for batch in tqdm(loader):
        opt.zero_grad()

        pos_out = decoder(z[batch.edge_src], z[batch.edge_dst])
        neg_out = decoder(z[batch.edge_src], z[batch.neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss) / batch.edge_src.shape[0]

        # update the model if the prediction batch has moved to next snapshot.
        while (
            batch.edge_time[-1] > (snapshot_batch.edge_time[-1] + 1) * conversion_rate
        ):
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_x)
                z = z.detach()
            except StopIteration:
                pass

    return total_loss, z


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    z: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
    conversion_rate: int,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)

    for batch in tqdm(loader):
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = batch.edge_src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.edge_dst[idx].unsqueeze(0), neg_batch])

            y_pred = decoder(z[query_src], z[query_dst]).sigmoid()
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        # update the model if the prediction batch has moved to next snapshot.
        while (
            batch.edge_time[-1] > (snapshot_batch.edge_time[-1] + 1) * conversion_rate
        ):
            try:
                snapshot_batch = next(snapshots_iterator)
                z = encoder(snapshot_batch, static_node_x)
            except StopIteration:
                pass

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
if full_data.static_node_x is None:
    full_data.static_node_x = torch.randn(
        (full_data.num_nodes, args.node_dim), device=args.device
    )

train_data, val_data, test_data = full_data.split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

snapshot_td = TimeDeltaDG(args.snapshot_time_gran)
conversion_rate = int(snapshot_td.convert(train_dg.time_delta))

train_data_discretized = train_data.discretize(args.snapshot_time_gran)
val_data_discretized = val_data.discretize(args.snapshot_time_gran)
test_data_discretized = test_data.discretize(args.snapshot_time_gran)

train_snapshots = DGraph(train_data_discretized, device=args.device)
val_snapshots = DGraph(val_data_discretized, device=args.device)
test_snapshots = DGraph(test_data_discretized, device=args.device)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

train_snapshots_loader = DGDataLoader(
    train_snapshots, batch_unit=args.snapshot_time_gran
)
val_snapshots_loader = DGDataLoader(val_snapshots, batch_unit=args.snapshot_time_gran)
test_snapshots_loader = DGDataLoader(test_snapshots, batch_unit=args.snapshot_time_gran)


encoder = GCNEncoder(
    in_channels=train_dg.static_node_x_dim,
    embed_dim=args.embed_dim,
    out_channels=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
).to(args.device)
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss, z = train(
            train_loader,
            train_snapshots_loader,
            encoder,
            decoder,
            opt,
            conversion_rate,
        )

    with hm.activate(val_key):
        val_mrr = eval(
            val_loader,
            val_snapshots_loader,
            z,
            encoder,
            decoder,
            evaluator,
            conversion_rate,
        )
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)


with hm.activate(test_key):
    test_mrr = eval(
        test_loader,
        test_snapshots_loader,
        z,
        encoder,
        decoder,
        evaluator,
        conversion_rate,
    )
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

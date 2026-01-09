import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.nn import NodePredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='Y',
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

        self.convs.append(GCNConv(in_channels, embed_dim, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels, cached=True))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
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
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_feats = loader.dgraph.static_node_feats

    for batch in tqdm(loader):
        opt.zero_grad()
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z = encoder(batch, static_node_feats)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        opt.step()
        total_loss += float(loss)

    return total_loss


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_feats = loader.dgraph.static_node_feats

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

train_loader = DGDataLoader(train_dg, batch_unit=args.snapshot_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.snapshot_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.snapshot_time_gran)

num_classes = train_dg.dynamic_node_feats_dim

encoder = GCNEncoder(
    in_channels=train_dg.static_node_feats_dim,
    embed_dim=args.embed_dim,
    out_channels=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
).to(args.device)
decoder = NodePredictor(
    in_dim=args.embed_dim, out_dim=num_classes, hidden_dim=args.embed_dim
).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    loss = train(train_loader, encoder, decoder, opt)
    val_ndcg = eval(val_loader, encoder, decoder, evaluator)
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_ndcg, epoch=epoch)

test_ndcg = eval(test_loader, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_ndcg, epoch=args.epochs)

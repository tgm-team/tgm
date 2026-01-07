import argparse
import pathlib
from typing import Callable, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader, TemporalRatioSplit
from tgm.nn import GraphPredictor
from tgm.util.logging import (
    enable_logging,
    log_gpu,
    log_latency,
    log_metric,
    log_metrics_dict,
)
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN GraphPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

"""
Adapted graph property prediction as proposed in https://openreview.net/forum?id=DZqic2sPTY

Note:
 - `lag` is excluded from this example's setting.
 - Graph property prediction is always DTDG setting (snapshot-based)
"""

parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--train-ratio', type=float, default=0.7, help='train ratio')
parser.add_argument('--val-ratio', type=float, default=0.15, help='validation ratio')
parser.add_argument('--test-ratio', type=float, default=0.15, help='test ratio')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of TGCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='examples/graphproppred/tokens_data/test-token.csv',
    help=(
        'Either Path to dataset csv file. '
        'You can run `./scripts/download_graph_prop_datasets.sh examples/graphproppred` '
        'to download the relevant token data to the default path.'
        'Or TGB dataset'
    ),
)
parser.add_argument(
    '--raw-time-gran', type=str, default='s', help='raw time granularity for dataset'
)
parser.add_argument(
    '--batch-time-gran',
    type=str,
    default='W',
    help='time granularity to operate on for snapshots',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


# ETL steps
def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    # time offset
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    # normalize edge weight
    max_weight = float(df['value'].max())
    min_weight = float(df['value'].min())
    df['value'] = df['value'].apply(
        lambda x: [1 + (9 * ((float(x) - min_weight) / (max_weight - min_weight)))]
    )

    # Key generator
    node_id_map = {}
    df['from'] = df['from'].apply(lambda x: node_id_map.setdefault(x, len(node_id_map)))
    df['to'] = df['to'].apply(lambda x: node_id_map.setdefault(x, len(node_id_map)))
    return df


def load_data(dataset_str: str) -> Tuple[DGData, TemporalRatioSplit | None]:
    r"""Load full dataset and its corresponding split."""
    if pathlib.Path(dataset_str).exists() and dataset_str.endswith('.csv'):
        df = pd.read_csv(dataset_str)
        df = preprocess_raw_data(df)
        full_data = DGData.from_pandas(
            edge_df=df,
            edge_src_col='from',
            edge_dst_col='to',
            edge_time_col='timestamp',
            edge_feats_col='value',
            time_delta=args.raw_time_gran,
        )
        split_strategy = TemporalRatioSplit(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        return full_data, split_strategy
    elif dataset_str.startswith('tgb'):
        full_data = DGData.from_tgb(args.dataset).discretize(args.batch_time_gran)
        split_strategy = None  # For TGB dataset, will use TGB split by default
    else:
        raise ValueError('This example only supports TGB and test tokens from MiNT.')
    return full_data, split_strategy


def edge_count(snapshot: DGBatch):  # return number of edges of current snapshot
    return snapshot.src.shape[0]


def node_count(snapshot: DGBatch):  # return number of nodes of current snapshot
    return len(snapshot.unique_nids)


def generate_binary_trend_labels(
    loader: DGDataLoader, snapshot_measurement: Callable = edge_count
) -> torch.Tensor:
    labels = []
    prev_snapshot_metric = -1
    for idx, snapshot in enumerate(loader):
        if idx == 0:
            prev_snapshot_metric = snapshot_measurement(snapshot)
        else:
            curr_snapshot_metric = snapshot_measurement(snapshot)
            label = 1 if curr_snapshot_metric > prev_snapshot_metric else 0
            labels.append(label)
            prev_snapshot_metric = curr_snapshot_metric
    return torch.tensor(labels, dtype=torch.int64)


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
    labels: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    metrics: Metric,
) -> Tuple[float, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_feats = loader.dgraph.static_node_feats

    y_pred = torch.zeros_like(labels, dtype=torch.float)
    for i, batch in enumerate(tqdm(loader)):
        if i != len(loader) - 1:  # Skip last snapshot as we don't have labels for it
            opt.zero_grad()
            z = encoder(batch, static_node_feats)
            z_node = z[torch.cat([batch.src, batch.dst])]
            pred = decoder(z_node)

            loss = F.binary_cross_entropy_with_logits(
                pred, labels[i].unsqueeze(0).float()
            )
            loss.backward()
            opt.step()

            total_loss += float(loss)
            y_pred[i] = pred

    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
    metrics(y_pred, labels, indexes=indexes)
    metrics_dict = {k: v.item() for k, v in metrics.compute().items()}
    return total_loss, metrics_dict


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    y_true: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    metrics: Metric,
) -> dict:
    encoder.eval()
    decoder.eval()
    y_pred = torch.zeros_like(y_true, dtype=torch.float)
    static_node_feats = loader.dgraph.static_node_feats

    for i, batch in enumerate(tqdm(loader)):
        if i != len(loader) - 1:  # Skip last snapshot as we don't have labels for it
            z = encoder(batch, static_node_feats)
            z_node = z[torch.cat([batch.src, batch.dst])]
            y_pred[i] = decoder(z_node).sigmoid()

    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
    metrics(y_pred, y_true, indexes=indexes)
    metrics_dict = {k: v.item() for k, v in metrics.compute().items()}
    return metrics_dict


seed_everything(args.seed)

full_data, split_strategy = load_data(args.dataset)
full_data = full_data.discretize(
    args.batch_time_gran
)  # discretize to adapt to graphproppred setting

if full_data.static_node_feats is None:
    full_data.static_node_feats = torch.randn(
        (full_data.num_nodes, args.node_dim), device=args.device
    )

train_data, val_data, test_data = full_data.split(split_strategy)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(train_dg, batch_unit=args.batch_time_gran, on_empty='raise')
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran, on_empty='raise')
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran, on_empty='raise')

encoder = GCNEncoder(
    in_channels=train_dg.static_node_feats_dim,
    embed_dim=args.embed_dim,
    out_channels=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
).to(args.device)
decoder = GraphPredictor(in_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
train_metrics = MetricCollection(metrics, prefix='Train')
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

train_labels = generate_binary_trend_labels(
    train_loader, snapshot_measurement=edge_count
).to(args.device)
val_labels = generate_binary_trend_labels(
    val_loader, snapshot_measurement=edge_count
).to(args.device)
test_labels = generate_binary_trend_labels(
    test_loader, snapshot_measurement=edge_count
).to(args.device)

for epoch in range(1, args.epochs + 1):
    loss, train_results = train(
        train_loader, train_labels, encoder, decoder, opt, train_metrics
    )

    val_results = eval(val_loader, val_labels, encoder, decoder, val_metrics)

    log_metric('Loss', loss, epoch=epoch)
    log_metrics_dict(train_results, epoch=epoch)
    log_metrics_dict(val_results, epoch=epoch)

test_results = eval(test_loader, test_labels, encoder, decoder, test_metrics)
log_metrics_dict(test_results, epoch=args.epochs)

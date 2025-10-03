import argparse
from typing import Callable, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.loader import DGDataLoader
from tgm.nn import TGCN
from tgm.split import TemporalRatioSplit
from tgm.util.logging import (
    enable_logging,
    log_gpu,
    log_latency,
    log_metric,
    log_metrics_dict,
)
from tgm.util.seed import seed_everything

"""
Adapted graph property prediction as proposed in https://openreview.net/forum?id=DZqic2sPTY

Note: `lag` is excluded from this example's setting.
"""

parser = argparse.ArgumentParser(
    description='TGCN Graph Property Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
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
    '--path-dataset',
    type=str,
    default='examples/graphproppred/tokens_data/test-token.csv',
    help=(
        'Path to dataset csv file. '
        'You can run `./scripts/download_graph_prop_datasets.sh examples/graphproppred` '
        'to download the relevant token data to the default path.'
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


# ETL step
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


def sum_pooling(z: torch.Tensor) -> torch.Tensor:
    return torch.sum(z, dim=0).squeeze()


def mean_pooling(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z, dim=0).squeeze()


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.tensor,
        h: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None  # type: ignore

        h_0 = self.recurrent(node_feat, edge_index, edge_weight, h)
        z = F.relu(h_0)
        z = self.linear(z)
        return z, h_0


class GraphPredictor(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, graph_pooling: Callable = mean_pooling
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.graph_pooling = graph_pooling

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        z_graph = self.graph_pooling(z_node)
        h = self.fc1(z_graph)
        h = h.relu()
        return self.fc2(h)


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    labels: torch.Tensor,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    metrics: Metric,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0
    h_0 = None

    y_pred = torch.zeros_like(labels, dtype=torch.float)
    for i, batch in enumerate(tqdm(loader)):
        if i != len(loader) - 1:  # Skip last snapshot as we don't have labels for it
            opt.zero_grad()
            z, h_0 = encoder(batch, static_node_feats, h_0)
            z_node = z[torch.cat([batch.src, batch.dst])]
            pred = decoder(z_node)

            loss = F.binary_cross_entropy_with_logits(
                pred, labels[i].unsqueeze(0).float()
            )
            loss.backward()
            opt.step()

            total_loss += float(loss)
            y_pred[i] = pred
            h_0 = h_0.detach()

    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
    metrics(y_pred, labels, indexes=indexes)
    return total_loss, h_0, metrics.compute()


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    y_true: torch.Tensor,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    h_0: torch.Tensor,
    metrics: Metric,
) -> Tuple[dict, torch.Tensor]:
    encoder.eval()
    decoder.eval()
    y_pred = torch.zeros_like(y_true, dtype=torch.float)

    for i, batch in enumerate(tqdm(loader)):
        if i != len(loader) - 1:  # Skip last snapshot as we don't have labels for it
            z, h_0 = encoder(batch, static_node_feats, h_0)
            z_node = z[torch.cat([batch.src, batch.dst])]
            y_pred[i] = decoder(z_node).sigmoid()

    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
    metrics(y_pred, y_true, indexes=indexes)
    return metrics.compute(), h_0


seed_everything(args.seed)

df = pd.read_csv(args.path_dataset)
df = preprocess_raw_data(df)

full_data = DGData.from_pandas(
    edge_df=df,
    edge_src_col='from',
    edge_dst_col='to',
    edge_time_col='timestamp',
    edge_feats_col='value',
    time_delta=args.raw_time_gran,
).discretize(args.batch_time_gran)

train_data, val_data, test_data = full_data.split(
    TemporalRatioSplit(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(train_dg, batch_unit=args.batch_time_gran, on_empty='raise')
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran, on_empty='raise')
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran, on_empty='raise')

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

encoder = RecurrentGCN(
    node_dim=static_node_feats.shape[1], embed_dim=args.embed_dim
).to(args.device)
decoder = GraphPredictor(in_dim=args.embed_dim, out_dim=1).to(args.device)
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
    loss, h_0, train_results = train(
        train_loader,
        train_labels,
        static_node_feats,
        encoder,
        decoder,
        opt,
        train_metrics,
    )

    val_results, h_0 = eval(
        val_loader, val_labels, static_node_feats, encoder, decoder, h_0, val_metrics
    )
    log_metric('Loss', loss, epoch=epoch)
    log_metrics_dict(train_results, epoch=epoch)
    log_metrics_dict(val_results, epoch=epoch)

test_results, h_0 = eval(
    test_loader, test_labels, static_node_feats, encoder, decoder, h_0, test_metrics
)
log_metrics_dict(test_results, epoch=args.epochs)

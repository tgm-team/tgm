import argparse
import time
from typing import Callable, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import DeduplicationHook, HookManager
from tgm.loader import DGDataLoader
from tgm.nn.recurrent import TGCN
from tgm.split import TemporalRatioSplit
from tgm.util.seed import seed_everything

r"""
Adapted the setting for graph property prediction proposed in
https://openreview.net/forum?id=DZqic2sPTY

This example can be run with any token networks provided in
https://arxiv.org/abs/2406.10426
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
parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.00015, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of TGCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
parser.add_argument(
    '--path-dataset',
    type=str,
    default='examples/graphproppred/tokens_data/test_token.csv',
    help='Path to dataset csv file',
)

parser.add_argument(
    '--raw-time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--batch-time-gran',
    type=str,
    default='W',
    help='time granularity to operate on for snapshots',
)


def edge_count(snapshot: DGBatch):
    # return number of edges of current snapshot
    return snapshot.src.shape[0]


def node_count(snapshot: DGBatch):
    # return number of nodes of current snapshot
    return len(snapshot.unique_nids)


def label_generator_next_binary_classification(
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
def preproccess_raw_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    # time offset
    dataframe['timestamp'] = dataframe['timestamp'] - dataframe['timestamp'].min()

    # normalize edge weight
    max_weight = float(dataframe['value'].max())
    min_weight = float(dataframe['value'].min())
    dataframe['value'] = dataframe['value'].apply(
        lambda x: [1 + (9 * ((float(x) - min_weight) / (max_weight - min_weight)))]
    )

    # Key generator
    node_id_map = {}
    dataframe['from'] = dataframe['from'].apply(
        lambda x: node_id_map.setdefault(x, len(node_id_map))
    )
    dataframe['to'] = dataframe['to'].apply(
        lambda x: node_id_map.setdefault(x, len(node_id_map))
    )

    return dataframe


# graph pooling
def sum_pooling(z: torch.Tensor) -> torch.Tensor:
    return torch.sum(z, dim=0).squeeze()


def mean_pooling(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z, dim=0).squeeze()


class TGCN_Model(nn.Module):
    def __init__(
        self,
        node_dim: int,
        embed_dim: int,
        num_classes: int,
        graph_pooling: Callable = mean_pooling,
    ) -> None:
        super().__init__()
        self.encoder = RecurrentGCN(node_dim=node_dim, embed_dim=embed_dim)
        self.graph_pooling = graph_pooling
        self.decoder = GraphPredictor(in_dim=embed_dim, out_dim=num_classes)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.Tensor,
        h_0: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None  # type: ignore
        z, h_0 = self.encoder(node_feat, edge_index, edge_weight, h_0)

        z_node = z[batch.global_to_local(batch.unique_nids)]  # type: ignore
        z_graph = self.graph_pooling(z_node)
        pred = self.decoder(z_graph)
        return pred, h_0


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        h: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, ...]:
        h_0 = self.recurrent(x, edge_index, edge_weight, h)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0


class GraphPredictor(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_node = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        return h.sigmoid()


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    y_true: torch.Tensor,
    static_node_feats: torch.Tensor,
    model: torch.nn.Module,
    h_0: torch.Tensor,
    metrics: Metric,
    ignore_last_snapshot=False,
) -> dict:
    all_preds = []
    number_of_snapshot = len(loader)
    for idx, snapshot in enumerate(tqdm(loader)):
        if not (ignore_last_snapshot and idx == number_of_snapshot - 1):
            pred, h_0 = model(snapshot, static_node_feats, h_0)
            all_preds.append(pred)
    y_pred = torch.Tensor(all_preds).float()
    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)

    metrics(y_pred, y_true, indexes=indexes)
    return metrics.compute(), h_0


def train(
    loader: DGDataLoader,
    labels: torch.Tensor,
    node_feat: torch.Tensor,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    metrics: Metric,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.train()
    total_loss = 0
    h_0 = None
    criterion = torch.nn.BCELoss()
    all_preds = []
    for idx, snapshot in enumerate(tqdm(loader)):
        opt.zero_grad()
        pred, h_0 = model(snapshot, node_feat, h_0)
        loss = criterion(pred.float(), labels[idx].unsqueeze(0).float())
        all_preds.append(pred)
        loss.backward()
        opt.step()
        total_loss += float(loss)
        h_0 = h_0.detach()
    y_pred = torch.Tensor(all_preds).float()
    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)

    metrics(y_pred, labels, indexes=indexes)
    return total_loss, h_0, metrics.compute()


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    metrics = [BinaryAveragePrecision(), BinaryAUROC()]
    train_metrics = MetricCollection(metrics, prefix='Train')
    val_metrics = MetricCollection(metrics, prefix='Validation')
    test_metrics = MetricCollection(metrics, prefix='Test')

    token = pd.read_csv(args.path_dataset)
    token = preproccess_raw_data(token)

    full_data = DGData.from_pandas(
        edge_df=token,
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

    full_dg = DGraph(full_data, device=args.device)
    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

    full_loader = DGDataLoader(
        full_dg, batch_unit=args.batch_time_gran, on_empty='raise'
    )

    hm = HookManager(keys=['global'])
    hm.register('global', DeduplicationHook())
    train_loader = DGDataLoader(
        train_dg, batch_unit=args.batch_time_gran, on_empty='raise', hook_manager=hm
    )
    val_loader = DGDataLoader(
        val_dg, batch_unit=args.batch_time_gran, on_empty='raise', hook_manager=hm
    )
    test_loader = DGDataLoader(
        test_dg, batch_unit=args.batch_time_gran, on_empty='raise', hook_manager=hm
    )

    num_train_snapshots = len(train_loader)
    num_val_snapshots = len(val_loader)
    num_test_snapshots = len(test_loader) - 1

    labels = label_generator_next_binary_classification(
        loader=full_loader, snapshot_measurement=edge_count
    )  # shape: number of snapshots - 1

    train_labels = labels[:num_train_snapshots]
    val_labels = labels[num_train_snapshots : num_train_snapshots + num_val_snapshots]
    test_labels = labels[-num_test_snapshots:]

    if train_dg.static_node_feats is not None:
        static_node_feats = train_dg.static_node_feats
    else:
        static_node_feats = torch.randn(
            (test_dg.num_nodes, args.node_dim), device=args.device
        )
    model = TGCN_Model(
        node_dim=static_node_feats.shape[1], embed_dim=args.embed_dim, num_classes=1
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    with hm.activate('global'):
        for epoch in range(1, args.epochs + 1):
            start_time = time.perf_counter()
            loss, h_0, train_results = train(
                train_loader, train_labels, static_node_feats, model, opt, train_metrics
            )
            end_time = time.perf_counter()
            latency = end_time - start_time

            val_results, h_0 = eval(
                val_loader,
                val_labels,
                static_node_feats,
                model,
                h_0,
                val_metrics,
                ignore_last_snapshot=False,
            )
            print(
                f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
                + ' '.join(f'{k}={v:.4f}' for k, v in train_results.items())
                + ' '
                + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
            )

        test_results, h_0 = eval(
            test_loader,
            test_labels,
            static_node_feats,
            model,
            h_0,
            test_metrics,
            ignore_last_snapshot=True,
        )
        print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

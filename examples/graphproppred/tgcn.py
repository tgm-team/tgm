import argparse
from typing import Callable, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator

from tgm import DGBatch, DGraph
from tgm.loader import DGDataLoader
from tgm.nn.recurrent import TGCN

parser = argparse.ArgumentParser(
    description='TGCN Graph Property Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of TGCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--time-gran',
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
    # return number of ndoes of current snapshot
    return torch.unique(torch.cat([snapshot.src, snapshot.dst])).numel()


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
    start_time = dataframe['timestamp'].min()
    dataframe['timestamp'] = dataframe['timestamp'].apply(lambda x: x - start_time)

    # normalize edge weight
    max_weight = float(dataframe['value'].max())
    min_weight = float(dataframe['value'].min())
    dataframe['value'] = dataframe['value'].apply(
        lambda x: 1 + (9 * ((float(x) - min_weight) / (max_weight - min_weight)))
    )

    # Key generator
    node_id_map = {}
    dataframe['from'] = dataframe['from'].apply(
        lambda x: node_id_map.setdefault(x, len(node_id_map))
    )
    dataframe['to'] = dataframe['to'].apply(
        lambda x: node_id_map.setdefault(x, len(node_id_map))
    )

    dataframe['value'] = dataframe['value'].apply(lambda x: [x])

    return dataframe


# graph pooling
def sum_pooling(z: torch.Tensor) -> torch.Tensor:
    return torch.sum(z, dim=0)[0].squeeze()


def mean_pooling(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z, dim=0)[0].squeeze()


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
        z_node = z[batch.global_to_local(batch.node_ids)]  # type: ignore
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


if __name__ == '__main__':
    args = parser.parse_args()
    train_dg = DGraph(args.dataset, time_delta='s', split='train', device=args.device)
    train_dg = train_dg.discretize(args.time_gran)

    val_dg = DGraph(args.dataset, time_delta='s', split='val', device=args.device)
    val_dg = val_dg.discretize(args.time_gran)

    test_dg = DGraph(args.dataset, time_delta='s', split='test', device=args.device)
    test_dg = test_dg.discretize(args.time_gran)

    dgraph = DGraph(args.dataset)
    num_nodes = dgraph.num_nodes
    label_dim = train_dg.dynamic_node_feats_dim
    evaluator = Evaluator(name=args.dataset)
    train_loader = DGDataLoader(
        train_dg,
        batch_unit=args.batch_time_gran,
    )
    val_loader = DGDataLoader(
        val_dg,
        batch_unit=args.batch_time_gran,
    )
    test_loader = DGDataLoader(
        test_dg,
        batch_unit=args.batch_time_gran,
    )

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    for batch in train_loader:
        print(batch.src.shape)
        print('\n')

    labels = label_generator_next_binary_classification(train_loader)
    token = pd.read_csv('examples/graphproppred/test_token.csv')
    token = preproccess_raw_data(token)

    dgraph_token = DGraph.from_pandas(
        edge_df=token,
        edge_src_col='from',
        edge_dst_col='to',
        edge_time_col='timestamp',
        edge_feats_col='value',  # @TODO: CHECK
        time_delta='s',
        device=args.device,
        # split = 'train' # @TODO: CHECK
    )

    train_loader = DGDataLoader(
        dgraph_token,
        batch_unit=args.batch_time_gran,
    )
    val_loader = DGDataLoader(
        dgraph_token,
        batch_unit=args.batch_time_gran,
    )
    test_loader = DGDataLoader(
        dgraph_token,
        batch_unit=args.batch_time_gran,
    )

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

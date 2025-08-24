import argparse
from typing import Callable

import pandas as pd
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.loader import DGDataLoader
from tgm.split import TemporalRatioSplit
from tgm.util.seed import seed_everything

"""
Adapted the setting for graph property prediction proposed in
https://openreview.net/forum?id=DZqic2sPTY
(`lag` is excluded from this example's setting)

This example can be run with any token networks provided in
https://arxiv.org/abs/2406.10426
"""

parser = argparse.ArgumentParser(
    description='Persistant Forecast Graph Property Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')

parser.add_argument('--train_ratio', type=float, default=0.7, help='train ratio')
parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio')
parser.add_argument('--test_ratio', type=float, default=0.15, help='test ratio')

parser.add_argument(
    '--path-dataset',
    type=str,
    default='examples/graphproppred/test_token.csv',
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
    loader: DGDataLoader, snapshot_measurement: Callable = node_count
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


class PersistantForecast:
    def __init__(self, task: Callable = node_count):
        self.comparator = task
        self.prev_measurement = None

    def reset(self) -> None:
        self.prev_measurement = None

    def __call__(self, batch: DGBatch) -> torch.tensor:
        current_measurement = self.comparator(batch)

        if self.prev_measurement is None:
            pred = torch.randint(0, 2, ())  # random guess for the first prediction
        else:
            pred = torch.tensor(int(current_measurement > self.prev_measurement))
        self.prev_measurement = current_measurement

        return pred


def eval(
    loader: DGDataLoader,
    y_true: torch.Tensor,
    model: PersistantForecast,
    metrics: Metric,
    ignore_last_snapshot=False,
) -> dict:
    all_preds = []
    number_of_snapshot = len(loader)
    for idx, snapshot in tqdm(enumerate(loader)):
        if not (ignore_last_snapshot and idx == number_of_snapshot - 1):
            pred = model(snapshot)
            all_preds.append(pred)
    y_pred = torch.Tensor(all_preds).float()
    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)

    assert torch.all(y_pred[1:] == y_true[: y_true.shape[0] - 1])
    metrics(y_pred, y_true, indexes=indexes)
    return metrics.compute()


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)
    metrics = [BinaryAveragePrecision(), BinaryAUROC()]
    val_metrics = MetricCollection(metrics, prefix='Test')

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

    full_dg = DGraph(full_data)
    train_dg = DGraph(train_data)
    val_dg = DGraph(val_data)
    test_dg = DGraph(test_data)

    full_loader = DGDataLoader(
        full_dg, batch_unit=args.batch_time_gran, on_empty='raise'
    )
    train_loader = DGDataLoader(
        train_dg, batch_unit=args.batch_time_gran, on_empty='raise'
    )
    val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran, on_empty='raise')
    test_loader = DGDataLoader(
        test_dg, batch_unit=args.batch_time_gran, on_empty='raise'
    )

    train_snapshots = len(train_loader)
    val_snapshots = len(val_loader)
    test_snapshots = len(test_loader) - 1

    labels = label_generator_next_binary_classification(
        loader=full_loader, snapshot_measurement=edge_count
    )  # shape: number of snapshots - 1

    train_labels = labels[:train_snapshots]
    val_labels = labels[train_snapshots : train_snapshots + val_snapshots]
    test_labels = labels[
        train_snapshots + val_snapshots : train_snapshots
        + val_snapshots
        + test_snapshots
    ]

    baseline = PersistantForecast(edge_count)
    test_results = eval(test_loader, test_labels, baseline, val_metrics, True)
    print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

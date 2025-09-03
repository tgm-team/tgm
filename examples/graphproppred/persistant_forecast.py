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
Adapted graph property prediction as proposed in https://openreview.net/forum?id=DZqic2sPTY

Note: `lag` is excluded from this example's setting.
"""

parser = argparse.ArgumentParser(
    description='Persistant Forecast Graph Property Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--train-ratio', type=float, default=0.7, help='train ratio')
parser.add_argument('--val-ratio', type=float, default=0.15, help='validation ratio')
parser.add_argument('--test-ratio', type=float, default=0.15, help='test ratio')
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


def edge_count(snapshot: DGBatch):  # return number of edges of current snapshot
    return snapshot.src.shape[0]


def node_count(snapshot: DGBatch):  # return number of nodes of current snapshot
    return len(snapshot.unique_nids)


def generate_binary_trend_labels(
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
) -> dict:
    y_pred = torch.zeros_like(y_true, dtype=torch.float)
    for i, snapshot in enumerate(tqdm(loader)):
        if i != len(loader) - 1:  # Skip last snapshot as we don't have labels for it
            y_pred[i] = model(snapshot)
    indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)

    assert torch.all(y_pred[1:] == y_true[: y_true.shape[0] - 1])
    metrics(y_pred, y_true, indexes=indexes)
    return metrics.compute()


args = parser.parse_args()
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

_, val_data, test_data = full_data.split(
    TemporalRatioSplit(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
)

val_dg, test_dg = DGraph(val_data), DGraph(test_data)
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran, on_empty='raise')
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran, on_empty='raise')

model = PersistantForecast(edge_count)

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

val_labels = generate_binary_trend_labels(val_loader, snapshot_measurement=edge_count)
val_results = eval(val_loader, val_labels, model, val_metrics)
print(' '.join(f'{k}={v:.4f}' for k, v in val_results.items()))

test_labels = generate_binary_trend_labels(test_loader, snapshot_measurement=edge_count)
test_results = eval(test_loader, test_labels, model, test_metrics)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

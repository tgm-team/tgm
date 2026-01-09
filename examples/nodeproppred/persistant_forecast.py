import argparse

import numpy as np
import torch
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.util.logging import enable_logging, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Persistant Forecast NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
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


class PersistantForecaster:
    def __init__(self, num_classes: int) -> None:
        self.memory = {}
        self._default_prediction = torch.zeros(num_classes)

    def update(self, node_id: int, label: torch.Tensor) -> None:
        self.memory[node_id] = label

    def __call__(self, node_id: int) -> torch.Tensor:
        return self.memory.get(node_id, self._default_prediction)


@log_latency
def eval(
    loader: DGDataLoader,
    model: PersistantForecaster,
    evaluator: Evaluator,
) -> float:
    perf_list = []

    for batch in tqdm(loader):
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        y_pred = torch.zeros_like(y_true)
        for i, node_id in enumerate(batch.node_ids.tolist()):
            y_pred[i] = model(node_id)
            model.update(node_id, y_true[i])

        input_dict = {
            'y_true': y_true,
            'y_pred': y_pred,
            'eval_metric': [METRIC_TGB_NODEPROPPRED],
        }
        perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

train_loader = DGDataLoader(train_dg, batch_unit=args.snapshot_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.snapshot_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.snapshot_time_gran)

num_classes = train_dg.node_x_dim
model = PersistantForecaster(num_classes=num_classes)

eval(train_loader, model, evaluator)

val_ndcg = eval(val_loader, model, evaluator)
log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_ndcg)

test_ndcg = eval(test_loader, model, evaluator)
log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_ndcg)

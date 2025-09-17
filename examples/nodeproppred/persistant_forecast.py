import argparse
import time

import numpy as np
import torch
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGData, DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.loader import DGDataLoader
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
    '--capture-gpu', action=argparse.BooleanOptionalAction, help='record peak gpu usage'
)
parser.add_argument(
    '--capture-cprofile', action=argparse.BooleanOptionalAction, help='record cprofiler'
)


class PersistantForecaster:
    def __init__(self, num_classes: int) -> None:
        self.memory = {}
        self._default_prediction = torch.zeros(num_classes)

    def update(self, node_id: int, label: torch.Tensor) -> None:
        self.memory[node_id] = label

    def __call__(self, node_id: int) -> torch.Tensor:
        return self.memory.get(node_id, self._default_prediction)


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


args = parser.parse_args()
seed_everything(args.seed)

from pathlib import Path

from experiments import save_experiment_results_and_exit, setup_experiment

results = setup_experiment(args, Path(__file__))

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

train_loader = DGDataLoader(train_dg, batch_unit=args.snapshot_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.snapshot_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.snapshot_time_gran)

evaluator = Evaluator(name=args.dataset)
num_classes = train_dg.dynamic_node_feats_dim
model = PersistantForecaster(num_classes=num_classes)

start_time = time.perf_counter()
eval(train_loader, model, evaluator)
end_time = time.perf_counter()
latency = end_time - start_time

start_time = time.perf_counter()
val_ndcg = eval(val_loader, model, evaluator)
print(f'Latency={latency:.4f} Validation {METRIC_TGB_NODEPROPPRED}={val_ndcg:.4f}')
end_time = time.perf_counter()

results[f'val_{METRIC_TGB_NODEPROPPRED}'] = val_ndcg

test_ndcg = eval(test_loader, model, evaluator)
print(f'Test {METRIC_TGB_NODEPROPPRED}={test_ndcg:.4f}')

results[f'test_{METRIC_TGB_NODEPROPPRED}'] = test_ndcg
save_experiment_results_and_exit(results)

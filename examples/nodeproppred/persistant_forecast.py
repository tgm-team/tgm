r"""python -u persistant_forecast.py --dataset tgbn-trade --time-gran Y --batch-time-gran Y
python -u persistant_forecast.py --dataset tgbn-genre --time-gran s --batch-time-gran D
example commands to run this script.
"""

import argparse

import numpy as np
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGData, DGraph
from tgm.loader import DGDataLoader
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Persistant Forecast Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--batch-time-gran',
    type=str,
    default='D',
    help='time granularity to operate on for snapshots',
)


class PersistantForecaster:
    def __init__(self, num_classes: int) -> None:
        self.dict = {}
        self.num_classes = num_classes

    def update(self, node_id: int, label: np.ndarray) -> None:
        self.dict[node_id] = label

    def __call__(self, node_id: int) -> np.ndarray:
        if node_id in self.dict:
            return self.dict[node_id]
        else:
            return np.zeros(self.num_classes)


def eval(loader: DGDataLoader, model: PersistantForecaster) -> dict:
    eval_metric = 'ndcg'
    total_score, num_batches = 0, 0

    for batch in tqdm(loader):
        if batch.dynamic_node_feats is None:
            continue

        labels = batch.dynamic_node_feats.cpu().detach().numpy()
        node_ids = batch.node_ids.cpu().detach().numpy()

        preds = []
        for node_id, label in zip(node_ids, labels):
            preds.append(model(node_id))
            model.update(node_id, label)

        result = evaluator.eval(
            {
                'y_true': labels,
                'y_pred': np.stack(preds),
                'eval_metric': [eval_metric],
            }
        )
        total_score += result[eval_metric]
        num_batches += 1

    return {eval_metric: total_score / num_batches}


args = parser.parse_args()
seed_everything(args.seed)

data = DGData.from_tgb(args.dataset).discretize(args.time_gran)
train_data, val_data, test_data = data.split()

train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

train_loader = DGDataLoader(train_dg, batch_unit=args.batch_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran)

evaluator = Evaluator(name=args.dataset)
model = PersistantForecaster(num_classes=test_dg.dynamic_node_feats_dim)

train_results = eval(train_loader, model)
print('Training results: ', ' '.join(f'{k}={v:.4f}' for k, v in train_results.items()))

val_results = eval(val_loader, model)
print('Validation results: ', ' '.join(f'{k}={v:.4f}' for k, v in val_results.items()))

test_results = eval(test_loader, model)
print('Test results: ', ' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

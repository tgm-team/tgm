r"""python -u persistant_forecast.py --dataset tgbn-trade --time-gran s --batch-time-gran Y
python -u persistant_forecast.py --dataset tgbn-genre --time-gran s --batch-time-gran D
example commands to run this script.
"""

import argparse

import numpy as np
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

from tgm import DGraph
from tgm.loader import DGDataLoader
from tgm.util.seed import seed_everything


class PersistantForecaster:
    def __init__(self, num_class):
        self.dict = {}
        self.num_class = num_class

    def update_dict(self, node_id, label):
        self.dict[node_id] = label

    def query_dict(self, node_id):
        r"""Query the dictionary for the last seen label of a node.

        Args:
            node_id: the node to query
        Returns:
            returns the last seen label of the node if it exists, if not return zero vector
        """
        if node_id in self.dict:
            return self.dict[node_id]
        else:
            return np.zeros(self.num_class)


def test_n_upate(
    loader: DGDataLoader,
    forecaster: PersistantForecaster,
) -> dict:
    eval_metric = 'ndcg'
    total_score = 0
    num_labels = 0
    for batch in tqdm(loader):
        # TODO: Consider skipping empty batches natively, when iterating by time (instead of events)
        if not len(batch.src):
            continue

        if batch.dynamic_node_feats is None:
            continue

        preds = []
        label = batch.dynamic_node_feats.cpu().detach().numpy()
        label_srcs = batch.node_ids.cpu().detach().numpy()
        for i in range(0, label_srcs.shape[0]):
            node_id = label_srcs[i]
            pred_vec = forecaster.query_dict(node_id)
            preds.append(pred_vec)
            forecaster.update_dict(node_id, label[i])
            num_labels += 1

        np_pred = np.stack(preds, axis=0)
        np_true = label

        input_dict = {
            'y_true': np_true,
            'y_pred': np_pred,
            'eval_metric': [eval_metric],
        }
        result_dict = evaluator.eval(input_dict)
        score = result_dict[eval_metric]
        total_score += score
    metric_dict = {}
    metric_dict[eval_metric] = float(total_score) / len(loader)
    return metric_dict, num_labels


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


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(args.dataset, time_delta=args.time_gran, split='train')
val_dg = DGraph(args.dataset, time_delta=args.time_gran, split='val')
test_dg = DGraph(args.dataset, time_delta=args.time_gran, split='test')


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

num_nodes = test_dg.num_nodes
num_classes = train_dg.dynamic_node_feats_dim
evaluator = Evaluator(name=args.dataset)
forecaster = PersistantForecaster(num_classes)

train_results, num_train_labels_tgm = test_n_upate(train_loader, forecaster)
print('Training results:')
print(' '.join(f'{k}={v:.4f}' for k, v in train_results.items()))


val_results, num_val_labels_tgm = test_n_upate(val_loader, forecaster)
print('Validation results:')
print(' '.join(f'{k}={v:.4f}' for k, v in val_results.items()))


test_results, num_test_labels_tgm = test_n_upate(test_loader, forecaster)
print('Test results:')
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))


name = args.dataset
dataset = PyGNodePropPredDataset(name=name, root='datasets')
num_classes = dataset.num_classes
data = dataset.get_TemporalData()

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15
)

batch_size = 200
train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)


def loop_data(loader):
    label_t = dataset.get_label_time()  # check when does the first label start
    num_labels = 0
    for batch in loader:
        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            label_ts, label_srcs, labels = (
                label_tuple[0],
                label_tuple[1],
                label_tuple[2],
            )
            label_ts = label_ts.numpy()
            label_srcs = label_srcs.numpy()
            labels = labels.numpy()
            label_t = dataset.get_label_time()

            for i in range(0, label_srcs.shape[0]):
                label_srcs[i]
                num_labels += 1
    return num_labels


num_labels = loop_data(train_loader)
print(f'Number of labels seen during training in TGM: {num_train_labels_tgm}')
print(f'Number of labels seen during training in TGB: {num_labels}')

num_labels = loop_data(val_loader)
print(f'Number of labels seen during validation in TGM: {num_val_labels_tgm}')
print(f'Number of labels seen during validation in TGB: {num_labels}')

num_labels = loop_data(test_loader)
print(f'Number of labels seen during testing in TGM: {num_test_labels_tgm}')
print(f'Number of labels seen during testing in TGB: {num_labels}')

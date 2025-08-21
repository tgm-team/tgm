import argparse

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGData, DGraph
from tgm.hooks import NegativeEdgeSamplerHook
from tgm.loader import DGDataLoader
from tgm.nn import EdgeBankPredictor
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeBank Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--window_ratio', type=float, default=0.15, help='Window ratio')
parser.add_argument('--pos_prob', type=float, default=1.0, help='Positive edge prob')
parser.add_argument(
    '--memory_mode',
    type=str,
    default='unlimited',
    choices=['unlimited', 'fixed'],
    help='Memory mode',
)


def eval(loader: DGDataLoader, model: EdgeBankPredictor, metrics: Metric) -> dict:
    for batch in tqdm(loader):
        pos_out = model(batch.src, batch.dst)
        neg_out = model(batch.src, batch.neg)
        y_pred = torch.cat([pos_out, neg_out], dim=0).float()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        ).long()
        indexes = torch.zeros(y_pred.size(0), dtype=torch.long)
        metrics(y_pred, y_true, indexes=indexes)
        model.update(batch.src, batch.dst, batch.neg)
    return metrics.compute()


args = parser.parse_args()
seed_everything(args.seed)

train_data, _, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data)
test_dg = DGraph(test_data)

train_data = train_dg.materialize(materialize_features=False)
test_loader = DGDataLoader(
    test_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=test_dg.num_nodes),
    batch_size=args.bsize,
)

model = EdgeBankPredictor(
    train_data.src,
    train_data.dst,
    train_data.time,
    memory_mode=args.memory_mode,
    window_ratio=args.window_ratio,
    pos_prob=args.pos_prob,
)

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
test_metrics = MetricCollection(metrics, prefix='Test')

test_results = eval(test_loader, model, test_metrics)
print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))

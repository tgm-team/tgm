import argparse
from pprint import pprint

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR
from tqdm import tqdm

from opendg.graph import DGraph
from opendg.hooks import NegativeEdgeSamplerHook
from opendg.loader import DGDataLoader
from opendg.nn import EdgeBankPredictor
from opendg.util.perf import Usage
from opendg.util.seed import seed_everything

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


def eval(loader: DGDataLoader, model: EdgeBankPredictor, metrics: Metric) -> None:
    for batch in tqdm(loader):
        pos_out = model(batch.src, batch.dst)
        neg_out = model(batch.src, batch.neg)
        y_pred = torch.cat([pos_out, neg_out], dim=0)
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        )
        indexes = torch.zeros(y_pred.size(0))
        metrics(y_true, y_pred.long(), indexes=indexes.long())
    pprint(metrics.compute())


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

train_data = train_dg.materialize(materialize_features=False)
val_loader = DGDataLoader(
    val_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=val_dg.num_nodes),
    batch_size=args.bsize,
)
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

metrics = [BinaryAveragePrecision(), BinaryAUROC(), RetrievalHitRate(), RetrievalMRR()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

with Usage(prefix='Edgebank Validation'):
    eval(val_loader, model, val_metrics)
    eval(test_loader, model, test_metrics)
    val_metrics.reset()
    test_metrics.reset()

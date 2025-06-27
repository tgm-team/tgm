import argparse

import numpy as np
import torch
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm.graph import DGraph
from tgm.hooks import TGBNegativeEdgeSamplerHook
from tgm.loader import DGDataLoader
from tgm.nn import EdgeBankPredictor
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeBank TGB Example',
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


def eval(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    eval_metric: str,
    evaluator: Evaluator,
) -> dict:
    perf_list = []
    for batch in tqdm(loader):
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.tensor(
                [batch.src[idx] for _ in range(len(neg_batch) + 1)]
            )
            query_dst = torch.cat([torch.tensor([batch.dst[idx]]), neg_batch])

            y_pred = model(query_src, query_dst)
            # compute MRR
            input_dict = {
                'y_pred_pos': np.array([y_pred[0]]),
                'y_pred_neg': np.array(y_pred[1:]),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])
        model.update(batch.src, batch.dst, batch.time)
    metric_dict = {}
    metric_dict[eval_metric] = float(np.mean(perf_list))
    return metric_dict


args = parser.parse_args()
seed_everything(args.seed)

dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='val')
test_dg = DGraph(args.dataset, split='test')

train_data = train_dg.materialize(materialize_features=False)
val_loader = DGDataLoader(
    val_dg,
    hook=[TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val')],
    batch_size=args.bsize,
)
test_loader = DGDataLoader(
    test_dg,
    hook=[TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='test')],
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

val_results = eval(val_loader, model, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in val_results.items()))
test_results = eval(test_loader, model, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

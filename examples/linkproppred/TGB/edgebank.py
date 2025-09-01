import argparse
import time

import numpy as np
import torch
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGData, DGraph
from tgm.hooks import HookManager, TGBNegativeEdgeSamplerHook
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
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = model(query_src, query_dst)
            input_dict = {
                'y_pred_pos': y_pred[0].cpu().numpy(),
                'y_pred_neg': y_pred[1:].cpu().numpy(),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])
        model.update(batch.src, batch.dst, batch.time)

    return float(np.mean(perf_list))


args = parser.parse_args()
seed_everything(args.seed)

dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

train_data = train_dg.materialize(materialize_features=False)

hm = HookManager(keys=['val', 'test'])
hm.register('val', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val'))
hm.register('test', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='test'))

val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

model = EdgeBankPredictor(
    train_data.src,
    train_data.dst,
    train_data.time,
    memory_mode=args.memory_mode,
    window_ratio=args.window_ratio,
    pos_prob=args.pos_prob,
)

with hm.activate('val'):
    start_time = time.perf_counter()
    val_mrr = eval(val_loader, model, eval_metric, evaluator)
    end_time = time.perf_counter()
    latency = end_time - start_time
    print(f'Latency={latency:.4f} Validation {eval_metric}={val_mrr:.4f}')

with hm.activate('test'):
    test_mrr = eval(test_loader, model, eval_metric, evaluator)
    print(f'Test {eval_metric}={test_mrr:.4f}')

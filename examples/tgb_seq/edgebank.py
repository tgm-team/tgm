import argparse

import numpy as np
import torch
from tgb_seq.LinkPred.evaluator import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, StatelessHook
from tgm.nn import EdgeBankPredictor
from tgm.util.logging import enable_logging, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeBank TGB-Seq Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='GoogleLocal', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--window-ratio', type=float, default=0.15, help='Window ratio')
parser.add_argument('--pos-prob', type=float, default=1.0, help='Positive edge prob')
parser.add_argument(
    '--memory-mode',
    type=str,
    default='unlimited',
    choices=['unlimited', 'fixed'],
    help='Memory mode',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


@log_latency
def eval(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    evaluator: Evaluator,
) -> float:
    y_pred_pos, y_pred_neg = [], []
    for batch in tqdm(loader):
        negs_per_pos = len(batch.neg)

        for idx in range(negs_per_pos):
            query_src = batch.src[idx].repeat(negs_per_pos + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), batch.neg[idx]])
            y_pred = model(query_src, query_dst)
            y_pred_pos.append(y_pred[0])
            y_pred_neg.append(y_pred[1:])

        model.update(batch.src, batch.dst, batch.time)

    y_pred_pos, y_pred_neg = torch.stack(y_pred_pos), torch.stack(y_pred_neg)
    mrr_list = evaluator.eval(y_pred_pos, y_pred_neg)
    return float(np.mean(mrr_list))


class TGBSEQ_NegativeEdgeSamplerHook(StatelessHook):
    produces = {'neg', 'neg_time'}

    def __init__(self, dataset_name: str, split_mode: str, dgraph: DGraph) -> None:
        self.split = split_mode
        self.num_negs = 100  # TGB-SEQ hardcodes 100 negatives per positive link

        if self.split == 'test':
            # TGB-SEQ precomputed negative destination only for test split.
            from tgb_seq.LinkPred.dataloader import TGBSeqLoader

            self.negs = torch.from_numpy(
                TGBSeqLoader(dataset_name, root='./').negative_samples
            )
            self.neg_idx = 0
        else:
            _, dst, _ = dgraph.edges
            self.low, self.high = int(dst.min()), int(dst.max())

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch_size = len(batch.src)

        if self.split == 'test':
            batch.neg = self.negs[self.neg_idx : self.neg_idx + batch_size]
            self.neg_idx += batch_size
        else:
            size = (self.num_negs, batch.dst.size(0))
            batch.neg = torch.randint(  # type: ignore
                self.low, self.high, size, dtype=torch.int32, device=dg.device
            )
        batch.neg_time = batch.time.clone()
        return batch


seed_everything(args.seed)
evaluator = Evaluator()

train_data, val_data, test_data = DGData.from_tgb_seq(
    args.dataset, root='./data'
).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

_, dst, _ = test_dg.edges
low, high = int(dst.min()), int(dst.max())

hm = HookManager(keys=['val', 'test'])
hm.register(
    'val', TGBSEQ_NegativeEdgeSamplerHook(args.dataset, split_mode='val', dgraph=val_dg)
)
hm.register(
    'test',
    TGBSEQ_NegativeEdgeSamplerHook(args.dataset, split_mode='test', dgraph=test_dg),
)

val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm, drop_last=True)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm, drop_last=True)

train_data = train_dg.materialize(materialize_features=False)
model = EdgeBankPredictor(
    train_data.src,
    train_data.dst,
    train_data.time,
    memory_mode=args.memory_mode,
    window_ratio=args.window_ratio,
    pos_prob=args.pos_prob,
)

with hm.activate('val'):
    val_mrr = eval(val_loader, model, evaluator)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr)

with hm.activate('test'):
    test_mrr = eval(test_loader, model, evaluator)
    log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr)

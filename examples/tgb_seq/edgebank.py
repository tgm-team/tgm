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
parser.add_argument(
    '--data_root', type=str, default='./data', help='Path to store TGB_SEQ datasets'
)


args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


@log_latency
def eval(
    loader: DGDataLoader,
    model: EdgeBankPredictor,
    evaluator: Evaluator,
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        negs_per_pos = len(batch.neg)

        for idx in range(negs_per_pos):
            query_src = batch.src[idx].repeat(negs_per_pos + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), batch.neg[idx]])

            y_pred = model(query_src, query_dst)
            y_pred_pos, y_pred_neg = y_pred[0].unsqueeze(0), y_pred[1:]
            perf_list.append(evaluator.eval(y_pred_pos, y_pred_neg))
        model.update(batch.src, batch.dst, batch.edge_time)

    return float(np.mean(perf_list))


class TGBSEQ_NegativeEdgeSamplerHook(StatelessHook):
    produces = {'neg', 'neg_time'}

    def __init__(
        self, dataset_name: str, split_mode: str, dgraph: DGraph, root: str = './data'
    ) -> None:
        # TGB-SEQ precomputed negative destination only for test split.
        self.has_precomputed_negatives = split_mode == 'test'

        if self.has_precomputed_negatives:
            # TODO: It would be better if we didn't need to load the dataset from disc
            # a second time here, just to access the negatives. This would require
            # copying the internal TGB_SEQ data download logic which seems fragile and
            # in opposition to the expected public API.
            from tgb_seq.LinkPred.dataloader import TGBSeqLoader

            self.negs = torch.from_numpy(
                TGBSeqLoader(dataset_name, root=root).negative_samples
            )
            self.neg_idx = 0
        else:
            # Fallback to random negative sampler on train/val splits
            _, dst, _ = dgraph.edge_events
            self.low, self.high = int(dst.min()), int(dst.max())
            self.num_negs = 100  # TGB-SEQ hardcodes 100 negatives per positive link

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch_size = len(batch.src)

        if self.has_precomputed_negatives:
            batch.neg = self.negs[self.neg_idx : self.neg_idx + batch_size]  # type: ignore
            self.neg_idx += batch_size
        else:
            size = (self.num_negs, batch_size)
            batch.neg = torch.randint(  # type: ignore
                self.low, self.high, size, dtype=torch.int32, device=dg.device
            )

        # TODO: decide whether to keep this (similar to random negative sampler), or sample
        # negative time stamps (similar to TGB negative sampler)
        batch.neg_time = batch.edge_time.clone()  # type: ignore
        return batch


seed_everything(args.seed)
evaluator = Evaluator()

train_data, val_data, test_data = DGData.from_tgb_seq(
    args.dataset, root=args.data_root
).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)

_, dst, _ = test_dg.edge_events
low, high = int(dst.min()), int(dst.max())

hm = HookManager(keys=['val', 'test'])
hm.register(
    'val',
    TGBSEQ_NegativeEdgeSamplerHook(
        args.dataset, split_mode='val', dgraph=val_dg, root=args.data_root
    ),
)
hm.register(
    'test',
    TGBSEQ_NegativeEdgeSamplerHook(
        args.dataset, split_mode='test', dgraph=test_dg, root=args.data_root
    ),
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

import argparse

import numpy as np
import torch
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, TGBNegativeEdgeSamplerHook
from tgm.nn import PopTrackPredictor
from tgm.util.logging import enable_logging, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='PopTrack LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--k',
    type=int,
    default=200,
    help='Number of nodes to consider in top popularity ranking',
)
parser.add_argument('--pos-prob', type=float, default=1.0, help='Positive edge prob')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)

init_decays = {  # from original code's parameter search
    'tgbl-wiki': 0.36,
    'tgbl-coin': 0.93,
    'tgbl-review': 0.997,
    'tgbl-comment': 0.94,
}

decay = init_decays.get(args.dataset, 0.9)


@log_latency
def eval(
    loader: DGDataLoader,
    model: PopTrackPredictor,
    evaluator: Evaluator,
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = model(query_src, query_dst)
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])
        model.update(batch.src, batch.dst, batch.time, decay)

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data)
val_dg = DGraph(val_data)
test_dg = DGraph(test_data)
num_nodes = train_dg.num_nodes + val_dg.num_nodes + test_dg.num_nodes

train_data = train_dg.materialize(materialize_features=False)

hm = HookManager(keys=['val', 'test'])
hm.register('val', TGBNegativeEdgeSamplerHook(args.dataset, split_mode='val'))
hm.register('test', TGBNegativeEdgeSamplerHook(args.dataset, split_mode='test'))

train_loader = DGDataLoader(train_dg, args.bsize)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

model = PopTrackPredictor(
    train_data.src,
    train_data.dst,
    train_data.time,
    num_nodes,
    k=args.k,
    pos_prob=args.pos_prob,
    decay=decay,
)

with hm.activate('val'):
    val_mrr = eval(val_loader, model, evaluator)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr)

with hm.activate('test'):
    test_mrr = eval(test_loader, model, evaluator)
    log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr)

import argparse

import numpy as np

from opendg.graph import DGraph
from opendg.loader import DGBaseLoader, DGDataLoader
from opendg.nn import EdgeBankPredictor
from opendg.util.perf import Usage
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeBank Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--window_ratio', type=float, default=0.15, help='Window ratio')
parser.add_argument('--pos_prob', type=float, default=1.0, help='Positive edge prob')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument(
    '--memory_mode',
    type=str,
    default='unlimited',
    choices=['unlimited', 'fixed'],
    help='Memory mode',
)


def eval(loader: DGBaseLoader, model: EdgeBankPredictor) -> float:
    perf_list = []
    for batch in loader:
        # where is the negative batch list?
        src, dst, time = batch.src, batch.dst, batch.time
        neg_dst = dst
        model(src, dst)
        model(src, neg_dst)
        perf_list.append(0)  # TODO: MRR eval
        model.update(src, dst, time)
    return np.mean(perf_list)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    train_dg = DGraph(args.dataset, split='train')
    val_dg = DGraph(args.dataset, split='valid')

    train_data = train_dg.materialize(materialize_features=False)
    model = EdgeBankPredictor(
        train_data.src,
        train_data.dst,
        train_data.time,
        memory_mode=args.memory_mode,
        window_ratio=args.window_ratio,
        pos_prob=args.pos_prob,
    )

    val_loader = DGDataLoader(val_dg, batch_size=args.bsize)
    with Usage(prefix='Edgebank Validation'):
        eval(val_loader, model)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)

import argparse

import numpy as np

from opendg.graph import DGraph
from opendg.loader import DGBaseLoader, DGDataLoader
from opendg.nn import EdgeBankPredictor
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeBank Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--gpu', type=int, default=-1, help='gpu to use (or -1 for cpu)')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--window_ratio', type=float, default=0.15, help='Window ratio')
parser.add_argument('--pos_prob', type=float, default=1.0, help='Positive edge prob')
parser.add_argument(
    '--dataset', type=str, required=True, default='tgbl-wiki', help='Dataset name'
)
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
        src, pos_dst, neg_dst, time, features = batch

        # TODO: Make model use __call__ for simpler api
        prob_pos, prob_neg = model(batch)
        perf_list.append(prob_pos - prob_neg)  # TODO: MRR eval

        # Need a way to extract from tensors
        model.update_memory(src=src, dst=pos_dst, ts=time)
    return np.mean(perf_list)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    train_dg = DGraph(args.dataset, split='train')
    val_dg = DGraph(args.dataset, split='valid')

    # TODO: Would be convenient to have a dispatcher based on sampling_type
    val_loader = DGDataLoader(val_dg, batch_size=args.bsize)

    # TODO: Need a way to extract src, dst, ts edges without the features
    # device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    model = EdgeBankPredictor(  # TODO: Why doesn't edgebank work with tensors
        src=None,
        dst=None,
        ts=None,
        memory_mode=args.memory_mode,
        window_ratio=args.window_ratio,
        pos_prob=args.pos_prob,
    )
    mrr = eval(val_loader, model)
    print(f'Val MRR: {mrr:.4f}')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)

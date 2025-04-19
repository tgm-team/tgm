import argparse

from opendg.graph import DGraph
from opendg.hooks import NegativeEdgeSamplerHook
from opendg.loader import DGDataLoader
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument(
    '--time_gran',
    type=str,
    default='h',
    help='time granularity to operate on for snapshots',
)


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

train_loader = DGDataLoader(
    train_dg,
    hook=NegativeEdgeSamplerHook(low=0, high=train_dg.num_nodes),
    batch_size=1,
    batch_unit=args.time_gran,
)

for batch in train_loader:
    print(batch)
    break

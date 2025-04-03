import argparse

from opendg.graph import DGraph
from opendg.hooks import RecencyNeighborSamplerHook
from opendg.loader import DGDataLoader
from opendg.util.perf import Usage
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Recency Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')

args = parser.parse_args()
seed_everything(args.seed)

dg = DGraph(args.dataset, split='all')
loader = DGDataLoader(
    dg,
    hook=RecencyNeighborSamplerHook(num_nbrs=[20], num_nodes=dg.num_nodes),
    batch_size=args.bsize,
)

with Usage('RecencySampler'):
    for batch in loader:
        ...

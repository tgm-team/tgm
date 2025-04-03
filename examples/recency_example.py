import argparse
import timeit

from opendg.graph import DGraph
from opendg.hooks import RecencyNeighborSamplerHook
from opendg.loader import DGDataLoader
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeBank Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')


args = parser.parse_args()
seed_everything(args.seed)

dg = DGraph(args.dataset, split='all')

start_time = timeit.default_timer()

loader = DGDataLoader(
    dg,
    hook=RecencyNeighborSamplerHook(num_nbrs=[20], num_nodes=dg.num_nodes),
    batch_size=args.bsize,
)

for batch in loader:
    batch.src, batch.dst, batch.nbrs

end_time = timeit.default_timer()
print('Time taken to process all neighbors in the data:', end_time - start_time)

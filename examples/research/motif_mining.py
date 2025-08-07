import argparse
import csv
from typing import List

from tqdm import tqdm

from tgm.graph import DGraph
from tgm.hooks import DGHook, NeighborSamplerHook, RecencyNeighborHook
from tgm.loader import DGDataLoader

parser = argparse.ArgumentParser(
    description='TGAT TGB Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument(
    '--n-nbrs',
    type=int,
    nargs='+',
    default=[20, 20, 20],
    help='num sampled nbrs at each hop',
)
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['uniform', 'recency'],
    help='sampling strategy',
)

args = parser.parse_args()
train_dg = DGraph(args.dataset, split='train', device=args.device)


def _init_hooks(dg: DGraph, sampling_type: str, split_mode: str) -> List[DGHook]:
    if sampling_type == 'uniform':
        nbr_hook = NeighborSamplerHook(num_nbrs=args.n_nbrs)
    elif sampling_type == 'recency':
        nbr_hook = RecencyNeighborHook(
            num_nbrs=args.n_nbrs,
            num_nodes=dg.num_nodes,
            edge_feats_dim=dg.edge_feats_dim,
        )
    else:
        raise ValueError(f'Unknown sampling type: {args.sampling}')

    return [nbr_hook]


def mine_motifs(
    loader: DGDataLoader, max_motifs: int = 100, outfile: str = 'motifs.csv'
):
    valid_motifs = {}  # {src, timestamp}
    skip_batches = 100
    with open(outfile, 'w') as csvfile:
        csv.writer(csvfile)
        for batch in tqdm(loader):
            skip_batches -= 1
            if skip_batches > 0:
                continue

            for hop in range(3):
                seed_nodes = batch.nids[hop]
                batch.nbr_mask[hop].bool()
                nbrs = batch.nbr_nids[hop]
                # nbrs = nbrs[nbr_mask]
                batch.nbr_times[hop]

                print(seed_nodes.shape)
                # print (nbr_mask)
                print(nbrs.shape)
                # print (nbr_times)
            quit()


train_loader = DGDataLoader(
    train_dg,
    hook=_init_hooks(train_dg, args.sampling, 'train'),
    batch_size=args.bsize,
)


mine_motifs(train_loader, max_motifs=500, outfile='wiki_motifs.csv')

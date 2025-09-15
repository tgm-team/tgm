import argparse

import networkx as nx
import numpy as np
import numpy.random as nr
import scipy.sparse as ss

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import HookManager, StatelessHook
from tgm.loader import DGDataLoader
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Density of State Estimation Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--moments', type=int, default=20, help='# of Chebyshev moments')
parser.add_argument(
    '--probing-vectors', type=int, default=100, help='# of probing vectors'
)


class DosHook(StatelessHook):
    """Computes Density of States (DOS): https://arxiv.org/abs/2305.08750."""

    produces = {'dos'}

    def __init__(self, moments: int, probing_vectors: int) -> None:
        self._moments = moments
        self._probing_vectors = probing_vectors

    @staticmethod
    def rescale_matrix(
        A: np.ndarray, lo: int = 0, hi: int = 1, fudge: int = 0
    ) -> np.ndarray:
        I = ss.eye(A.shape[0]) if ss.issparse(A) else np.eye(A.shape[0])
        ab = [(hi - lo) / (2 - fudge), (lo + hi) / 2]
        A = (A - ab[1] * I) / ab[0]
        return A

    @staticmethod
    def filter_jackson(c: np.ndarray) -> np.ndarray:
        N = c.shape[0]
        n = np.arange(N)
        tau = np.pi / (N + 1)
        g = ((N - n + 1) * np.cos(tau * n) + np.sin(tau * n) / np.tan(tau)) / (N + 1)
        return g * c

    @staticmethod
    def mean_moments_cheb(A: np.ndarray, nZ: int, nM: int, kind: int = 1) -> np.ndarray:
        nM = max(nM, 2)
        Z = np.sign(nr.randn(A.shape[1], nZ))
        c = np.zeros((nM, nZ))

        # Run three-term recurrence to compute moments
        TVp = Z
        TVk = kind * A @ Z
        c[0] = np.sum(Z * TVp, 0)
        c[1] = np.sum(Z * TVk, 0)
        for i in range(2, nM):
            TV = 2 * (A @ TVk) - TVp
            TVp = TVk
            TVk = TV
            c[i] = sum(Z * TVk, 0)

        return c.mean(axis=1)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        G = nx.from_edgelist(zip(batch.src.tolist(), batch.dst.tolist()))
        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G)._asfptype()
        L = self.rescale_matrix(L, lo=0, hi=2)
        c = self.mean_moments_cheb(L, nM=self._moments, nZ=self._probing_vectors)
        c = self.filter_jackson(c)
        # _, yy = plot_chebhist((c,), npts=(20 + 1), pflag=False)
        yy = c
        batch.dos = yy  # type: ignore
        return batch


args = parser.parse_args()
seed_everything(args.seed)

dg = DGraph(DGData.from_tgb(args.dataset))

hm = HookManager(keys=['dos'])
hm.register('dos', DosHook(moments=args.moments, probing_vectors=args.probing_vectors))
hm.set_active_hooks('dos')

loader = DGDataLoader(dg, args.bsize, hook_manager=hm)
for batch in loader:
    print(batch.dos)

import argparse

import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

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
parser.add_argument('--n-pts', type=int, default=1001, help='# points for cdf estimate')
parser.add_argument('--moments', type=int, default=20, help='# of Chebyshev moments')
parser.add_argument(
    '--probing-vectors', type=int, default=100, help='# of probing vectors'
)


class DensityOfStateEstimatorHook(StatelessHook):
    """Computes Density of States (DOS): https://arxiv.org/abs/2305.08750."""

    produces = {'dos'}

    def __init__(
        self, moments: int = 20, probing_vectors: int = 100, n_pts: int = 50
    ) -> None:
        self._moments = moments
        self._probing_vectors = probing_vectors
        self._n_pts = n_pts

    @staticmethod
    def rescale_matrix(
        A: np.ndarray, lo: int = 0, hi: int = 1, fudge: int = 0
    ) -> np.ndarray:
        I = sp.eye(A.shape[0])
        a, b = (hi - lo) / (2 - fudge), (lo + hi) / 2
        return (A - b * I) / a

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
        Z = np.sign(np.random.randn(A.shape[1], nZ))
        c = np.zeros((nM, nZ))

        TVp = Z
        TVk = kind * A @ Z
        c[0] = np.sum(Z * TVp, axis=0)
        c[1] = np.sum(Z * TVk, axis=0)
        for i in range(2, nM):
            TV = 2 * (A @ TVk) - TVp
            TVp, TVk = TVk, TV
            c[i] = np.sum(Z * TVk, axis=0)

        return c.mean(axis=1)

    @staticmethod
    def cdf_via_chebyshev_integration(c: np.ndarray, npts: int = 1001) -> np.ndarray:
        xx = np.linspace(-1 + 1e-8, 1 - 1e-8, npts)
        tx = np.arccos(xx)

        yy = c[0] * (tx - np.pi) / 2.0
        idx = np.arange(1, len(c))
        if idx.size > 0:
            yy += np.sum(c[idx, None] * np.sin(idx[:, None] * tx) / idx[:, None], 0)
        return yy * (-2.0 / np.pi)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        G = nx.from_edgelist(zip(batch.src.tolist(), batch.dst.tolist()))
        L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G)._asfptype()
        L = self.rescale_matrix(L, lo=0, hi=2)
        c = self.mean_moments_cheb(L, nM=self._moments, nZ=self._probing_vectors)
        c = self.filter_jackson(c)
        cdf = self.cdf_via_chebyshev_integration(c, npts=self._n_pts)
        batch.dos = np.diff(cdf).clip(min=0)  # type: ignore
        return batch


args = parser.parse_args()
seed_everything(args.seed)

dg = DGraph(DGData.from_tgb(args.dataset))
dos_hook = DensityOfStateEstimatorHook(
    moments=args.moments, probing_vectors=args.probing_vectors, n_pts=args.n_pts
)
hm = HookManager(keys=['dos'])
hm.register('dos', dos_hook)
hm.set_active_hooks('dos')

loader = DGDataLoader(dg, args.bsize, hook_manager=hm)
dos_list = []
for batch in tqdm(loader):
    dos_list.append(batch.dos)

print(f'Computed {len(dos_list)} density state estimates')

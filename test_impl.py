from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Set, Tuple

import torch
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.hooks import HookManager, RecencyNeighborHook, StatefulHook
from tgm.loader import DGDataLoader


class SlowRecencyNeighborHook(StatefulHook):
    requires: Set[str] = set()
    produces = {'nids', 'nbr_nids', 'times', 'nbr_times', 'nbr_feats'}

    """Load neighbors from DGraph using a recency sampling. Each node maintains a fixed number of recent neighbors.

    Args:
        num_nodes (int): Total number of nodes to track.
        num_nbrs (List[int]): Number of neighbors to sample at each hop (max neighbors to keep).
        edge_feats_dim (int): Edge feature dimension on the dynamic graph.
        directed (bool): If true, aggregates interactions in src->dst direction only (default=False).

    Raises:
        ValueError: If the num_nbrs list is empty.
    """

    def __init__(
        self,
        num_nodes: int,
        num_nbrs: List[int],
        edge_feats_dim: int,
        directed: bool = False,
    ) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')

        self._num_nbrs = num_nbrs
        self._max_nbrs = max(num_nbrs)
        self._directed = directed

        # We need edge_feats_dim to pre-allocate the right shape for self._nbr_feats
        self._edge_feats_dim = edge_feats_dim
        self._history: Dict[int, Deque[Any]] = defaultdict(
            lambda: deque(maxlen=self._max_nbrs)
        )

        self._device = torch.device('cpu')

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def reset_state(self) -> None:
        self._history = defaultdict(lambda: deque(maxlen=self._max_nbrs))

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        # TODO: Consider the case where no edge features exist
        device = dg.device
        self._move_queues_to_device_if_needed(device)  # No-op after first batch

        batch.nids, batch.times = [], []  # type: ignore
        batch.nbr_nids, batch.nbr_times = [], []  # type: ignore
        batch.nbr_feats = []  # type: ignore

        # print(
        #    f'Queue checking 3: nbrs = {[x[0] for x in self._history[3]]}, times = {[x[1] for x in self._history[3]]}, feats: {[x[2] for x in self._history[3]]}'
        # )

        for hop, num_nbrs in enumerate(self.num_nbrs):
            if hop == 0:
                seed = [batch.src, batch.dst]
                times = [batch.time.repeat(2)]  # Real link times
                if hasattr(batch, 'neg'):
                    batch.neg = batch.neg.to(device)
                    seed.append(batch.neg)
                    times.append(batch.neg_time)  # type: ignore
                seed_nodes = torch.cat(seed)
                seed_times = torch.cat(times)
            else:
                seed_nodes = batch.nbr_nids[hop - 1].flatten()  # type: ignore
                seed_times = batch.nbr_times[hop - 1].flatten()  # type: ignore

            nbr_nids, nbr_times, nbr_feats = self._get_recency_neighbors(
                seed_nodes, seed_times, num_nbrs
            )

            batch.nids.append(seed_nodes)  # type: ignore
            batch.times.append(seed_times)  # type: ignore
            batch.nbr_nids.append(nbr_nids)  # type: ignore
            batch.nbr_times.append(nbr_times)  # type: ignore
            batch.nbr_feats.append(nbr_feats)  # type: ignore

        self._update(batch)
        return batch

    def _get_recency_neighbors(
        self, node_ids: torch.Tensor, query_times: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, ...]:
        num_nodes = node_ids.size(0)
        device = node_ids.device
        nbr_nids = torch.full(
            (num_nodes, k), PADDED_NODE_ID, dtype=torch.long, device=device
        )
        nbr_times = torch.zeros((num_nodes, k), dtype=torch.long, device=device)
        nbr_feats = torch.zeros((num_nodes, k, self._edge_feats_dim), device=device)

        for i in range(num_nodes):
            nid, qtime = int(node_ids[i]), int(query_times[i])
            history = self._history[nid]
            valid = [(nbr, t, f) for (nbr, t, f) in history if t < qtime]
            if not valid:
                continue
            valid = valid[-k:]  # most recent k

            nbr_nids[i, -len(valid) :] = torch.tensor(
                [x[0] for x in valid], dtype=torch.long, device=device
            )
            nbr_times[i, -len(valid) :] = torch.tensor(
                [x[1] for x in valid], dtype=torch.long, device=device
            )
            nbr_feats[i, -len(valid) :] = torch.stack([x[2] for x in valid])

        return nbr_nids, nbr_times, nbr_feats

    def _update(self, batch: DGBatch) -> None:
        src, dst, time = batch.src.tolist(), batch.dst.tolist(), batch.time.tolist()
        # if batch.edge_feats is None:
        if True:
            edge_feats = (
                torch.arange(len(src) * self._edge_feats_dim)
                .reshape((len(src), self._edge_feats_dim))
                .to(self._device)
                .float()
            )
        else:
            edge_feats = batch.edge_feats

        for s, d, t, f in zip(src, dst, time, edge_feats):
            self._history[s].append((d, t, f.clone()))  # may need to f.clone()
            if not self._directed:
                self._history[d].append((s, t, f.clone()))  # may need to f.clone()

    def _move_queues_to_device_if_needed(self, device: torch.device) -> None:
        if device != self._device:
            self._device = device


def setup_loader(dg, nbr_class):
    sampler = nbr_class(
        num_nbrs=[1, 1],
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
        # edge_feats_dim=1,
    )
    hm = HookManager(keys=['global'])
    hm.register_shared(sampler)
    hm.set_active_hooks('global')
    return DGDataLoader(dg, batch_size=20, hook_manager=hm)


def assert_batch_eq(batch, exp_batch):
    # print('Batch: ', batch.src, batch.dst, batch.time)
    # torch.set_printoptions(profile='full')  # disables summarization
    for hop in range(1):
        num_nodes, _ = batch.nbr_nids[hop].shape

        for i in range(num_nodes):
            actual_nids = batch.nbr_nids[hop][i]
            expected_nids = exp_batch.nbr_nids[hop][i]

            # Check that they are permutations
            sorted_actual, actual_idx = torch.sort(actual_nids, stable=True)
            sorted_expected, expected_idx = torch.sort(expected_nids, stable=True)
            assert torch.equal(
                sorted_actual, sorted_expected
            ), f'nbr_nids mismatch  at {batch.nids[hop][i]}'

            # Compute permutation from actual -> expected
            # Map actual_idx -> expected_idx
            perm = torch.empty_like(actual_idx)
            perm[actual_idx] = expected_idx

            # Apply permutation to times and feats
            permuted_times = batch.nbr_times[hop][i][perm]
            permuted_feats = batch.nbr_feats[hop][i][perm, :]

            # Assert equality
            assert torch.equal(
                permuted_times, exp_batch.nbr_times[hop][i]
            ), f'nbr_times mismatch at {batch.nids[hop][i]}, exp: {exp_batch.nbr_times[hop]}, actu: {batch.nbr_times[hop]}'
            assert torch.allclose(
                permuted_feats, exp_batch.nbr_feats[hop][i], atol=1e-6, rtol=1e-5
            ), f'nbr_feats mismatch at {batch.nids[hop][i]}, exp: {exp_batch.nbr_feats[hop]}, actu: {batch.nbr_feats[hop]}'


data = DGData.from_tgb('tgbl-wiki')
dg = DGraph(data)
slow_loader = setup_loader(dg, SlowRecencyNeighborHook)  # From master
fast_loader = setup_loader(dg, RecencyNeighborHook)

fast_loader_iter = iter(fast_loader)
for exp_batch in tqdm(slow_loader):
    batch = next(fast_loader_iter)
    assert_batch_eq(batch, exp_batch)

slow_loader._hook_manager.reset_state()
fast_loader._hook_manager.reset_state()

fast_loader_iter = iter(fast_loader)
for exp_batch in tqdm(slow_loader):
    batch = next(fast_loader_iter)
    assert_batch_eq(batch, exp_batch)

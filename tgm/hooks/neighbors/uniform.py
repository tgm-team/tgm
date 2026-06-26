from __future__ import annotations

import warnings
from typing import List, Tuple

import torch

from tgm import DGBatch, DGraph
from tgm.core._storage import DGSliceTracker
from tgm.hooks import SeedableHook, StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class NeighborSamplerHook(StatelessHook, SeedableHook):
    """Load neighbors from DGraph using a memory based sampling function.

    Args:
        num_nbrs (List[int]): Number of neighbors to sample at each hop (-1 to keep all)
        directed (bool): If true, aggregates interactions in edge_src->edge_dst direction only (default=False).
        seed_nodes_keys ([List[str]): List of batch attribute keys to identify the initial seed nodes to sample for.
        seed_times_keys ([List[str]): List of batch attribute keys to identify the initial seed times to sample for.
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.


    Note:
        The order of the output tensors respect the order of seed_nodes_keys.
        For instance, for seed node keys ['edge_src', 'edge_dst', 'neg'] will have the first output index (hop 0) contain the concatenation
        of batch.edge_src, batch.edge_dst, batch.neg (in that order). The next index (hop 1) will contain first-hop neighbors of batch.edge_src
        followed by first-hop neighbors of batch.edge_dst, and then those of batch.neg. This pattern repeats for deeper hops.

    Raises:
        ValueError: If the num_nbrs list is empty or has non-positive entries.
        ValueError: If len(seed_nodes_keys) != len(seed_times_keys).
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {
        'seed_nids',
        'seed_times',
        'nbr_nids',
        'nbr_edge_time',
        'nbr_edge_x',
        'seed_node_nbr_mask',
    }

    def __init__(
        self,
        num_nbrs: List[int],
        seed_nodes_keys: List[str],
        seed_times_keys: List[str],
        directed: bool = False,
        id: str | None = None,
    ) -> None:
        super().__init__()
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')
        if not all([isinstance(x, int) and (x > 0) for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')
        self._num_nbrs = num_nbrs
        self._directed = directed

        if len(seed_nodes_keys) != len(seed_times_keys):
            raise ValueError(
                f'len(seed_nodes_keys) ({len(seed_nodes_keys)}) '
                f'!= len(seed_times_keys) ({len(seed_times_keys)})\n'
                f'seed_nodes_keys={seed_nodes_keys}, '
                f'seed_times_keys={seed_times_keys}'
            )
        self._seed_nodes_keys = seed_nodes_keys
        self._seed_times_keys = seed_times_keys
        logger.debug(
            'Seed nodes keys: %s, Seed times keys: %s',
            self._seed_nodes_keys,
            self._seed_times_keys,
        )
        self._warned_seed_None = False
        self._id = id
        self.seed_keys = seed_nodes_keys
        self.__post_init__()

    @property
    def num_nbrs(self) -> List[int]:
        return self._num_nbrs

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch_seed_nids, batch_seed_times = [], []
        batch_nbr_nids, batch_nbr_edge_time = [], []
        batch_nbr_edge_x = []

        def _append_empty_hop() -> None:
            batch_seed_nids.append(torch.empty(0, dtype=torch.int32))
            batch_seed_times.append(torch.empty(0, dtype=torch.int64))
            batch_nbr_nids.append(torch.empty(0, dtype=torch.int32))
            batch_nbr_edge_time.append(torch.empty(0, dtype=torch.int64))
            batch_nbr_edge_x.append(
                torch.empty(0, dg.edge_x_dim).float()  # type: ignore
            )

        seed_nodes, seed_times, seed_node_nbr_mask = self._get_seed_tensors(batch)
        if not seed_nodes.numel():
            logger.debug('No seed_nodes found, appending empty hop information')
            for _ in self.num_nbrs:
                _append_empty_hop()

        else:
            for hop, num_nbrs in enumerate(self.num_nbrs):
                if hop > 0:
                    seed_nodes = batch_nbr_nids[hop - 1].flatten()
                    seed_times = batch_nbr_edge_time[hop - 1].flatten()

                # TODO: Storage needs to use the right device

                # We slice on batch.start_time so that we only consider neighbor events
                # that occurred strictly before this batch
                logger.debug(
                    'Getting uniform nbrs for hop %d with %d seed nodes',
                    hop,
                    seed_nodes.numel(),
                )
                nbr_nids, nbr_edge_time, nbr_edge_x = dg._storage.get_nbrs(
                    seed_nodes,
                    num_nbrs=num_nbrs,
                    slice=DGSliceTracker(end_time=int(batch.edge_time.min()) - 1),
                    directed=self._directed,
                )

                batch_seed_nids.append(seed_nodes)
                batch_seed_times.append(seed_times)
                batch_nbr_nids.append(nbr_nids)
                batch_nbr_edge_time.append(nbr_edge_time)
                batch_nbr_edge_x.append(nbr_edge_x)

        self.add_batch_attribute(batch, 'seed_nids', batch_seed_nids)
        self.add_batch_attribute(batch, 'seed_times', batch_seed_times)
        self.add_batch_attribute(batch, 'nbr_nids', batch_nbr_nids)
        self.add_batch_attribute(batch, 'nbr_edge_time', batch_nbr_edge_time)
        self.add_batch_attribute(batch, 'nbr_edge_x', batch_nbr_edge_x)
        self.add_batch_attribute(batch, 'seed_node_nbr_mask', seed_node_nbr_mask)

        return batch

    def _get_seed_tensors(
        self, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        device = batch.edge_src.device
        seeds, seed_times = [], []
        seed_node_mask = dict()

        offset = 0
        for node_attr, time_attr in zip(self._seed_nodes_keys, self._seed_times_keys):
            missing = [
                attr for attr in (node_attr, time_attr) if not hasattr(batch, attr)
            ]
            if missing:
                raise ValueError(f'Missing seed attributes {missing} on batch')

            seed = getattr(batch, node_attr)
            time = getattr(batch, time_attr)

            for name, tensor in [(node_attr, seed), (time_attr, time)]:
                # We recover from tensor = None, since the current batch could just
                # be missing certain attributes (e.g. dynamic node events), but for
                # non-Tensor and non-None attrs we explicitly raise
                if tensor is None:
                    logger.debug(
                        'Seed attribute %s is None on this batch, skipping', name
                    )
                    if not self._warned_seed_None:
                        warnings.warn(
                            f'Seed attribute {name} is None on this batch, skipping this batch. '
                            'Future occurrences will also be skipped but the warning will be suppressed',
                            UserWarning,
                        )
                        self._warned_seed_None = True
                    break
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f'{name} must be a Tensor, got {type(tensor)}')
                if tensor.ndim != 1:
                    raise ValueError(f'{name} must be 1-D, got shape {tensor.shape}')

                # Bounds checks
                # TODO: Infer self._num_nodes from underlying graph
                self._num_nodes = float('inf')
                if name == node_attr:
                    if (tensor < 0).any() or (tensor >= self._num_nodes).any():
                        raise ValueError(
                            f'Seed nodes in {name} must satisfy 0 <= x < {self._num_nodes}, '
                            f'got values in range [{tensor.min().item()}, {tensor.max().item()}]'
                        )
                    seeds.append(seed.to(device))
                    num_seed_nodes = tensor.shape[0]
                    seed_node_mask[name] = torch.arange(
                        offset, offset + num_seed_nodes, device=device
                    )
                    offset += num_seed_nodes
                elif name == time_attr:
                    if (tensor < 0).any():
                        raise ValueError(
                            f'Seed times in {name} must be >= 0, got min value: {tensor.min().item()}'
                        )
                    seed_times.append(time.to(device))

        if seeds and seed_times:
            seed_nodes, seed_times = torch.cat(seeds), torch.cat(seed_times)  # type: ignore
        else:
            seed_nodes = torch.empty(0, dtype=torch.int32, device=device)
            seed_times = torch.empty(0, dtype=torch.int64, device=device)  # type: ignore
        return seed_nodes, seed_times, seed_node_mask  # type: ignore

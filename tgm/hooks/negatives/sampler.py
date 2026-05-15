from __future__ import annotations

import torch

from tgm.constants import PADDED_NODE_ID
from tgm.core import DGBatch, DGraph
from tgm.hooks.base import StatefulHook, StatelessHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class RandomNegativeEdgeSamplerHook(StatelessHook):
    """Random sampling negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {'neg', 'neg_time'}

    def __init__(
        self, low: int, high: int, neg_ratio: float = 1.0, id: str | None = None
    ) -> None:
        super().__init__()
        if not 0 < neg_ratio <= 1:
            raise ValueError(f'neg_ratio must be in (0, 1], got: {neg_ratio}')
        if not low < high:
            raise ValueError(f'low ({low}) must be strictly less than high ({high})')
        self.low = low
        self.high = high
        self.neg_ratio = neg_ratio
        self._id = id
        self.__post_init__()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        size = (round(self.neg_ratio * batch.edge_dst.size(0)),)
        if size[0] == 0:
            self.add_batch_attribute(
                batch, 'neg', torch.empty(size, dtype=torch.int32, device=dg.device)
            )
            self.add_batch_attribute(
                batch,
                'neg_time',
                torch.empty(size, dtype=torch.int64, device=dg.device),
            )
        else:
            self.add_batch_attribute(
                batch,
                'neg',
                torch.randint(
                    self.low, self.high, size, dtype=torch.int32, device=dg.device
                ),
            )
            self.add_batch_attribute(batch, 'neg_time', batch.edge_time.clone())
        return batch


class HistoricalNegativeEdgeSamplerHook(StatefulHook):
    """Sample negative edges from past interactions for dynamic link prediction.

    Args:
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.

    Notes:
        If a node doesn't have past interactions, we return `PADDED_NODE_ID`(-1) as the negative destination.
        `valid_neg_mask` (BoolTensor): Boolean mask of shape ``(num_neg,)`` indicating
            which entries in ``neg`` are real negative samples. ``True`` means the
            corresponding node id is a valid negative; ``False`` means the entry is
            a padding placeholder (``PADDED_NODE_ID``) and should be excluded from
            loss computation and evaluation.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time'}
    _cls_produces = {'neg', 'neg_time', 'valid_neg_mask'}

    def __init__(
        self,
        id: str | None = None,
    ) -> None:
        super().__init__()

        self._id = id

        self._memory: torch.Tensor | None = None
        self._count: int = 0
        self.__post_init__()

    def reset_state(self) -> None:
        self._memory = None
        self._count = 0

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if self._count == 0:
            neg = torch.full(
                (batch.edge_dst.size(0),),
                PADDED_NODE_ID,
                dtype=batch.edge_dst.dtype,
                device=dg.device,
            )
        else:
            neg = self._hist_sampling(dg, batch)

        neg_time = batch.edge_time.clone()
        valid_neg_mask = neg != PADDED_NODE_ID
        self._update_hst_memory(dg, batch)

        self.add_batch_attribute(batch, 'neg', neg)
        self.add_batch_attribute(batch, 'neg_time', neg_time)
        self.add_batch_attribute(batch, 'valid_neg_mask', valid_neg_mask)
        return batch

    def _hist_sampling(self, dg: DGraph, batch: DGBatch) -> torch.Tensor:
        """Sample negative destination nodes from each source node's historical interactions.

        For each source node in the batch, randomly selects a destination node from
        its past interactions stored in memory. If a source node has no recorded past
        interactions, its corresponding negative sample is set to PADDED_NODE_ID as
        a sentinel value indicating no history is available.

        The random selection is performed via a vectorized scatter-max over random
        weights assigned to each historical edge, avoiding explicit loops.

        Args:
            dg (DGraph): The dynamic graph, used to determine the target device.
            batch (DGBatch): The current batch of edges.

        Returns:
            neg (torch.Tensor): Historically sampled negative destination nodes
                of shape (batch_size,) and dtype int32. Nodes with no historical
                interactions are set to PADDED_NODE_ID.

        Note:
            Assumes self._memory is a tensor of shape (2, num_observed_edges) where
            row 0 contains source nodes and row 1 contains destination nodes of all
            previously observed edges.
        """
        assert self._memory is not None

        mask = torch.isin(self._memory[0], batch.edge_src)
        sampling_edges = self._memory[:, mask]

        # Group duplicate srcs: for each unique src, collect all batch positions
        unique_srcs, inverse = torch.unique(batch.edge_src, return_inverse=True)

        unique_src_to_idx = torch.zeros(
            (int(batch.edge_src.max().item()) + 1,), dtype=torch.long, device=dg.device
        )
        unique_src_to_idx[unique_srcs] = torch.arange(
            unique_srcs.size(0), device=dg.device
        )

        edge_to_unique_idx = unique_src_to_idx[sampling_edges[0]]

        sampling_edges_random_weights = torch.rand(
            sampling_edges.size(1), device=dg.device
        )
        best_weights = torch.full((unique_srcs.size(0),), -1.0, device=dg.device)

        best_weights.scatter_reduce_(
            0, edge_to_unique_idx, sampling_edges_random_weights, reduce='amax'
        )
        best_edge_mask = (
            sampling_edges_random_weights == best_weights[edge_to_unique_idx]
        )

        # Sample one neg per unique src
        neg_per_unique = torch.full(
            (unique_srcs.size(0),),
            PADDED_NODE_ID,
            dtype=sampling_edges.dtype,
            device=dg.device,
        )
        neg_per_unique[edge_to_unique_idx[best_edge_mask]] = sampling_edges[
            1, best_edge_mask
        ]

        # Broadcast back to all batch positions (duplicates get the same sampled neg)
        neg = neg_per_unique[inverse]
        return neg

    def _update_hst_memory(self, dg: DGraph, batch: DGBatch) -> None:
        """Append the current batch of edges to the historical memory buffer.

        Maintains a dynamically resizing memory buffer of observed edges for use
        in historical negative sampling. The buffer doubles in size when capacity
        is exceeded, ensuring expected O(1) time complexity insertion and amortized O(E) space complexity
        where E is the total number of observed edges.

        Args:
            dg (DGraph): The dynamic graph, used to determine the target device.
            batch (DGBatch): The current batch of edges whose source and destination
                nodes will be appended to memory.

        Note:
            - Memory is lazily initialized on the first call with twice the initial
            batch size as the starting capacity.
            - When the buffer is full, it is expanded to the maximum of twice its
            current size or twice the required size, ensuring no immediate
            back-to-back resizes even for large batches.
            - Only source and destination nodes are stored; edge timestamps are
            not retained in memory.
            - This scale linear w.r.t the number of interaction event rather than number of edges.
            Since _memory can contain duplicated edges.
        """
        batch_size = batch.edge_src.size(0)

        if self._memory is None:
            self._memory = torch.zeros(
                (2, batch_size * 2), dtype=torch.int32, device=dg.device
            )

        if self._count + batch_size > self._memory.size(1):
            new_size = max(self._memory.size(1) * 2, (self._count + batch_size) * 2)

            edge_buffer = torch.zeros(
                (2, new_size - self._memory.size(1)),
                dtype=torch.int32,
                device=dg.device,
            )
            self._memory = torch.cat([self._memory, edge_buffer], dim=1)

        self._memory[0, self._count : self._count + batch_size] = batch.edge_src
        self._memory[1, self._count : self._count + batch_size] = batch.edge_dst

        self._count += batch_size

from __future__ import annotations

from typing import Tuple

import torch

from tgm.constants import PADDED_NODE_ID
from tgm.core import DGBatch, DGraph
from tgm.hooks.base import StatefulHook, StatelessHook
from tgm.hooks.registry import hook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


@hook
class RandomNegativeEdgeSamplerHook(StatelessHook):
    """Random sampling negative edges for dynamic link prediction.

    Args:
        low (int): The minimum node id to sample
        high (int) : The maximum node id to sample
        neg_ratio (float): The ratio of sampled negative destination nodes
            to the number of positive destination nodes (default = 1.0).
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.

    Key words: negative sampler, random, uniform,training, link prediction.
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


@hook
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

    Key words: negative sampler, historical,training, link prediction.
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


@hook
class NodeTypeNegativeSamplerHook(StatefulHook):
    """Sample negative destinations matching the node type of each postive edge's destination node.

    For each edge in the batch, samples a negative destination node from previously
    observed destination nodes that share the same node type as the current destination.
    This ensures type-consistency in negative samples, which is important for
    temporal heterogeneous graphs where edges respect node type constraints.

    Args:
        id (str | None): A unique identifier for the hook. The hook's name and all
            attributes it produces will be suffixed with this `id`.

    Notes:
        On the first batch, all negative samples are set to ``PADDED_NODE_ID`` (-1)
        since no historical destinations are available yet.
        ``valid_neg_mask`` (BoolTensor): Boolean mask of shape ``(num_neg,)`` indicating
            which entries in ``neg`` are real negative samples. ``True`` means the
            corresponding node id is a valid negative; ``False`` means the entry is
            a padding placeholder (``PADDED_NODE_ID``) and should be excluded from
            loss computation and evaluation. This occurs when no previously observed
            destination node shares the same type as the current destination.

    Key words: negative sampler, node type, training, link prediction, thg, temporal heterogeneous graph.
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

        self.__post_init__()

    def reset_state(self) -> None:
        self._memory = None

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if dg.node_type is None:
            raise ValueError(
                'dg.node_type is None. NodeTypeNegativeSamplerHook requires node type information in the DGraph.'
            )
        if self._memory is None:
            neg = torch.full(
                (batch.edge_dst.size(0),),
                PADDED_NODE_ID,
                dtype=batch.edge_dst.dtype,
                device=dg.device,
            )
            valid_neg_mask = torch.zeros_like(neg, dtype=torch.bool, device=dg.device)
        else:
            neg, valid_neg_mask = self._node_type_sampling(dg, batch)

        neg_time = batch.edge_time.clone()
        self._update_memory(dg, batch)

        self.add_batch_attribute(batch, 'neg', neg)
        self.add_batch_attribute(batch, 'neg_time', neg_time)
        self.add_batch_attribute(batch, 'valid_neg_mask', valid_neg_mask)
        return batch

    def _node_type_sampling(
        self, dg: DGraph, batch: DGBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random negative destination per edge, constrained to the same node type.

        For each edge, looks up the node type of its destination node, then finds
        all previously seen destination nodes (stored in memory) that share that type.
        A random one is selected via vectorized scatter-argmax over uniform random weights,
        avoiding any explicit Python loops.

        If no previously seen node matches the required type, the negative is set to
        ``PADDED_NODE_ID`` and marked invalid in ``valid_neg_mask``.

        Args:
            dg (DGraph): The dynamic graph containing node type information via ``dg.node_type``.
            batch (DGBatch): The current batch of edges.

        Returns:
            neg (torch.Tensor): Sampled negative destination node ids of shape ``(batch_size,)``.
                Edges with no type-matching historical destination are set to ``PADDED_NODE_ID``.
            valid_neg_mask (torch.BoolTensor): Boolean mask of shape ``(batch_size,)`` where
                ``True`` indicates a valid negative was found for that edge.

        Note:
            Assumes ``self._memory`` is a tensor of shape ``(num_nodes,)`` where index ``i``
            holds the node type of node ``i`` if it has been observed as a destination,
            or ``-1`` if it has not yet been observed.
        """
        assert self._memory is not None
        assert dg.node_type is not None

        dst_node_types = dg.node_type[batch.edge_dst]

        match_node_types = dst_node_types.unsqueeze(1) == self._memory.unsqueeze(0)
        valid_neg_mask = match_node_types.any(dim=1)

        rand_weights = torch.where(
            match_node_types,
            torch.rand_like(match_node_types, dtype=torch.float),
            torch.tensor(-float('inf'), device=dg.device),
        )

        neg = rand_weights.argmax(dim=1)
        neg[~valid_neg_mask] = PADDED_NODE_ID

        return neg, valid_neg_mask

    def _update_memory(self, dg: DGraph, batch: DGBatch) -> None:
        """Update memory with the node types of destination nodes in the current batch.

        Lazily initializes memory on the first call as a tensor of shape ``(num_nodes,)``
        filled with ``-1`` (unobserved). Then records the node type for each destination
        node seen in the current batch, so they become candidates for future negative sampling.

        Args:
            dg (DGraph): The dynamic graph, used to determine device and total node count.
            batch (DGBatch): The current batch of edges whose destination nodes will
                be recorded in memory.

        Note:
            Memory maps node index → node type. A value of ``-1`` means the node
            has not yet appeared as a destination and is not eligible for sampling.
        """
        if self._memory is None:
            self._memory = torch.full(
                (dg.num_nodes,),
                -1,
                dtype=torch.int32,
                device=dg.device,
            )
        assert dg.node_type is not None

        self._memory[batch.edge_dst] = dg.node_type[batch.edge_dst]

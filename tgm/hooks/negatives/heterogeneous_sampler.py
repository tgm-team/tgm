from __future__ import annotations

from typing import Any, Tuple

import torch

from tgm.constants import PADDED_NODE_ID
from tgm.core import DGBatch, DGraph
from tgm.hooks.base import StatefulHook
from tgm.hooks.negatives.tgb_base_sampler import TGBNegativeEdgeSamplerBase
from tgm.hooks.registry import hook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


@hook
class NodeTypeNegativeSamplerHook(StatefulHook):
    """Sample negative destinations matching the node type of each positive edge's destination node.

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
        num_nodes: int,
        id: str | None = None,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError('num_nodes must be a positive integer.')

        self._id = id
        self._memory: torch.Tensor | None = None
        self._num_nodes = num_nodes

        self.__post_init__()

    def reset_state(self) -> None:
        self._memory = None

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        if dg.node_type is None:
            raise ValueError(
                'dg.node_type is None. NodeTypeNegativeSamplerHook requires node type information in the DGraph.'
            )
        if self._memory is None:
            logger.debug(
                f'NodeTypeNegativeSamplerHook: Empty node label memory on first batch. All negatives will be set to PADDED_NODE_ID ({PADDED_NODE_ID}).'
            )
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
            logger.debug(
                f'Initializing memory for NodeTypeNegativeSamplerHook with shape ({dg.num_nodes},) on device {dg.device}.'
            )
            self._memory = torch.full(
                (self._num_nodes,),
                -1,
                dtype=torch.int32,
                device=dg.device,
            )
        assert dg.node_type is not None

        self._memory[batch.edge_dst] = dg.node_type[batch.edge_dst]


@hook
class TGBTHGNegativeEdgeSamplerHook(TGBNegativeEdgeSamplerBase):
    """Load data from DGraph using pre-generated TGB negative samples for heterogeneous graph.
    Make sure to perform `dataset.load_val_ns()` or `dataset.load_test_ns()` before using this hook.

    Args:
        dataset_name (str): The name of the TGB dataset to produce sampler for.
        split_mode (str): The split mode to use for sampling, either 'val' or 'test'.
        first_node_id (int): identity of the first node
        last_node_id (int): identity of the last destination node
        node_type (Tensor): the node type of each node
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.



    Raises:
        ValueError: If neg_sampler is not provided.

    Key words: heterogeneous graph, link prediction, evaluation, negative sampler, thgl.
    """

    _cls_requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type'}
    _cls_produces = {'neg', 'neg_batch_list', 'neg_time'}
    _dataset_prefix = 'thgl'

    def __init__(
        self,
        dataset_name: str,
        split_mode: str,
        first_node_id: int,
        last_node_id: int,
        node_type: torch.Tensor,
        id: str | None = None,
    ) -> None:
        if first_node_id < 0 or last_node_id < 0:
            raise ValueError('First and last ID of node must be positive')

        if node_type is None:
            raise ValueError('Node type must not be None')

        if node_type.shape[0] < last_node_id:
            raise ValueError(f'last_node_id {last_node_id} must be within node_type')

        self._first_node_id = first_node_id
        self._last_node_id = last_node_id
        self._node_type = node_type
        super().__init__(dataset_name, split_mode, id)

    def _build_sampler(self, dataset_name: str) -> Any:
        try:
            from tgb.linkproppred.thg_negative_sampler import (
                THGNegativeEdgeSampler,
            )
        except ImportError:
            raise ImportError(
                f'TGB required for {self.__class__.__name__}, try `pip install py-tgb`'
            )

        return THGNegativeEdgeSampler(
            dataset_name=dataset_name,
            first_node_id=self._first_node_id,
            last_node_id=self._last_node_id,
            node_type=self._node_type.numpy(),
        )

    def _query_batch(self, batch: DGBatch) -> list:
        return self.neg_sampler.query_batch(
            batch.edge_src,
            batch.edge_dst,
            batch.edge_time,
            batch.edge_type,
            split_mode=self.split_mode,
        )

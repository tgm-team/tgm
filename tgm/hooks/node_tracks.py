from __future__ import annotations

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import StatefulHook
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class EdgeEventsSeenNodesTrackHook(StatefulHook):
    """This hook return all nodes appearing in node labels of the current batch that have seen in the past edge events.
    This hook is for the use case of nodeproppred for models computing node embeddings according to edges such as `DyGFormer` and `TPNet`.

    Args:
        num_nodes (int): Total number of nodes to track.
        id (str): A unique identifier for the hook. The hook’s name and all attributes it produces will be suffixed with this `id`.

    Raises:
        ValueError: If the num_nodes list is negative.
    """

    _cls_requires = {'edge_src', 'edge_dst'}
    _cls_produces = {'seen_nodes', 'batch_nodes_mask'}

    def __init__(self, num_nodes: int, id: str | None = None) -> None:
        super().__init__()
        if num_nodes < 0:
            raise ValueError('num_nodes must be non-negative')
        self._seen_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self._device = torch.device('cpu')

        self._id = id
        self.__post_init__()

    def reset_state(self) -> None:
        logger.debug('Reset state of the hook')
        self._seen_mask.fill_(False)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        self._move_to_device_if_needed(dg.device)  # No-op after first batch

        if batch.node_y_nids is not None:
            batch_nodes = batch.node_y_nids
        else:
            logger.debug('No node event found in the batch')
            batch_nodes = torch.empty(0, device=self._device, dtype=torch.int)

        edge_event_nodes = torch.unique(torch.cat([batch.edge_src, batch.edge_dst]))

        self._seen_mask[edge_event_nodes] = True
        previous_seen = self._seen_mask[batch_nodes]
        self.add_batch_attribute(
            batch, 'batch_nodes_mask', torch.nonzero(previous_seen, as_tuple=True)[0]
        )
        self.add_batch_attribute(batch, 'seen_nodes', batch_nodes[previous_seen])
        return batch

    def _move_to_device_if_needed(self, device: torch.device) -> None:
        if device != self._device:
            self._device = device
            self._seen_mask = self._seen_mask.to(device)

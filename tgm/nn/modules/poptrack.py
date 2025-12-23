from typing import Any

import numpy as np
import torch


class PopTrackPredictor:
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        num_nodes: int,
        k: int = 50,
        decay: float = 0.9,
    ) -> None:
        """PopTrack Predictor.

        Reference: https://openreview.net/pdf?id=9kLDrE5rsW

        This predictor implements the PopTrack baseline for dynamic link prediction,
        introduced in `https://openreview.net/pdf?id=9kLDrE5rsW`.
        It predicts the probability of a link reoccurring based on
        the popularity score of the queried edge's destination.

        Args:
            src (torch.Tensor): Source node IDs of edges used for initialization.
            dst (torch.Tensor): Destination node IDs of edges used for initialization.
            ts (torch.Tensor): Timestamps of edges used for initialization.
            num_nodes (int): The total number of nodes.
            k (int, optional): Number of popular nodes to retrieve from.
            decay (float, optional): temporal decay parameter.
                Must be in ``(0, 1]``. Defaults to ``0.9``.

        Raises:
            ValueError: If ``k`` is nonpositive.
            TypeError: If ``src``, ``dst``, or ``ts`` are not all ``torch.Tensor``.
            ValueError: If ``src``, ``dst``, and ``ts`` do not have the same length,
                or if they are empty.

        Note:
            - The predictions are not conditional on the source.
        """
        if 0 >= k:
            raise ValueError('K must be positive')

        if decay <= 0 or decay > 1:
            raise ValueError('Decay must be in (0,1]')

        if num_nodes <= 0:
            raise ValueError('``num_nodes`` must be set to the total number of nodes.')

        if k > num_nodes:
            raise ValueError('``k`` must be smaller than ``num_nodes``.')

        self._check_input_data(src, dst, ts)
        self.popularity = torch.zeros(num_nodes)
        self.k = k
        self.decay = decay
        self.update(src, dst, ts)

    def update(self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor) -> None:
        """Update PopTrack cache with a batch of edges.

        Args:
            src (torch.Tensor): Source node IDs of the edges.
            dst (torch.Tensor): Destination node IDs of the edges.
            ts (torch.Tensor): Timestamps of the edges.

        Raises:
            TypeError: If inputs are not ``torch.Tensor``.
            ValueError: If input tensors do not have the same length, or are empty.
        """
        self._check_input_data(src, dst, ts)
        self.popularity.index_add_(
            0, dst, torch.ones_like(dst, dtype=self.popularity.dtype)
        )
        self.popularity *= self.decay

    def __call__(
        self, query_src: torch.Tensor, query_dst: torch.Tensor
    ) -> torch.Tensor:
        """Predict link probabilities for a batch of query edges.

        Args:
            query_src (torch.Tensor): Source node IDs of the query edges.
            query_dst (torch.Tensor): Destination node IDs of the query edges.

        Returns:
            torch.Tensor: Predictions of shape ``(len(query_src),)``, where:
                - An edge's probability is the popularity value of
                    its destination node (= original implementation)
                - Otherwise, the probability is ``0.0``.
        """
        pred = self.popularity[query_dst]
        return pred

    def _check_input_data(
        self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor
    ) -> None:
        def _get_info(fn: Any) -> str:
            return f'src: {fn(src)}, dst: {fn(dst)}, ts: {fn(ts)}'

        if not (type(src) == type(dst) == type(ts) == torch.Tensor):
            raise TypeError(f'src, dst, ts must all be Tensor, got {_get_info(type)}')
        if not (len(src) == len(dst) == len(ts)):
            raise ValueError(f'mismatch shape: {_get_info(len)}')
        if len(src) == 0:
            raise ValueError(f'src, dst, ts must have at len > 1, got {_get_info(len)}')

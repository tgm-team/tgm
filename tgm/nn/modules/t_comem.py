from typing import Any, Dict, Literal, Tuple, Deque, DefaultDict
from collections import defaultdict, deque
import torch
import numpy as np
import math


class tCoMemPredictor:
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        num_nodes: int,
        k: int = 50,
        window_ratio: float = 0.15,
        co_occurrence_weight: float = 0.8,
    ) -> None:
        """t-CoMem link predictor with fixed or unlimited memory.

        Reference: https://www.arxiv.org/abs/2506.12764

        This predictor implements the t-CoMem module for dynamic link prediction,
        introduced in `https://www.arxiv.org/abs/2506.12764`.
        It is a memory-based module that mixes popularity with co-occurence.

        Args:
            src (torch.Tensor): Source node IDs of edges used for initialization.
            dst (torch.Tensor): Destination node IDs of edges used for initialization.
            ts (torch.Tensor): Timestamps of edges used for initialization.
            num_nodes (int): Total number of nodes in the dataset.
            k (int, optional): threshold for popularity effect.
                Defaults to 50, must be positive and smaller than ``num_nodes``.
                In general, larger ``k`` leads to better performance but higher memory usage,
                though this usually stops being true after a certain point.
            window_ratio (float, optional): Ratio of the sliding window length to
                the total observed time span (only used if ``memory_mode='fixed'``).
                Must be in ``(0, 1]``. Defaults to ``0.15``.
            co_occurrence_weight (float, optional): Weighting parameter for co-occurrence.
                Must be in ``(0, 1]``. Defaults to ``0.8``.

        Raises:
            TypeError: If ``src``, ``dst``, or ``ts`` are not all ``torch.Tensor``.
            ValueError: If ``src``, ``dst``, and ``ts`` do not have the same length,
                or if they are empty.

        """
        if not 0 < window_ratio <= 1.0:
            raise ValueError(f'Window ratio must be in (0, 1]')

        if not 0 < co_occurrence_weight <= 1.0:
            raise ValueError(f'Co-occurrence weight must be in (0, 1]')

        if 0 >= k:
            raise ValueError(f'K must be positive')

        if num_nodes <= 0:
            raise ValueError('``num_nodes`` must be set to the total number of nodes.')

        if k > num_nodes:
            raise ValueError('``k`` must be smaller than ``num_nodes``.')

        self._check_input_data(src, dst, ts)

        self._window_ratio = window_ratio
        self._window_start, self._window_end = ts.min(), ts.max()
        self._window_size = torch.clamp(self._window_end - self._window_start, min=1.0)

        self.device = src.device
        self.num_nodes = num_nodes
        self.k = k

        self.recent_ts = torch.full(
            (self.num_nodes, self.k), fill_value=-float('inf'), device=self.device
        )

        self.recent_dst = torch.full(
            (self.num_nodes, self.k), fill_value=-1, device=self.device
        )

        self.recent_len = torch.zeros(self.num_nodes)
        self.recent_pos = torch.zeros(self.num_nodes)

        self.node_to_co_occurrence: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
        self.popularity = torch.zeros(num_nodes)
        self.co_occurrence_weight = co_occurrence_weight

        self.update(src, dst, ts)

    def update(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
    ) -> None:
        """Update EdgeBank memory with a batch of edges.

        Args:
            src (torch.Tensor): Source node IDs of the edges.
            dst (torch.Tensor): Destination node IDs of the edges.
            ts (torch.Tensor): Timestamps of the edges.

        Raises:
            TypeError: If inputs are not ``torch.Tensor``.
            ValueError: If input tensors do not have the same length, or are empty.
        """
        self._check_input_data(src, dst, ts)

        self._window_end = torch.max(self._window_end, ts.max())
        self._window_start = self._window_end - self._window_size

        for s, d, t in zip(src.long().tolist(), dst.long().tolist(), ts.tolist()):
            pos = int(self.recent_pos[s])

            self.recent_ts[s, pos] = t
            self.recent_dst[s, pos] = d
            self.recent_pos[s] = (pos + 1) % self.k
            if self.recent_len[s] < self.k:
                self.recent_len[s] += 1
            self.node_to_co_occurrence[s][d] = (
                self.node_to_co_occurrence[s].get(d, 0) + 1
            )
            self.node_to_co_occurrence[d][s] = (
                self.node_to_co_occurrence[d].get(s, 0) + 1
            )

        self.popularity.index_add_(
            0, dst.long(), torch.ones_like(dst, dtype=self.popularity.dtype)
        )

    def __call__(
        self,
        query_src: torch.Tensor,
        query_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Predict link probabilities for a batch of query edges.

        Args:
            query_src (torch.Tensor): Source node IDs of the query edges.
            query_dst (torch.Tensor): Destination node IDs of the query edges.

        Returns:
            torch.Tensor: Predictions of shape ``(len(query_src),)``, where:
                - If the source node has recent neighbors within the time window, the base score
                is computed over those neighbors, and if the pair ``(src, dst)`` has a recorded
                co-occurrence count, an additional co-occurrence term is added.
                - If the source has no valid recent interactions and there is no
                co-occurrence signal, the predicted probability is ``0.0``.
        """
        pred = torch.zeros_like(query_src)

        unique_src, inv = torch.unique(query_src, return_inverse=True)

        # popularity signal
        ts_mat = self.recent_ts[unique_src.long()]
        nbr_mat = self.recent_dst[unique_src.long()].long()
        len_vec = self.recent_len[unique_src.long()].long()

        pos_idx = torch.arange(self.k, device=ts_mat.device)
        valid_mask = pos_idx.unsqueeze(0) < len_vec.unsqueeze(1)

        time_mask = (ts_mat >= self._window_start) & (ts_mat <= self._window_end)
        mask = valid_mask & time_mask

        ts_valid = torch.where(mask, ts_mat, torch.full_like(ts_mat, -float('inf')))
        nbr_valid = torch.where(mask, nbr_mat, torch.zeros_like(nbr_mat))

        decay_vals = torch.exp(-(self._window_end - ts_valid) / self._window_size)
        pop_vals = torch.sigmoid(self.popularity[nbr_valid])

        base_scores = (decay_vals * pop_vals * mask).sum(dim=1)
        pred = base_scores[inv].clone()
        co_vals = torch.zeros_like(query_src)

        # co-occurrence signal
        for i, (s, d) in enumerate(zip(query_src, query_dst)):
            s = s.item()
            d = d.item()
            c = self.node_to_co_occurrence.get(s, {}).get(d, 0)
            co_vals[i] = self.co_occurrence_weight * (c / (1 + c))

        pred += co_vals
        return pred

    @property
    def window_start(self) -> int | float:
        """Return the start timestamp of the current memory window."""
        return self._window_start.item()

    @property
    def window_end(self) -> int | float:
        """Return the end timestamp of the current memory window."""
        return self._window_end.item()

    @property
    def window_ratio(self) -> float:
        """Return the ratio of the memory window size to the full time span."""
        return self._window_ratio

    @property
    def window_size(self) -> int:
        """Return the absolute size of the memory window."""
        return int(self._window_end - self._window_start)

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

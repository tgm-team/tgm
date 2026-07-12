from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple

import torch

from tgm.util.logging import _get_logger

logger = _get_logger(__name__)


class _Event:
    def __init__(
        self,
        edge: Tuple[int, int],
        ts: int,
        left: Optional['_Event'] = None,
        right: Optional['_Event'] = None,
    ) -> None:
        """A node of a simple bidirectional linked list with 2 pointers."""
        self.edge = edge
        self.ts = ts
        self.left = left
        self.right = right


class _EdgeBankBackend(ABC):
    """Interface for EdgeBank memory storage/lookup strategies."""

    @abstractmethod
    def update(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        window_start: float,
    ) -> None:
        """Insert/update a batch of edges into memory.

        Args:
            src (torch.Tensor): Source node IDs of the edges.
            dst (torch.Tensor): Destination node IDs of the edges.
            ts (torch.Tensor): Timestamps of the edges.
            window_start (float): Current start of the memory window (used by
                backends that evict/filter stale edges).
        """
        ...

    @abstractmethod
    def query(
        self,
        query_src: torch.Tensor,
        query_dst: torch.Tensor,
        window_start: float,
    ) -> torch.Tensor:
        """Predict link probabilities for a batch of query edges.

        Args:
            query_src (torch.Tensor): Source node IDs of the query edges.
            query_dst (torch.Tensor): Destination node IDs of the query edges.
            window_start (float): Current start of the memory window.

        Returns:
            torch.Tensor: Predictions of shape ``(len(query_src),)``.
        """
        ...


class _EdgeBankVectorizedBackend(_EdgeBankBackend):
    """Dense N x N tensor backend. Fast, O(N^2) memory."""

    def __init__(
        self,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
        fixed_memory: bool,
        pos_prob: float,
    ) -> None:
        self._fixed_memory = fixed_memory
        self._pos_prob = pos_prob
        self._memory = torch.full((N, N), fill_value=-1, dtype=dtype, device=device)

    def update(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        window_start: float,
    ) -> None:
        idx = src.long() * self._memory.shape[1] + dst.long()
        flat = self._memory.view(-1)
        flat.scatter_reduce_(0, idx, ts.to(flat.dtype), reduce='amax')

    def query(
        self, query_src: torch.Tensor, query_dst: torch.Tensor, window_start: float
    ) -> torch.Tensor:
        mem = self._memory[query_src.long(), query_dst.long()]
        hit = mem != -1
        if self._fixed_memory:
            hit &= mem >= window_start
        return hit.float() * self._pos_prob


class _EdgeBankLookupBackend(_EdgeBankBackend):
    """Dict + doubly linked list backend. O(E) memory, good for sparse/large graphs."""

    def __init__(self, fixed_memory: bool, pos_prob: float) -> None:
        self.memory: Dict[Tuple[int, int], int] = {}
        self._head: Optional[_Event] = None
        self._tail: Optional[_Event] = None
        self._fixed_memory = fixed_memory
        self._pos_prob = pos_prob
        logger.warning(
            'EdgeBank will be slow if events are added/updated out of order.'
        )

    def _clean_up(self, window_start: float) -> None:
        """Clean up edges that are out of window in memory."""
        while self._head and self._head.ts < window_start:
            curr_event = self._head
            if self.memory.get(curr_event.edge, -1) == curr_event.ts:
                self.memory.pop(curr_event.edge)
            self._head = curr_event.right
            if self._head is None:
                self._tail = None

    def update(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        window_start: float,
    ) -> None:
        if (
            self._fixed_memory
            and self._head is not None
            and self._tail is not None
            and self._head.ts < window_start
        ):
            self._clean_up(window_start)

        for src_, dst_, ts_ in zip(src, dst, ts):
            src_, dst_, ts_ = src_.item(), dst_.item(), ts_.item()
            if ts_ < window_start:
                continue

            self.memory[(src_, dst_)] = ts_
            if self._head is None and self._tail is None:
                self._head = self._tail = _Event((src_, dst_), ts_, None, None)
                continue

            new_event = _Event((src_, dst_), ts_, left=None, right=None)
            curr: Optional[_Event] = self._tail

            # Should never loop if events arrive in time-ascending order.
            # O(n) worst case if events are added out of order.
            while curr is not None and ts_ < curr.ts:
                curr = curr.left

            if curr is None:
                new_event.right = self._head
                if self._head is not None:
                    self._head.left = new_event
                self._head = new_event
            else:
                new_event.left = curr
                new_event.right = curr.right
                if curr.right is not None:
                    curr.right.left = new_event
                curr.right = new_event
                if curr == self._tail:
                    self._tail = new_event

    def query(
        self,
        query_src: torch.Tensor,
        query_dst: torch.Tensor,
        window_start: float,
    ) -> torch.Tensor:
        pred = torch.zeros_like(query_src)
        for i, (s, d) in enumerate(zip(query_src.tolist(), query_dst.tolist())):
            mem_val = self.memory.get((s, d))
            if mem_val is not None and (
                not self._fixed_memory or mem_val >= window_start
            ):
                pred[i] = self._pos_prob
        return pred


class EdgeBankPredictor:
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        N: Optional[int] = None,
        memory_mode: Literal['unlimited', 'fixed'] = 'unlimited',
        window_ratio: float = 0.15,
        pos_prob: float = 1.0,
        backend: Literal['vectorized', 'lookup'] = 'vectorized',
    ) -> None:
        """Edgebank link predictor with fixed or unlimited memory.

        Reference: https://arxiv.org/abs/2207.10128.

        Args:
            src (torch.Tensor): Source node IDs of edges used for initialization.
            dst (torch.Tensor): Destination node IDs of edges used for initialization.
            ts (torch.Tensor): Timestamps of edges used for initialization.
            N (int, optional): Total number of nodes in the graph. Required
                when ``backend='vectorized'``.
            memory_mode (Literal['unlimited', 'fixed'], optional): Defaults to
                ``'unlimited'``.
            window_ratio (float, optional): Defaults to ``0.15``.
            pos_prob (float, optional): Defaults to ``1.0``.
            backend (Literal['vectorized', 'lookup'], optional):
                - ``'vectorized'``: Dense N x N tensor backend. Fast, O(N^2) memory.
                - ``'lookup'``: Dict + doubly linked list backend. O(E) memory, good for sparse/large graphs.
                Defaults to ``'lookup'``.

        Raises:
            ValueError: If ``memory_mode`` is not one of ``'unlimited'`` or ``'fixed'``.
            ValueError: If ``window_ratio`` is not in the range ``(0, 1]``.
            ValueError: If ``backend`` is not one of ``'vectorized'`` or ``'lookup'``.
            ValueError: If ``backend='vectorized'`` and ``N`` is not provided.
            TypeError: If ``src``, ``dst``, or ``ts`` are not all ``torch.Tensor``.
            ValueError: If ``src``, ``dst``, and ``ts`` do not have the same length,
                or if they are empty.
        """
        if memory_mode not in ['unlimited', 'fixed']:
            raise ValueError('memory_mode must be "unlimited" or "fixed"')
        if not 0 < window_ratio <= 1.0:
            raise ValueError('Window ratio must be in (0, 1]')
        if backend not in ['vectorized', 'lookup']:
            raise ValueError('backend must be "vectorized" or "lookup"')
        if backend == 'vectorized' and N is None:
            raise ValueError('N must be provided when backend="vectorized"')
        self._check_input_data(src, dst, ts)

        self.pos_prob = pos_prob
        self._window_ratio = window_ratio
        self._fixed_memory = memory_mode == 'fixed'

        self._window_start, self._window_end = ts.min(), ts.max()
        if self._fixed_memory:
            self._window_start = ts.max() - window_ratio * (ts.max() - ts.min())
        self._window_size = self._window_end - self._window_start

        if backend == 'vectorized':
            assert N is not None
            self.backend: _EdgeBankBackend = _EdgeBankVectorizedBackend(
                N, ts.dtype, ts.device, self._fixed_memory, self.pos_prob
            )
        else:
            self.backend = _EdgeBankLookupBackend(self._fixed_memory, self.pos_prob)

        self.update(src, dst, ts)

    def update(self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor) -> None:
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
        self.backend.update(src, dst, ts, self.window_start)

    def __call__(
        self, query_src: torch.Tensor, query_dst: torch.Tensor
    ) -> torch.Tensor:
        """Predict link probabilities for a batch of query edges.

        Args:
            query_src (torch.Tensor): Source node IDs of the query edges.
            query_dst (torch.Tensor): Destination node IDs of the query edges.

        Returns:
            torch.Tensor: Predictions of shape ``(len(query_src),)``, where:
                - If an edge is in memory and valid (within window if fixed mode),
                  its probability is ``self.pos_prob``.
                - Otherwise, the probability is ``0.0``.
        """
        return self.backend.query(query_src, query_dst, self.window_start)

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

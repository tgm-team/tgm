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


class EdgeBankPredictor:
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        memory_mode: Literal['unlimited', 'fixed'] = 'unlimited',
        window_ratio: float = 0.15,
        pos_prob: float = 1.0,
    ) -> None:
        """Edgebank link predictor with fixed or unlimited memory.

        Reference: https://arxiv.org/abs/2207.10128.

        This predictor implements the EdgeBank baseline for dynamic link prediction,
        introduced in `https://arxiv.org/abs/2207.10128`. It stores a memory of past
        edges and predicts the probability of a link reoccurring based on whether
        the queried edge is present in memory. The memory can be either unlimited
        (retains all edges) or fixed (retains only edges within a sliding window).

        Args:
            src (torch.Tensor): Source node IDs of edges used for initialization.
            dst (torch.Tensor): Destination node IDs of edges used for initialization.
            ts (torch.Tensor): Timestamps of edges used for initialization.
            memory_mode (Literal['unlimited', 'fixed'], optional):
                - ``'unlimited'``: Keeps all observed edges in memory.
                - ``'fixed'``: Keeps only edges within a sliding window of time.
                Defaults to ``'unlimited'``.
            window_ratio (float, optional): Ratio of the sliding window length to
                the total observed time span (only used if ``memory_mode='fixed'``).
                Must be in ``(0, 1]``. Defaults to ``0.15``.
            pos_prob (float, optional): The probability assigned to edges present
                in memory. Defaults to ``1.0``.

        Raises:
            ValueError: If ``memory_mode`` is not one of ``'unlimited'`` or ``'fixed'``.
            ValueError: If ``window_ratio`` is not in the range ``(0, 1]``.
            TypeError: If ``src``, ``dst``, or ``ts`` are not all ``torch.Tensor``.
            ValueError: If ``src``, ``dst``, and ``ts`` do not have the same length,
                or if they are empty.

        Note:
            - In ``unlimited`` mode, memory grows with the number of observed edges.
            - In ``fixed`` mode, only edges within the most recent time window are
              retained. The window size is proportional to ``window_ratio``.
        """
        if memory_mode not in ['unlimited', 'fixed']:
            raise ValueError(f'memory_mode must be "unlimited" or "fixed"')
        if not 0 < window_ratio <= 1.0:
            raise ValueError(f'Window ratio must be in (0, 1]')
        self._check_input_data(src, dst, ts)

        self.pos_prob = pos_prob
        self._window_ratio = window_ratio
        self._fixed_memory = memory_mode == 'fixed'

        self._window_start, self._window_end = ts.min(), ts.max()
        if self._fixed_memory:
            self._window_start = ts.max() - window_ratio * (ts.max() - ts.min())
        self._window_size = self._window_end - self._window_start

        self.memory: Dict[Tuple[int, int], int] = {}
        # maintain bidirectional linked list with 2 pointers
        self._head: Optional[_Event] = None
        self._tail: Optional[_Event] = None

        logger.warning(
            'EdgeBank will be slow if events are added/updated out of order.'
        )

        self.update(src, dst, ts)

    def _clean_up(self) -> None:
        """Clean up edges that are out of window in memory."""
        while self._head and self._head.ts < self._window_start:
            curr_event = self._head
            if self.memory.get(curr_event.edge, -1) == curr_event.ts:
                self.memory.pop(curr_event.edge)
            self._head = curr_event.right
            if self._head == None:
                self._tail = None

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

        if (
            self._fixed_memory
            and self._head is not None
            and self._tail is not None
            and self._head.ts < self._window_start
        ):
            self._clean_up()

        for src_, dst_, ts_ in zip(src, dst, ts):
            src_, dst_, ts_ = src_.item(), dst_.item(), ts_.item()
            if ts_ >= self._window_start:
                self.memory[(src_, dst_)] = ts_
                if self._head == self._tail == None:
                    self._head = self._tail = _Event((src_, dst_), ts_, None, None)
                elif self._head is not None and self._tail is not None:
                    new_event = _Event((src_, dst_), ts_, left=None, right=None)
                    curr: _Event | None = self._tail

                    # This while loop should never run assuming events added in time-ascending order.
                    # When events added out of order, time complexity would be O(n)
                    while curr is not None and ts_ < curr.ts:
                        curr = curr.left

                    if curr == None:
                        new_event.right = self._head
                        if self._head is not None:
                            self._head.left = new_event
                        self._head = new_event
                    else:
                        new_event.left = curr
                        new_event.right = curr.right  # type: ignore[union-attr]
                        if curr.right is not None:  # type: ignore[union-attr]
                            curr.right.left = new_event  # type: ignore[union-attr]
                        curr.right = new_event  # type: ignore[union-attr]
                        if curr == self._tail:
                            self._tail = new_event

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
        pred = torch.zeros_like(query_src)
        src_list = query_src.tolist()
        dst_list = query_dst.tolist()
        for i, (s, d) in enumerate(zip(src_list, dst_list)):
            mem_val = self.memory.get((s, d))
            if mem_val is not None:
                if not self._fixed_memory or mem_val >= self.window_start:
                    pred[i] = self.pos_prob
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

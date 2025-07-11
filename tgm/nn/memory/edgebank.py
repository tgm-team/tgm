from typing import Any, Dict, Literal, Tuple

import torch


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
        r"""Edgebank link predictor with fixed or unlimited memory. Reference: https://arxiv.org/abs/2207.10128.

        Args:
            src(torch.Tensor): source node id of the edges for initialization
            dst(torch.Tensor): destination node id of the edges for initialization
            ts(torch.Tensor): timestamp of the edges for initialization
            memory_mode(str): 'unlimited' or 'fixed'
            window_ratio(float): the ratio of the time window length to the total time length (if using fixed memory_mode)
            pos_prob(float): the probability of the link existence for the edges in memory.
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
        self.update(src, dst, ts)

    def update(self, src: torch.Tensor, dst: torch.Tensor, ts: torch.Tensor) -> None:
        r"""Update Edgebank memory with a batch of edges.

        Args:
            src(torch.Tensor): source node id of the edges.
            dst(torch.Tensor): destination node id of the edges.
            ts(torch.Tensor): timestamp of the edges.
        """
        self._check_input_data(src, dst, ts)
        self._window_end = torch.max(self._window_end, ts.max())
        self._window_start = self._window_end - self._window_size
        for src_, dst_, ts_ in zip(src, dst, ts):
            src_, dst_, ts_ = src_.item(), dst_.item(), ts_.item()
            self.memory[(src_, dst_)] = ts_

    def __call__(
        self, query_src: torch.Tensor, query_dst: torch.Tensor
    ) -> torch.Tensor:
        r"""Predict the link probability for each src,dst edge given the current memory.

        Args:
            query_src(torch.Tensor): source node id of the query edges.
            query_dst(torch.Tensor): destination node id of the query edges.

        Returns:
            torch.Tensor: Predictions array where edges in memory return self.pos_prob, otherwise 0.0
        """
        pred = torch.zeros_like(query_src)
        for i, edge in enumerate(zip(query_src, query_dst)):
            edge = (edge[0].item(), edge[1].item())
            if edge in self.memory:
                hit = not self._fixed_memory
                hit |= self._fixed_memory and self.memory[edge] >= self.window_start
                if hit:
                    pred[i] = self.pos_prob
        return pred

    @property
    def window_start(self) -> int | float:
        return self._window_start.item()

    @property
    def window_end(self) -> int | float:
        return self._window_end.item()

    @property
    def window_ratio(self) -> float:
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

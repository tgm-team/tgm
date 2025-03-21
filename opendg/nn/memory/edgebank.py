from typing import Dict, Literal, Tuple

import numpy as np


class EdgeBankPredictor:
    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        ts: np.ndarray,
        memory_mode: Literal['unlimited', 'fixed'] = 'unlimited',
        window_ratio: float = 0.15,
        pos_prob: float = 1.0,
    ) -> None:
        r"""Edgebank link predictor with fixed or unlimited memory. Reference: https://arxiv.org/abs/2207.10128.

        Args:
            src(np.ndarray): source node id of the edges for initialization
            dst(np.ndarray): destination node id of the edges for initialization
            ts(np.ndarray): timestamp of the edges for initialization
            memory_mode(str): 'unlimited' or 'fixed'
            window_ratio(float): the ratio of the time window length to the total time length
            pos_prob(float): the probability of the link existence for the edges in memory.
        """
        if memory_mode not in ['unlimited', 'fixed']:
            raise ValueError(f'memory_mode must be "unlimited" or "fixed"')
        if not 0 < window_ratio <= 1.0:
            raise ValueError(f'Window ratio must be in (0, 1]')
        self._check_input_data(src, dst, ts)

        self.memory_mode = memory_mode
        self.pos_prob = pos_prob
        self._window_ratio = window_ratio
        self._fixed_memory = memory_mode == 'fixed'

        self._window_start = ts.min()
        if self._fixed_memory:
            self._window_start = ts.min() + (ts.max() - ts.min()) * (1.0 - window_ratio)
        self._window_end = ts.max()
        self._duration = self._window_end - self._window_start

        self.memory: Dict[Tuple[int, int], int] = {}
        self.update_memory(src, dst, ts)

    def update_memory(self, src: np.ndarray, dst: np.ndarray, ts: np.ndarray) -> None:
        r"""Update Edgebank memory with a batch of edges.

        Args:
            src(np.ndarray): source node id of the edges.
            dst(np.ndarray): destination node id of the edges.
            ts(np.ndarray): timestamp of the edges.
        """
        self._check_input_data(src, dst, ts)
        self._window_end = max(self._window_end, ts.max())
        self._window_start = self._window_end - self._duration
        for src_, dst_, ts_ in zip(src, dst, ts):
            self.memory[(src_, dst_)] = ts_

    def predict_link(self, query_src: np.ndarray, query_dst: np.ndarray) -> np.ndarray:
        r"""Predict the probability from query src,dst pair given the current memory,
        all edges not in memory will return 0.0 while all observed edges in memory will return self.pos_prob.

        Args:
            query_src(np.ndarray): source node id of the query edges.
            query_dst(np.ndarray): destination node id of the query edges.

        Returns:
            pred(np.ndarray): the prediction for all query edges.
        """
        pred = np.zeros(len(query_src))
        for i, edge in enumerate(zip(query_src, query_dst)):
            if edge in self.memory:
                hit = not self._fixed_memory
                hit |= self._fixed_memory and self.memory[edge] >= self.window_start
                if hit:
                    pred[i] = self.pos_prob
        return pred

    @property
    def window_start(self) -> int:
        return self._window_start

    @property
    def window_end(self) -> int:
        return self._window_end

    @property
    def window_ratio(self) -> float:
        return self._window_ratio

    def _check_input_data(
        self, src: np.ndarray, dst: np.ndarray, ts: np.ndarray
    ) -> None:
        if not (type(src) == type(dst) == type(ts) == np.ndarray):
            raise TypeError('src, dst, ts must all be np.ndarray')
        if not (len(src) == len(dst) == len(ts)):
            raise ValueError(f'mismatch src:{len(src)}, dst: {len(dst)}, ts: {len(ts)}')
        if len(src) == 0:
            raise ValueError('src, dst, and ts must have at least one element')

import numpy as np


class EdgeBankPredictor:
    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        ts: np.ndarray,
        memory_mode: str = 'unlimited',  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15,
        pos_prob: float = 1.0,
    ) -> None:
        r"""Intialize edgebank and specify the memory mode.

        Args:
            src(np.ndarray): source node id of the edges for initialization
            dst(np.ndarray): destination node id of the edges for initialization
            ts(np.ndarray): timestamp of the edges for initialization
            memory_mode(str): 'unlimited' or 'fixed_time_window'
            time_window_ratio(float): the ratio of the time window length to the total time length
            pos_prob(float): the probability of the link existence for the edges in memory.
        """
        if memory_mode not in ['unlimited', 'fixed_time_window']:
            raise ValueError(f'Invalide memory mode for EdgeBank')

        if (time_window_ratio <= 0.0) or (time_window_ratio > 1.0):
            raise ValueError(
                f'Invalide time window ratio for EdgeBank, must be in (0,1]'
            )

        self.memory_mode = memory_mode

        self.cur_t = ts.max()

        if self.memory_mode == 'fixed_time_window':
            # determine the time window size based on ratio from the given src, dst, and ts for initialization
            duration = ts.max() - ts.min()
            self.prev_t = ts.min() + duration * (
                1.0 - time_window_ratio
            )  # the time windows starts from the last ratio% of time
        else:
            self.prev_t = ts.min()
        self.duration = self.cur_t - self.prev_t

        self.memory: dict[tuple[int, int], int] = {}  # {(u,v):1}
        self.pos_prob = pos_prob
        self.update_memory(src, dst, ts)

    def update_memory(self, src: np.ndarray, dst: np.ndarray, ts: np.ndarray) -> None:
        r"""Generate the current and correct state of the memory with the observed edges so far.
        note that historical edges may include training, validation, and already observed test edges.

        Args:
            src(np.ndarray): source node id of the edges.
            dst(np.ndarray): destination node id of the edges.
            ts(np.ndarray): timestamp of the edges.
        """
        if self.memory_mode == 'unlimited':
            self._update_unlimited_memory(src, dst)  # ignores time
        elif self.memory_mode == 'fixed_time_window':
            self._update_time_window_memory(src, dst, ts)
        else:
            raise ValueError('Invalide memory mode!')

    @property
    def start_time(self) -> int:
        r"""The start of time window for edgebank.

        Returns:
            int: start of time window.
        """
        return self.prev_t

    @property
    def end_time(self) -> int:
        r"""The end of time window for edgebank.

        Returns:
            int: end of time window.
        """
        return self.cur_t

    def _update_unlimited_memory(
        self, update_src: np.ndarray, update_dst: np.ndarray
    ) -> None:
        for src, dst in zip(update_src, update_dst):
            if (src, dst) not in self.memory:
                self.memory[(src, dst)] = 1

    def _update_time_window_memory(
        self, update_src: np.ndarray, update_dst: np.ndarray, update_ts: np.ndarray
    ) -> None:
        # * update the memory if it is not empty
        if update_ts.max() > self.cur_t:
            self.cur_t = update_ts.max()
            self.prev_t = self.cur_t - self.duration

        # * add new edges to the time window
        for src, dst, ts in zip(update_src, update_dst, update_ts):
            self.memory[(src, dst)] = ts

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
        idx = 0
        for src, dst in zip(query_src, query_dst):
            if (src, dst) in self.memory:
                if self.memory_mode == 'fixed_time_window':
                    if self.memory[(src, dst)] >= self.prev_t:
                        pred[idx] = self.pos_prob
                else:
                    pred[idx] = self.pos_prob
            idx += 1

        return pred

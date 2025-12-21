from typing import Any, Dict, Literal, Tuple, Deque, DefaultDict
from collections import defaultdict, deque
import torch
import numpy as np

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
        decay: float = 0.9,
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
            num_nodes (int): 
            k (int, optional): threshold for popularity effect. Defaults to 50.
            window_ratio (float, optional): Ratio of the sliding window length to
                the total observed time span (only used if ``memory_mode='fixed'``).
                Must be in ``(0, 1]``. Defaults to ``0.15``.
            co_occurrence_weight (float, optional): Weighting parameter for co-occurrence. 
                Must be in ``(0, 1]``. Defaults to ``0.8``.
            decay (float, optional): temporal decay parameter. 
                Must be in ``(0, 1]``. Defaults to ``0.9``. 

        Raises:
            TypeError: If ``src``, ``dst``, or ``ts`` are not all ``torch.Tensor``.
            ValueError: If ``src``, ``dst``, and ``ts`` do not have the same length,
                or if they are empty.

        """
        if not 0 < window_ratio <= 1.0:
            raise ValueError(f'Window ratio must be in (0, 1]')
        self._check_input_data(src, dst, ts)
        
        self._window_ratio = window_ratio
        self._window_start, self._window_end = ts.min(), ts.max()
        self._window_size = self._window_end - self._window_start
        
        self.device = src.device

        self.node_to_recent_dests: DefaultDict[int, Deque[Tuple[float, int]]] = defaultdict(lambda: deque(maxlen=k))
        self.node_to_co_occurrence: DefaultDict[int, Dict[int, int]] = defaultdict(dict)  
        self.popularity = np.zeros(num_nodes)
        self.k = k 
        self.top_k: np.ndarray = np.zeros(self.k, dtype=int)
        self.co_occurrence_weight = co_occurrence_weight
        
        self.update(src, dst, ts, decay=decay)

    def update(self, 
               src: torch.Tensor, 
               dst: torch.Tensor, 
               ts: torch.Tensor, 
               decay: float) -> None:
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
        
        for src_, dst_, ts_ in zip(src, dst, ts):
            src_, dst_, ts_ = int(src_.item()), int(dst_.item()), float(ts_.item())
            self.node_to_recent_dests[src_].append((ts_, dst_))
            self.node_to_co_occurrence[src_][dst_] = self.node_to_co_occurrence[src_].get(dst_, 0) + 1
            self.node_to_co_occurrence[dst_][src_] = self.node_to_co_occurrence[dst_].get(src_, 0) + 1
            self.popularity[dst_] += 1

        self.popularity *= decay
        top_k_idx = np.argpartition(self.popularity, -self.k)[-self.k:]
        top_k_idx = top_k_idx[np.argsort(self.popularity[top_k_idx])[::-1]]
        self.top_k = top_k_idx


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
                - If ...,
                  its probability is ``...``.
                - Otherwise, the probability is ``0.0``.
        """
        pred = torch.zeros_like(query_src)
        src_list = query_src.tolist()
        dst_list = query_dst.tolist()
        recent: Deque[Tuple[float, int]] = self.node_to_recent_dests.get(int(query_src), deque())
        for i, (s, d) in enumerate(zip(src_list, dst_list)):
            recent = self.node_to_recent_dests.get(s, deque())
            score = 0.0
            if recent:
                for ts, nbr in recent:
                    if ts < self.window_start or ts > self.window_end:
                        continue

                    decay = torch.exp(-(self._window_end - ts) / self._window_size)
                    score += decay * self.popularity[nbr]
            
            c = self.node_to_co_occurrence.get(s, {}).get(d, 0)
            if c > 0:
                score += self.co_occurrence_weight * (c / (1 + c))

            pred[i] = score / (1 + score) if score > 0 else 0.0

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

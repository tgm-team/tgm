from typing import Any
import torch
import numpy as np

# TODO: decay search

class PopTrackPredictor:
    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        num_nodes: int,
        k: int = 100,
        pos_prob: float = 1.0,
    ) -> None:
        """PopTrack Predictor.

        Reference: https://openreview.net/pdf?id=9kLDrE5rsW

        This predictor implements the PopTrack baseline for dynamic link prediction,
        introduced in `https://openreview.net/pdf?id=9kLDrE5rsW`. 
        It caches the k most popular nodes, and predicts the probability of a link 
        reoccurring based on whether the queried edge leads to one of these k-most
        popular nodes. 
        
        Args:
            src (torch.Tensor): Source node IDs of edges used for initialization.
            dst (torch.Tensor): Destination node IDs of edges used for initialization.
            ts (torch.Tensor): Timestamps of edges used for initialization.
            num_nodes (int): The total number of nodes.
            k (int, optional): Number of popular nodes to retrieve from.
            pos_prob (float, optional): The probability assigned to edges present
                in memory. Defaults to ``1.0``.

        Raises:  
            ValueError: If ``k`` is nonpositive. 
            TypeError: If ``src``, ``dst``, or ``ts`` are not all ``torch.Tensor``.
            ValueError: If ``src``, ``dst``, and ``ts`` do not have the same length,
                or if they are empty.

        Note: the predictions are not conditional on the source.
        """
        if 0 >= k:
            raise ValueError(f'K must be positive')

        self._check_input_data(src, dst, ts)
        self.popularity = np.zeros(num_nodes)
        self.k = k 
        self.top_k = np.zeros(k)
        self.pos_prob = pos_prob
        self.update(src, dst, ts)

    def update(self, 
               src: torch.Tensor, 
               dst: torch.Tensor, 
               ts: torch.Tensor,
               decay: float = 0.7) -> None:
        """Update PopTrack cache with a batch of edges.

        Args:
            src (torch.Tensor): Source node IDs of the edges.
            dst (torch.Tensor): Destination node IDs of the edges.
            ts (torch.Tensor): Timestamps of the edges.
            decay (float, optional): Decay for popularity. 

        Raises:
            TypeError: If inputs are not ``torch.Tensor``.
            ValueError: If input tensors do not have the same length, or are empty.
        """
        self._check_input_data(src, dst, ts)
        for dst_ in dst:
            dst_ = dst_.item()
            self.popularity[dst_] += 1 
        self.popularity *= decay
        top_k_idx = np.argpartition(self.popularity, -self.k)[-self.k:]
        top_k_idx = top_k_idx[np.argsort(self.popularity[top_k_idx])[::-1]]
        self.top_k = top_k_idx


    def __call__(
        self, 
        query_src: torch.Tensor, 
        query_dst: torch.Tensor
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
        pred = torch.zeros_like(query_src)
        src_list = query_src.tolist()
        dst_list = query_dst.tolist()
        for i, (_, d) in enumerate(zip(src_list, dst_list)):
            if d in self.top_k: 
                pred[i] = self.popularity[d]
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

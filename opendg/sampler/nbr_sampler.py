import random
from typing import Dict, List, Optional

from opendg.graph import DGraph


class NbrSampler:
    r"""Base class sampler over a DGraph.

    Args:
        dg (DGraph): The dynamic graph to iterate.
        num_nbrs (List[int]): The number of hops to sample, and the number of neighbours to sample at each hop.

    Raises:
        ValueError: If the num_nbrs list is empty.
        ValueError: If any of the values in num_nbrs is not a positive integer.
    """

    def __init__(self, dg: DGraph, num_nbrs: List[int]) -> None:
        if not len(num_nbrs):
            raise ValueError('num_nbrs must be non-empty')

        if not all([isinstance(x, int) and x > 0 for x in num_nbrs]):
            raise ValueError('Each value in num_nbrs must be a positive integer')

        self._dg = dg
        self._num_nbrs = num_nbrs

    @property
    def num_hops(self) -> int:
        r"""The number of hops that the sampler is using."""
        return len(self.num_nbrs)

    @property
    def num_nbrs(self) -> List[int]:
        r"""The number of hops to sample, and the number of neighbours to sample at each hop."""
        return self._num_nbrs

    def __call__(
        self,
        seed_nodes: List[int],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[int, List[int]]]:
        r"""Perform neighborhood sampling on the DGraph using the input seed nodes and optional time constraints.

        Args:
            seed_nodes (List[int]): The list of nodes to start sampling from.
            start_time: (Optional[int]): Optionally consider only events after the start_time.
            end_time: (Optional[int]): Optionally consider only events before the end_time.

        Returns:
            List of dictionaries, where each index i corresponds to the i'th neighbourhood information.
            The dictionary at each index contains the seed nodes as keys and the sampled neighbourhood as a list of nodes.
        """
        dg = self._dg.slice_time(start_time, end_time)

        nbrs: List[Dict[int, List[int]]] = []
        for hop in range(self.num_hops):
            # TODO: Remap seed nodes in subsequent hops
            hop_nbrs = dg.get_nbrs(seed_nodes)
            sampled_hop_nbrs = self._sample(hop, hop_nbrs)
            nbrs.append(sampled_hop_nbrs)

            # Update seed_nodes for next hop to all sampled_hop_nbrs node ids
            seed_nodes = sum(list(sampled_hop_nbrs.values()), [])
        return nbrs

    def _sample(self, hop: int, nbrs: Dict[int, List[int]]) -> Dict[int, List[int]]:
        # Only uniform sampling supported for now
        for nbr in nbrs:
            if len(nbrs[nbr]) > self.num_nbrs[hop]:
                nbrs[nbr] = random.sample(nbrs[nbr], k=self.num_nbrs[hop])
        return nbrs

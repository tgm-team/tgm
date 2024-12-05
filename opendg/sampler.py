from typing import Dict, List, Optional

from opendg.data import BaseData


class Sampler:
    pass


class UniformSampler(Sampler):
    def __init__(self, num_neighbors: List[int], seed: Optional[int] = None):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.seed = seed

    def sample(
        self, data: BaseData, start_time: int, end_time: int, node_ids: List[int]
    ) -> Dict[int, List[int]]:
        _ = data.aggregate_graph(start_time, end_time)  # edge_index
        """
        sampling logic
        """
        return {}


class OracleUniformSampler(Sampler):
    def __init__(
        self, data: BaseData, num_neighbors: List[int], seed: Optional[int] = None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.seed = seed
        self.data = data
        """
        find the neighbors of each node that it ever had
        """
        self.node_neighbors: Dict[int, List[int]] = {}

    def sample(
        self, start_time: int, end_time: int, node_ids: List[int]
    ) -> Dict[int, List[int]]:
        out_dict = {}
        for node_id in node_ids:
            if node_id not in self.node_neighbors:
                out_dict[node_id] = self.node_neighbors[
                    node_id
                ]  # filter here by time though
        return out_dict


class RecencySampler(Sampler):
    def __init__(self, num_neighbors: List[int], seed: Optional[int] = None):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.seed = seed
        self.stored_neighbors: Dict[int, List[int]] = {}

    def sample(
        self, data: BaseData, start_time: int, end_time: int, node_ids: List[int]
    ) -> Dict[int, List[int]]:
        return {}

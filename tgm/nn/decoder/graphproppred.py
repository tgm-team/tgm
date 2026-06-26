import torch
import torch.nn as nn
from torch.nn import Sequential

from tgm.exceptions import BadAggregatorProtocolError

from ..modules import Aggregator, MeanEmbdPooling


class GraphPredictor(torch.nn.Module):
    r"""Perform pooling over provided node features and perform graph level task.

    Args:
        in_dim (int): Dimension of input
        out_dim (int): Dimension of output
        nlayers (int): Number of layers
        hidden_dim (int): Size of each hidden embeddings
        graph_pooling (Aggregator): graph pooling operation (MeanEmbdPooling by default)

    Note:
        graph_pooling can be selected from [MeanEmbdPooling, SumEmbdPooling] or any custom merging operation, provided it subclasses `BaseEmbdPooling`.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        nlayers: int = 2,
        hidden_dim: int = 64,
        graph_pooling: Aggregator | None = None,
    ) -> None:
        super().__init__()

        self.model = Sequential()
        self.model.append(nn.Linear(in_dim, hidden_dim))
        self.model.append(nn.ReLU())

        for i in range(1, nlayers - 1):
            self.model.append(nn.Linear(hidden_dim, hidden_dim))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(hidden_dim, out_dim))
        self.graph_pooling = (
            graph_pooling if graph_pooling is not None else MeanEmbdPooling(dim=in_dim)
        )
        if not isinstance(self.graph_pooling, Aggregator):
            raise BadAggregatorProtocolError(
                f'Cannot validate {type(self.graph_pooling).__name__}: must implement __call__(*args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor and out_channels() -> int'
            )

    def forward(self, z_nodes: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            z_nodes (torch.Tensor): embedding of nodes
        """
        z_graph = self.graph_pooling(z_nodes)
        return self.model(z_graph)

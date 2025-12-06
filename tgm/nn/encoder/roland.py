from typing import List

import torch
import torch.nn.functional as F
from torch.nn import GRUCell
from torch_geometric.nn import GCNConv, Linear


class ROLAND(torch.nn.Module):
    r"""An implementation of ROLAND.
    https://arxiv.org/abs/2208.07239 .

    Args:
        input_channel (int): Dimension of input.
        out_channel (int): Dimension of output.
        num_nodes (int): Maximum number of nodes.
        dropout (float): dropout rate
        update (str): update mechanism. Choose from ['moving','learnable','gru','mlp',None]
                      If `update` is set to None, the embedding will be update with `tau`

    Reference: https://github.com/manuel-dileo/dynamic-gnn .
    """

    def __init__(
        self,
        input_channel: int,
        out_channel: int,
        num_nodes: int,
        dropout: float = 0.0,
        update: str | None = 'learnable',
        tau: float = 0.5,
    ) -> None:
        assert update in ('moving', 'learnable', 'gru', 'mlp', None)

        super(ROLAND, self).__init__()

        self.conv1 = GCNConv(input_channel, out_channel)
        self.conv2 = GCNConv(out_channel, out_channel)

        self.dropout = dropout
        self.update = update
        if update == 'moving':
            self.tau = torch.Tensor([0])
        elif update == 'learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0]))
        elif update == 'gru':
            self.gru1 = GRUCell(out_channel, out_channel)
            self.gru2 = GRUCell(out_channel, out_channel)
        elif update == 'mlp':
            self.mlp1 = Linear(out_channel * 2, out_channel)
            self.mlp2 = Linear(out_channel * 2, out_channel)
        else:
            assert tau >= 0 and tau <= 1
            self.tau = torch.Tensor([tau])
        self.previous_embeddings = [
            torch.Tensor([[0 for i in range(out_channel)] for j in range(num_nodes)]),
            torch.Tensor([[0 for i in range(out_channel)] for j in range(num_nodes)]),
        ]

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        previous_embeddings: List[torch.Tensor] | None = None,
        num_current_edges: int | None = None,
        num_previous_edges: int | None = None,
    ) -> List[torch.Tensor]:
        if previous_embeddings is not None:
            self.previous_embeddings = [
                previous_embeddings[0].clone(),
                previous_embeddings[1].clone(),
            ]
        if (
            self.update == 'moving'
            and num_current_edges is not None
            and num_previous_edges is not None
        ):  # None if test
            self.tau = torch.Tensor(
                [num_previous_edges / (num_previous_edges + num_current_edges)]
            ).clone()  # tau -- past weight

        current_embeddings = [torch.Tensor([]), torch.Tensor([])]

        # GraphConv1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Embedding Update after first layer
        if self.update == 'gru':
            h = torch.Tensor(
                self.gru1(h, self.previous_embeddings[0].clone().to(h.device)).detach()
            )  # .numpy()
        elif self.update == 'mlp':
            hin = torch.cat(
                (h, self.previous_embeddings[0].clone().to(h.device)), dim=1
            )
            h = torch.Tensor(self.mlp1(hin).detach())  # .numpy()
        else:
            self.tau.to(x.device)
            h = torch.Tensor(
                (
                    self.tau * self.previous_embeddings[0].clone()
                    + (1 - self.tau) * h.clone()
                ).detach()
            )  # .numpy()

        current_embeddings[0] = h.clone()

        # GraphConv2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Embedding Update after second layer
        if self.update == 'gru':
            h = torch.Tensor(
                self.gru2(h, self.previous_embeddings[1].clone().to(h.device)).detach()
            )  # .numpy()
        elif self.update == 'mlp':
            hin = torch.cat(
                (h, self.previous_embeddings[1].clone().to(h.device)), dim=1
            )
            h = torch.Tensor(self.mlp2(hin).detach())  # .numpy()
        else:
            h = torch.Tensor(
                (
                    self.tau * self.previous_embeddings[1].clone()
                    + (1 - self.tau) * h.clone()
                ).detach()
            )  # .numpy()
        current_embeddings[1] = h.clone()

        # NOTE: last GCNConv layer is considered as the embeddings
        return current_embeddings

"""Adapted from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/temporalgcn.py."""

import torch
from torch import zeros
from torch_geometric.nn import GCNConv


class TGCN(torch.nn.Module):
    r"""An implementation of Temporal Graph Convolutional Gated Recurrent Cell.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False. If `improved = True`, the self-loops are added `A+2I` instead of `A+I` giving each node’s own features more influence during aggregation
        cached (bool): Caching the message weights. Default is False. The layer computes the normalized adjacency matrix only once. Speed up training but limit to transductive learning scenario (graph structure is assumed to be static)
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.

    Reference: https://arxiv.org/abs/1811.05320
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_candidate_state_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_update_gate_parameters_and_layers()

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        H: torch.Tensor | None = None,
    ) -> torch.Tensor:
        f"""Forward pass.

        Args:
            X (PyTorch Tensor): Node features.
            edge_index (PyTorch Long Tensor): Graph edge indices.
            edge_weight (PyTorch Long Tensor, optional): Edge weight vector.
            H (PyTorch Tensor, optional): Hidden state matrix for all nodes.
            lambda_max (PyTorch Tensor, optional but mandatory if normalization is not sym): Largest eigenvalue of Laplacian.

        Returns:
            H (PyTorch Tensor): Hidden state matrix for all nodes.

        Note: If edge weights are not present the forward pass defaults to an unweighted graph.
        """
        edge_index = edge_index.to(torch.int64)
        H = self._set_hidden_state(X, H)

        # Eq.3 (Section 3.3.3)
        U_t = self._calculate_update_gate(X, edge_index, edge_weight, H)

        # Eq.4 (Section 3.3.3)
        R_t = self._calculate_reset_gate(X, edge_index, edge_weight, H)

        # Eq.5 (Section 3.3.3)
        C_t = self._calculate_candidate_state(X, edge_index, edge_weight, H, R_t)

        # Eq.6 (Section 3.3.3)
        H_t = self._calculate_hidden_state(U_t, H, C_t)
        return H_t

    def _create_candidate_state_parameters_and_layers(self) -> None:
        self.conv_c = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_c = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_update_gate_parameters_and_layers(self) -> None:
        self.conv_u = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        self.linear_u = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self) -> None:
        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _set_hidden_state(
        self, X: torch.Tensor, H: torch.Tensor | None
    ) -> torch.Tensor:
        if H is None:
            H = zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
    ) -> torch.Tensor:
        U = torch.cat([self.conv_u(X, edge_index, edge_weight), H], 1)
        U = self.linear_u(U)
        U = torch.sigmoid(U)
        return U

    def _calculate_reset_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
    ) -> torch.Tensor:
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], 1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor | None,
        R: torch.Tensor,
    ) -> torch.Tensor:
        C = torch.cat([self.conv_c(X, edge_index, edge_weight), H * R], 1)
        C = self.linear_c(C)
        C = torch.tanh(C)
        return C

    def _calculate_hidden_state(
        self, U: torch.Tensor, H: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        H = U * H + (1 - U) * C
        return H

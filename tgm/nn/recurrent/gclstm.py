"""Adapted from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/gc_lstm.py."""

from typing import Tuple

import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros


class GCLSTM(torch.nn.Module):
    r"""An implementation of Integrated Graph Convolutional Long Short Term Memory Cell.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    Reference: https://arxiv.org/abs/1812.04206
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = 'sym',
        bias: bool = True,
    ) -> None:
        super(GCLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()
        glorot(self.W_i)
        glorot(self.W_f)
        glorot(self.W_c)
        glorot(self.W_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        H: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        lambda_max: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Node features.
            edge_index (PyTorch Long Tensor): Graph edge indices.
            edge_weight (PyTorch Long Tensor, optional): Edge weight vector.
            H (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.
            C (PyTorch Float Tensor, optional): Cell state matrix for all nodes.
            lambda_max (PyTorch Tensor, optional but mandatory if normalization is not sym): Largest eigenvalue of Laplacian.

        Returns:
            H (PyTorch Float Tensor): Hidden state matrix for all nodes.
            C (PyTorch Float Tensor): Cell state matrix for all nodes.

        Note: If edge weights are not present the forward pass defaults to an unweighted graph.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._compute_input_gate(X, edge_index, edge_weight, H, lambda_max)
        F = self._compute_forget_gate(X, edge_index, edge_weight, H, lambda_max)
        C = self._compute_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._compute_output_gate(X, edge_index, edge_weight, H, lambda_max)
        H = self._compute_hidden_state(O, C)
        return H, C

    def _create_input_gate_parameters_and_layers(self) -> None:
        self.conv_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )
        self.W_i = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self) -> None:
        self.conv_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )
        self.W_f = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self) -> None:
        self.conv_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )
        self.W_c = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self) -> None:
        self.conv_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )
        self.W_o = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _set_hidden_state(
        self, X: torch.Tensor, H: torch.Tensor | None
    ) -> torch.Tensor:
        if H is None:
            return torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X: torch.Tensor, C: torch.Tensor | None) -> torch.Tensor:
        if C is None:
            return torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _compute_input_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
        lambda_max: torch.Tensor | None,
    ) -> torch.Tensor:
        I = torch.matmul(X, self.W_i)
        I = I + self.conv_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _compute_forget_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
        lambda_max: torch.Tensor | None,
    ) -> torch.Tensor:
        F = torch.matmul(X, self.W_f)
        F = F + self.conv_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _compute_cell_state(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
        C: torch.Tensor,
        I: torch.Tensor,
        F: torch.Tensor,
        lambda_max: torch.Tensor | None,
    ) -> torch.Tensor:
        T = torch.matmul(X, self.W_c)
        T = T + self.conv_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _compute_output_gate(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        H: torch.Tensor,
        lambda_max: torch.Tensor | None,
    ) -> torch.Tensor:
        O = torch.matmul(X, self.W_o)
        O = O + self.conv_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _compute_hidden_state(self, O: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        H = O * torch.tanh(C)
        return H

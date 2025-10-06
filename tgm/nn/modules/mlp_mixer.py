import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    r"""Two-layered MLP with GELU activation function.

    Args:
        input_dim(int): dimension of input
        dim_expansion_factor(float): dimension expansion factor
        dropout(float): dropout rate

    Reference: https://arxiv.org/abs/2410.04013.
    """

    def __init__(
        self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=int(dim_expansion_factor * input_dim),
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=int(dim_expansion_factor * input_dim),
                out_features=input_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Input tensor

        Returns:
            H : Output tensor
        """
        return self.ffn(X)


class MLPMixer(nn.Module):
    r"""Implementation of MLPMixer.

    Args:
        num_tokens(int): number of tokens
        num_channels(int): number of channels
        token_dim_expansion_factor(float): dimension expansion factor for tokens
        channel_dim_expansion_factor(float): dimension expansion factor for channels
        dropout(float): dropout rate

    Reference: https://openreview.net/forum?id=ayPPc0SyLv1
    """

    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(
            input_dim=num_tokens,
            dim_expansion_factor=token_dim_expansion_factor,
            dropout=dropout,
        )

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            input_dim=num_channels,
            dim_expansion_factor=channel_dim_expansion_factor,
            dropout=dropout,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        f"""Forward pass.
        Eq. 6 (Section  3.2)

        Args:
            X (PyTorch Float Tensor): Input tensor

        Returns:
            H : Output tensor
        """
        H_l_tilde = self.token_norm(X.permute(0, 2, 1))
        H_l_tilde = self.token_feedforward(H_l_tilde).permute(0, 2, 1)
        Z_l_tilde = X + H_l_tilde

        H_l = self.channel_norm(Z_l_tilde)
        H_l = self.channel_feedforward(H_l)
        Z_l = Z_l_tilde + H_l
        return Z_l

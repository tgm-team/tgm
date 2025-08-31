import torch
import torch.nn.functional as F


class TemporalAttention(torch.nn.Module):
    """Multi-head Temporal Attention Module for dynamic/temporal graphs.

    This module computes attention over a node's neighbors considering node features,
    edge features, and time features. It supports multiple attention heads and applies
    residual connection, dropout, and layer normalization to the output.

    Args:
        n_heads (int): Number of attention heads.
        node_dim (int): Dimensionality of node features.
        edge_dim (int): Dimensionality of edge features.
        time_dim (int): Dimensionality of temporal features.
        dropout (float, optional): Dropout probability applied to attention and output
            layers. Default is 0.1.

    Raises:
        ValueError: If `n_heads`, `node_dim`, `edge_dim`, or `time_dim` are <= 0.

    Note:
        The output dimension is `node_dim + time_dim`, padded to be divisible by
        `n_heads` if necessary.
    """

    def __init__(
        self,
        n_heads: int,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if any((x <= 0 for x in [n_heads, node_dim, edge_dim, time_dim])):
            raise ValueError('n_heads,node_dim,edge_dim,time_dim,out_dim must be > 0')

        out_dim = node_dim + time_dim
        self.pad_dim = 0
        if out_dim % n_heads != 0:
            self.pad_dim = n_heads - out_dim % n_heads
            out_dim += self.pad_dim

        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim

        key_dim = node_dim + edge_dim + time_dim
        self.W_Q = torch.nn.Linear(out_dim, out_dim, bias=False)
        self.W_KV = torch.nn.Linear(key_dim, out_dim * 2, bias=False)
        self.W_O = torch.nn.Linear(out_dim, out_dim)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(out_dim)

    def forward(
        self,
        node_feat: torch.Tensor,
        time_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        nbr_node_feat: torch.Tensor,
        nbr_time_feat: torch.Tensor,
        valid_nbr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Temporal Attention module.

        Computes multi-head attention over neighbors, using node, edge, and time
        features, followed by a residual connection, dropout, and layer normalization.

        Args:
            node_feat (torch.Tensor): Node features of shape (B, node_dim).
            time_feat (torch.Tensor): Node time features of shape (B, time_dim).
            edge_feat (torch.Tensor): Edge features for each neighbor of shape
                (B, num_nbrs, edge_dim).
            nbr_node_feat (torch.Tensor): Neighbor node features of shape
                (B, num_nbrs, node_dim).
            nbr_time_feat (torch.Tensor): Neighbor time features of shape
                (B, num_nbrs, time_dim).
            valid_nbr_mask (torch.Tensor): Boolean mask indicating valid neighbors
                of shape (B, num_nbrs). True indicates valid neighbors.

        Returns:
            torch.Tensor: Updated node features of shape (B, out_dim).

        Notes:
            - If a node has no neighbors, masked attention values are set to -1e10
              to avoid NaNs in softmax.
            - The output dimension is padded if necessary to be divisible by the number
              of heads.
        """
        node_feat = F.pad(node_feat, (0, self.pad_dim)) if self.pad_dim else node_feat

        Q = R = torch.cat([node_feat, time_feat], dim=1).unsqueeze(1)  # (B, 1, out_dim)
        Q = self.W_Q(Q)  # (B, out_dim)

        Z = torch.cat([nbr_node_feat, edge_feat, nbr_time_feat], dim=-1)
        Z = self.W_KV(Z)
        K = Z[:, :, : self.out_dim]  # (B, num_nbrs, out_dim)
        V = Z[:, :, self.out_dim :]  # (B, num_nbrs, out_dim)

        Q = Q.reshape(Q.shape[0], -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(K.shape[0], -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(V.shape[0], -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        del Z

        A = torch.einsum('bhld,bhnd->bhln', Q, K)  # (B, n_heads, 1, num_nbrs)
        A *= self.head_dim**-0.5
        del Q, K

        valid_nbr_mask = valid_nbr_mask.reshape(valid_nbr_mask.shape[0], 1, 1, -1)
        valid_nbr_mask = valid_nbr_mask.repeat(1, self.n_heads, 1, 1)

        # If a node has no neighbors (valid_nbr_mask all False), setting masks to -np.inf will cause softmax nans
        # Choose a very large negative number (-1e10 following TGAT) instead
        A = A.masked_fill(~valid_nbr_mask, -1e10)
        A = torch.softmax(A, dim=-1)
        A = self.dropout(A)

        O = torch.einsum('bhln,bhnd->bhld', A, V)  # (B, n_heads, 1, head_dim)
        O = O.flatten(start_dim=1)  # (B, out_dim)
        del A

        out = self.W_O(O)  # (B, out_dim)
        out = self.dropout(out)
        out = self.layer_norm(out + R.squeeze(1))
        return out

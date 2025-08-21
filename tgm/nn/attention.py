import torch
import numpy as np


class TemporalAttention(torch.nn.Module):
    r"""Multi-head Temporal Attention Module.

    Args:
        n_heads (int): The number of heads in the attention module.
        node_dim (int): Feature dimension of node features.
        edge_dim (int): Feature dimension of edge features.
        time_dim (int): Feature dimension of time features.
        out_dim (int): The output latent dimension (must be multiple of n_heads).
        dropout (float): Optional dropout to apply to output linear layer (default=0.1).
    """

    def __init__(
        self,
        n_heads: int,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        out_dim: int = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        out_dim = node_dim + time_dim
        if any((x <= 0 for x in [n_heads, node_dim, edge_dim, time_dim, out_dim])):
            raise ValueError('n_heads,node_dim,edge_dim,time_dim,out_dim must be > 0')

        self.n_heads = n_heads
        self.out_dim = node_dim + time_dim
        self.key_dim = node_dim + edge_dim + time_dim

        if self.out_dim % n_heads != 0:
            self.pad_dim = n_heads - self.out_dim % n_heads
            self.out_dim += self.pad_dim
        else:
            self.pad_dim = 0

        self.query_dim = self.out_dim
        self.head_dim = self.out_dim // n_heads
        self.W_Q = torch.nn.Linear(self.out_dim, n_heads * self.head_dim, bias=False)
        self.W_K = torch.nn.Linear(self.key_dim, n_heads * self.head_dim, bias=False)
        self.W_V = torch.nn.Linear(self.key_dim, n_heads * self.head_dim, bias=False)

        self.W_O = torch.nn.Linear(n_heads * self.head_dim, self.out_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.out_dim)

    def forward(
        self,
        node_feat: torch.Tensor,  # (batch_size, node_dim)
        time_feat: torch.Tensor,  # (batch_size, time_dim)
        nbr_node_feat: torch.Tensor,  # (batch_size, num_nbrs, node_dim)
        nbr_time_feat: torch.Tensor,  # (batch_size, num_nbrs, time_dim)
        nbr_edge_feat: torch.Tensor,  # (batch_size, num_nbrs, edge_dim)
        nbr_mask: np.ndarray,  # (batch_size, num_nbrs)
    ):
        node_feat = torch.unsqueeze(node_feat, dim=1)  # (batch_size, 1, node_dim)

        if self.pad_dim != 0:  # pad for the inputs
            z = torch.zeros(node_feat.shape[0], node_feat.shape[1], self.pad_dim)
            z = z.to(node_feat.device)
            node_feat = torch.cat([node_feat, z], dim=2)

        Q = residual = torch.cat([node_feat, time_feat], dim=2)  # (batch, 1, out_dim)
        # (batch_size, 1, n_heads, self.head_dim)
        Q = self.W_Q(Q).reshape(Q.shape[0], Q.shape[1], self.n_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        K = V = torch.cat([nbr_node_feat, nbr_edge_feat, nbr_time_feat], dim=2)
        K = self.W_K(K).reshape(K.shape[0], K.shape[1], self.n_heads, self.head_dim)
        V = self.W_V(V).reshape(V.shape[0], V.shape[1], self.n_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, n_heads, 1, self.head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, n_heads, num_nbrs, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, n_heads, num_nbrs, self.head_dim)

        A = torch.einsum('bhld,bhnd->bhln', Q, K)  # (batch, n_heads, 1, num_nbrs)
        A *= self.head_dim**-0.5

        # Tensor, shape (batch_size, 1, num_neighbors)
        attn_mask = torch.from_numpy(nbr_mask).to(node_feat.device).unsqueeze(dim=1)
        attn_mask = attn_mask == 0
        # Tensor, shape (batch_size, self.n_heads, 1, num_neighbors)
        attn_mask = torch.stack([attn_mask for _ in range(self.n_heads)], dim=1)

        # If a node has no neighbors (nbr_mask all zero), setting masks to -np.inf will cause softmax nans
        # Choose a very large negative number (-1e10 following TGAT) instead
        A = A.masked_fill(attn_mask, -1e10)
        A = torch.softmax(A, dim=-1)
        A = self.dropout(A)

        O = torch.einsum('bhln,bhnd->bhld', A, V)  # (batch, n_heads, 1, head_dim)
        O = O.permute(0, 2, 1, 3).flatten(start_dim=2)  # (batch, 1, out_dim)

        out = self.W_O(O)  # (batch_size, 1, out_dim)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        out = out.squeeze(dim=1)  # (batch_size, out_dim)
        return out

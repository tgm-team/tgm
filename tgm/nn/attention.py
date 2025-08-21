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

        self.node_feat_dim = node_dim
        self.edge_feat_dim = edge_dim
        self.time_feat_dim = time_dim
        self.n_heads = n_heads
        self.out_dim = node_dim + time_dim
        self.key_dim = node_dim + edge_dim + time_dim

        if self.out_dim % n_heads != 0:
            print('warning: out_dim cannot be divided by n_heads, padding')
            self.pad_dim = n_heads - self.out_dim % n_heads
            self.out_dim += self.pad_dim
        else:
            self.pad_dim = 0

        assert self.out_dim % n_heads == 0, (
            'The sum of node_feat_dim and time_feat_dim should be divided by n_heads!'
        )

        self.head_dim = self.out_dim // n_heads
        self.query_projection = torch.nn.Linear(
            self.out_dim, n_heads * self.head_dim, bias=False
        )
        self.key_projection = torch.nn.Linear(
            self.key_dim, n_heads * self.head_dim, bias=False
        )
        self.value_projection = torch.nn.Linear(
            self.key_dim, n_heads * self.head_dim, bias=False
        )

        self.scaling_factor = self.head_dim**-0.5
        self.layer_norm = torch.nn.LayerNorm(self.out_dim)
        self.residual_fc = torch.nn.Linear(n_heads * self.head_dim, self.out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_time_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_node_time_features: torch.Tensor,
        neighbor_node_edge_features: torch.Tensor,
        neighbor_masks: np.ndarray,
    ):
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # we need to pad for the inputs
        if self.pad_dim != 0:
            node_features = torch.cat(
                [
                    node_features,
                    torch.zeros(
                        node_features.shape[0], node_features.shape[1], self.pad_dim
                    ).to(node_features.device),
                ],
                dim=2,
            )

        # Tensor, shape (batch_size, 1, out_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, n_heads, self.head_dim)
        query = self.query_projection(query).reshape(
            query.shape[0], query.shape[1], self.n_heads, self.head_dim
        )

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat(
            [
                neighbor_node_features,
                neighbor_node_edge_features,
                neighbor_node_time_features,
            ],
            dim=2,
        )
        # Tensor, shape (batch_size, num_neighbors, n_heads, self.head_dim)
        self.key_projection(key)
        key = self.key_projection(key).reshape(
            key.shape[0], key.shape[1], self.n_heads, self.head_dim
        )
        # Tensor, shape (batch_size, num_neighbors, n_heads, self.head_dim)
        value = self.value_projection(value).reshape(
            value.shape[0], value.shape[1], self.n_heads, self.head_dim
        )

        # Tensor, shape (batch_size, n_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, n_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, n_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, n_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = (
            torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        )
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.n_heads, 1, num_neighbors)
        attention_mask = torch.stack(
            [attention_mask for _ in range(self.n_heads)], dim=1
        )

        # Tensor, shape (batch_size, self.n_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, n_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, n_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, n_heads * self.head_dim), where n_heads * self.head_dim is equal to out_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, out_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, out_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, out_dim)
        output = output.squeeze(dim=1)
        return output

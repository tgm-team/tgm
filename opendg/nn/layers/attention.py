from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Multi-head Attention module."""
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert (
            self.query_dim % num_heads == 0
        ), 'The sum of node_feat_dim and time_feat_dim should be divided by num_heads!'

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(
            self.query_dim, num_heads * self.head_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.key_dim, num_heads * self.head_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.key_dim, num_heads * self.head_dim, bias=False
        )

        self.scaling_factor = self.head_dim**-0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_time_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_node_time_features: torch.Tensor,
        neighbor_node_edge_features: torch.Tensor,
        neighbor_masks: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Temporal attention forward process.

        Args:
            node_features: Tensor, shape (batch_size, node_feat_dim)
            node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
            neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
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
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(
            key.shape[0], key.shape[1], self.num_heads, self.head_dim
        )
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        )

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = (
            torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        )
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack(
            [attention_mask for _ in range(self.num_heads)], dim=1
        )

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores

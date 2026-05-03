from typing import Dict, List

import torch
import torch.nn as nn

from tgm.constants import PADDED_NODE_ID

from ..modules import TemporalAttention, Time2Vec


class MergeLayer(nn.Module):
    """Merge Layer of TGAT.

    Args:
        in_dim1 (int): Dimension of the first input tensor.
        in_dim2 (int): Dimension of the second input tensor.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer.
    """

    def __init__(self, in_dim1: int, in_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim1 + in_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MergeLayer.

        Args:
            x1 (torch.Tensor): First input tensor of shape (batch_size, in_dim1).
            x2 (torch.Tensor): Second input tensor of shape (batch_size, in_dim2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        h = self.fc1(torch.cat([x1, x2], dim=1))
        h = h.relu()
        return self.fc2(h)


class TGAT(nn.Module):
    """Temporal Graph Attention Network (TGAT).

    Args:
        node_dim (int): Dimension of node features.
        edge_dim (int): Dimension of edge features.
        time_dim (int): Dimension of time encodings.
        embed_dim (int): Dimension of hidden and output embeddings.
        num_layers (int): Number of temporal attention layers.
        n_heads (int): Number of attention heads. Defaults to 2.
        dropout (float): Dropout probability. Defaults to 0.1.

    Note:
        The node embedding dimension must be the same as the hidden embedding dimension.

    Reference: https://arxiv.org/abs/2002.07962
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """In this implementation, the node embedding dimension must be the same as hidden embedding dimension."""
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.time_encoder = Time2Vec(time_dim=time_dim)

        self.attn, self.merge_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            self.attn.append(
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=node_dim if i == 0 else embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    dropout=dropout,
                )
            )
            self.merge_layers.append(
                MergeLayer(
                    in_dim1=self.attn[-1].out_dim,
                    in_dim2=node_dim,
                    hidden_dim=embed_dim,
                    output_dim=embed_dim,
                )
            )

    def forward(
        self,
        X: torch.Tensor,
        seed_nids: List[torch.Tensor],
        seed_times: List[torch.Tensor],
        nbr_nids: List[torch.Tensor],
        nbr_edge_x: List[torch.Tensor],
        nbr_edge_time: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X (torch.Tensor): Node feature matrix of shape (num_nodes, node_dim).
            seed_nids (List[torch.Tensor]): Seed node IDs for each hop, where
                seed_nids[i] contains node IDs of shape (num_seeds,).
            nbr_nids (List[torch.Tensor]): Neighbor node IDs for each hop, where
                nbr_nids[i] contains node IDs of shape (num_seeds, num_nbrs).
            nbr_edge_x (List[torch.Tensor]): Edge features for each hop, where
                nbr_edge_x[i] contains edge features of shape (num_seeds, num_nbrs, edge_dim).
            seed_times (List[torch.Tensor]): Timestamps of seed nodes for each hop, where
                seed_times[i] contains timestamps of shape (num_seeds,).
            nbr_edge_time (List[torch.Tensor]): Timestamps of neighbor edges for each hop, where
                nbr_edge_time[i] contains timestamps of shape (num_seeds, num_nbrs).

        Returns:
            torch.Tensor: Output embeddings of seed nodes of shape (num_seeds, embed_dim).
        """
        device = X.device
        z: Dict[int, Dict[int, torch.Tensor]] = {
            j: {} for j in range(self.num_layers + 1)
        }  # z[j][i] = z of nbr^i at hop j

        # Layer 0 (leaf nodes): z[0][i] = static_node_feat
        z[0][0] = X[seed_nids[0]]
        for i in range(1, self.num_layers + 1):
            z[0][i] = X[nbr_nids[i - 1].flatten()]

        # Layers 1..H: aggregate z[j][i] = agg(z[j - 1][i], z[j - 1][i + 1])
        for j in range(1, self.num_layers + 1):
            for i in range(self.num_layers - j + 1):
                num_nodes = z[j - 1][i].size(0)
                num_nbr = nbr_nids[j - 1].shape[-1]
                out = self.attn[j - 1](
                    X=z[j - 1][i],
                    time_feat=self.time_encoder(torch.zeros(num_nodes, device=device)),
                    nbr_node_feat=z[j - 1][i + 1].reshape(num_nodes, num_nbr, -1),
                    edge_feat=nbr_edge_x[i],
                    valid_nbr_mask=nbr_nids[i] != PADDED_NODE_ID,
                    nbr_time_feat=self.time_encoder(
                        seed_times[i][:, None] - nbr_edge_time[i]
                    ),
                )
                z[j][i] = self.merge_layers[j - 1](out, z[0][i])

        return z[self.num_layers][0]

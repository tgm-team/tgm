import math
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .time_encoding import Time2Vec


class RandomProjectionModule(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_layer: int,
        time_decay_weight: float,
        beginning_time: torch.FloatType,
        use_matrix: bool = True,
        scale_random_projection: bool = True,
        enforce_dim: int | None = None,
        num_edges: int | None = None,
        dim_factor: int | None = None,
        device: str = 'cpu',
    ) -> None:
        super(RandomProjectionModule, self).__init__()
        assert (not num_edges and not dim_factor) or not enforce_dim or use_matrix

        if enforce_dim is not None:
            self.dim = enforce_dim
        elif num_edges is not None and dim_factor is not None:
            self.dim = min(int(math.log(num_edges * 2)) * dim_factor, num_nodes)
        elif use_matrix:
            self.dim = num_nodes
        else:
            raise ValueError(
                'Must provide enforce_dim or (num_edges and dim_factor) or set use_matrix to true'
            )
        self.num_layer = num_layer
        self.time_decay_weight = time_decay_weight
        self.use_matrix = use_matrix
        self.device = device
        self.scale = scale_random_projection

        self.beginning_time = nn.Parameter(
            torch.tensor(beginning_time), requires_grad=False
        ).to(device)
        self.now_time = nn.Parameter(
            torch.tensor(beginning_time), requires_grad=False
        ).to(device)
        self.random_projections = nn.ParameterList()

        if use_matrix:
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(torch.eye(self.dim), requires_grad=False)
                    )
                else:
                    self.random_projections.append(
                        nn.Parameter(
                            torch.zeros_like(self.random_projections[i - 1]),
                            requires_grad=False,
                        )
                    )
        else:
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(
                            torch.normal(
                                0, 1 / math.sqrt(self.dim), (num_nodes, self.dim)
                            ),
                            requires_grad=False,
                        )
                    )
                else:
                    self.random_projections.append(
                        nn.Parameter(
                            torch.zeros_like(self.random_projections[i - 1]),
                            requires_grad=False,
                        )
                    )

        pair_wise_feature_dim = (2 * self.num_layer + 2) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(pair_wise_feature_dim, pair_wise_feature_dim * 4),
            nn.ReLU(),
            nn.Linear(pair_wise_feature_dim * 4, pair_wise_feature_dim),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        f"""Forward pass.
        Get pairwise feature for given source nodes and destination nodes.

        Args:
            X (PyTorch Float Tensor): Node features.
            src (PyTorch Tensor): Source node IDs
            dst (PyTorch Tensor): Destination node IDs.

        Returns:
            H_pairwise : Pairwise feature
        """
        src_random_projections = self.get_random_projections(src)
        dst_random_projections = self.get_random_projections(dst)
        random_projections = torch.cat(
            [src_random_projections, dst_random_projections], dim=1
        ).to(self.device)
        random_feature = torch.matmul(
            random_projections, random_projections.transpose(1, 2)
        ).reshape(src.shape[0], -1)

        if not self.scale:
            out = self.mlp(random_feature)
        else:
            random_feature[random_feature < 0] = 0
            random_feature = torch.log(random_feature + 1.0)
            out = self.mlp(random_feature)
        return out

    def update(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        time: torch.Tensor,
    ) -> None:
        f"""Updating the temporal walk matrices after observing a batch of interactions

        Args:
            src (PyTorch Tensor): Source node IDs
            dst (PyTorch Tensor): Destination node IDs.
            time (PyTorch Tensor): edge event time

        Returns:
        """
        next_time = time[-1]
        time_weight = torch.exp(-self.time_decay_weight * (next_time - time))[:, None]

        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = self.random_projections[
                i
            ].data * np.power(
                np.exp(
                    -self.time_decay_weight * (next_time - self.now_time.cpu().numpy())
                ),
                i,
            )

        for i in range(self.num_layer, 0, -1):
            src_update_messages = self.random_projections[i - 1][dst] * time_weight
            dst_update_messages = self.random_projections[i - 1][src] * time_weight
            self.random_projections[i].scatter_add_(
                dim=0, index=src[:, None].expand(-1, self.dim), src=src_update_messages
            )
            self.random_projections[i].scatter_add_(
                dim=0, index=dst[:, None].expand(-1, self.dim), src=dst_update_messages
            )

        # set current timestamp to the biggest timestamp in this batch
        self.now_time.data = torch.tensor(next_time, device=self.device)

    def get_random_projections(self, node_ids: torch.Tensor) -> torch.Tensor:
        f"""Get the random projections of the give node ids

        Args:
            node_ids (PyTorch Tensor): List of nodes.

        Returns:
            Random projection of nodes
        """
        random_projections = []
        for i in range(self.num_layer + 1):
            random_projections.append(self.random_projections[i][node_ids])
        return torch.stack(random_projections, dim=1).to(self.device)

    def reset_random_projections(
        self,
        reset_zero: bool = True,
    ) -> None:
        f"""Get the random projections of the give node ids

        Args:
            reset_zero (bool): whether reset temporal walk matrices to zero

        Returns:
        """
        for i in range(1, self.num_layer + 1):
            nn.init.zeros_(self.random_projections[i])
        self.now_time.data = self.beginning_time.clone()
        if not self.use_matrix and reset_zero:
            nn.init.normal_(
                self.random_projections[0], mean=0, std=1 / math.sqrt(self.dim)
            )

    def backup_random_projections(self) -> Tuple[torch.Tensor, List]:
        return self.now_time.clone(), [
            self.random_projections[i].clone() for i in range(1, self.num_layer + 1)
        ]

    def reload_random_projections(self, random_projections: Tuple) -> None:
        f"""Reload the random projections

        Args:
            random_projections (Tuple): tuple of (now_time,random_projections)

        Returns:
        """
        assert (
            len(random_projections) == 2
        ), (
            'Expect a tuple of (now_time,random_projections)'
        )  # @TODO: Need to raise custom exception
        now_time, random_projections = random_projections
        assert (
            torch.is_tensor(now_time) and len(random_projections) == self.num_layer
        ), (
            'Not a valid state of random projection'
        )  # @TODO: Need to raise custom exception

        self.now_time.data = now_time.clone()
        for i in range(1, self.num_layer + 1):
            assert torch.is_tensor(
                random_projections[i - 1]
            ), (
                'Not a valid state of random projection'
            )  # @TODO: Need to raise custom exception
            self.random_projections[i].data = random_projections[i - 1].clone()


class FeedForwardNet(nn.Module):
    def __init__(
        self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super(FeedForwardNet, self).__init__()
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
        raise self.ffn(X)


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        token_dim_expansion_factor: float = 0.5,
        channel_dim_expansion_factor: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super(MLPMixer, self).__init__()

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


class TPNet(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        output_dim: int,
        num_nodes: int,
        dropout: float,
        num_layers: int,
        not_embedding: bool = False,
        device: str = 'cpu',
        time_encoder: Callable[..., nn.Module] = Time2Vec,
    ) -> None:
        super(TPNet, self).__init__()
        # @TODO
        raise Exception('Not yet implement')

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        neighbours: torch.Tensor,
        neighbours_time: torch.Tensor,
        neighbours_edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f"""Forward pass.

        Args:
            X (PyTorch Float Tensor): Node features.
            edge_index (PyTorch Tensor): Graph edge indices.
            edge_time (PyTorch Tensor): Edge time vector.
            neighbours (PyTorch Tensor): Neighbours of src and dst nodes from edge_index
            neighbours_time (PyTorch Tensor): Interaction time of src/dst nodes and their neighbours
            neighbours_edge_feat (PyTorch Tensor): Features of edge between src/dst nodes and their neighbours

        Returns:
            H_source,H_dest (PyTorch Float Tensor): Time-aware representations of src and dst nodes.

        *Note: Information of about neighbours of both src and dst nodes are concated together. Neighbour information of all src nodes comes first, then that of all dst nodes*
        """
        # @TODO
        raise Exception('Not yet implement')

    def update_random_projection(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> None:
        f"""Update the random projections of temporal walk matrices after observing positive links

        Args:
            edge_index (PyTorch Tensor): Graph edge indices.
            edge_time (PyTorch Tensor): Edge time vector.
        """
        # @TODO: This need to call update from RandomProjectionModule
        raise Exception('Not yet implemented')

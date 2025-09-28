import math
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from tgm.constants import PADDED_NODE_ID

from ..time_encoding import Time2Vec


class RandomProjectionModule(nn.Module):
    r"""This model maintains a series of temporal walk matrices $A_^(0)(t),A_^(1)(t),...,A^(k)(t)$ through
    random feature propagation, and extract the pairwise features from the obtained random projections.

    Args:
        num_nodes(int): the number of nodes
        num_layer(int): the max hop of the maintained temporal walk matrices
        time_decay_weight(float): the time decay weight (lambda of the original paper)
        beginning_time(float): the earliest time in the given temporal graph
        use_matrix(bool): if True, explicitly maintain the temporal walk matrices
        scale_random_projection(bool) if True, the inner product of nodes' random projections will be scaled
        enforce_dim(int) if not None, explicitly set the dimension of random projections to enforce_dim
        num_edges(int): the number of edges
        dim_factor(int): the parameter to control the dimension of random projections. Specifically, the
                           dimension of the random projections is set to be dim_factor * log(2*edge_num)
        device(str): torch device

    *Note: For large-scale dataset, the authors suggested to set `use_matrix=False` and use number of edge and `dim_factor=10` to make it scalable.*
    """

    def __init__(
        self,
        num_nodes: int,
        num_layer: int,
        time_decay_weight: float,
        beginning_time: float,
        use_matrix: bool = True,
        scale_random_projection: bool = True,
        enforce_dim: int | None = None,
        num_edges: int | None = None,
        dim_factor: int | None = None,
        device: str = 'cpu',
    ) -> None:
        super(RandomProjectionModule, self).__init__()
        if not use_matrix:
            if enforce_dim is not None:
                self.dim = enforce_dim
            elif num_edges is not None and dim_factor is not None:
                self.dim = min(int(math.log(num_edges * 2)) * dim_factor, num_nodes)
            else:
                raise ValueError(
                    'When `use_matrix` is False, either providing enforce_dim or both num_edges and dim_factor'
                )
        else:
            self.dim = num_nodes
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

        self.out_dim = (2 * self.num_layer + 2) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 4),
            nn.ReLU(),
            nn.Linear(self.out_dim * 4, self.out_dim),
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
        random_projections = torch.cat(  # @TODO: This takes up a lot GPU memory, especially for TGB evaluation
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
        next_time = time[-1].unsqueeze(0)
        time_weight = torch.exp(-self.time_decay_weight * (next_time - time))[:, None]

        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = self.random_projections[
                i
            ].data * torch.pow(
                torch.exp(-self.time_decay_weight * (next_time - self.now_time)),
                i,
            )

        for i in range(self.num_layer, 0, -1):
            src_update_messages = self.random_projections[i - 1][dst] * time_weight
            dst_update_messages = self.random_projections[i - 1][src] * time_weight
            self.random_projections[i].scatter_add_(
                dim=0,
                index=src[:, None].expand(-1, self.dim).long(),
                src=src_update_messages,
            )
            self.random_projections[i].scatter_add_(
                dim=0,
                index=dst[:, None].expand(-1, self.dim).long(),
                src=dst_update_messages,
            )

        # set current timestamp to the biggest timestamp in this batch
        self.now_time.data = next_time.clone().detach()

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
    r"""An implementation of TPNet.

    Args:
        node_feat_dim (int): Dimension of static/dynamic node features (`d_N`).
        edge_feat_dim (int): Dimension of edge features (`d_E`).
        time_feat_dim (int): Dimension of time encodings (`d_T`).
        channel_embedding_dim (int): Dimension of each channel embedding.
        output_dim (int): Dimension of output embedding.
        dropout (float): Drop out rate.
        num_layers (int): Number of transformer layers.
        num_neighbors (int): Number of recent temporal neighbors consider
        random_projections (nn.Module): Random projection module that maintains a series temporal walk matrices
        device (str) : cpu or cuda
        time_encoder (PyTorch Module) : Time encoder module.


    Reference: https://arxiv.org/abs/2410.04013.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        num_neighbors: int,
        random_projections: RandomProjectionModule | None = None,
        device: str = 'cpu',
        time_encoder: Callable[..., nn.Module] = Time2Vec,
    ) -> None:
        super(TPNet, self).__init__()
        self.device = device
        self.time_encoder = time_encoder(time_feat_dim).to(device)
        self.random_projections = random_projections
        self.num_neighbors = num_neighbors
        if self.random_projections is None:
            self.random_feature_dim = 0
        else:
            self.random_feature_dim = self.random_projections.out_dim * 2

        self.projection_layer = nn.Sequential(
            nn.Linear(
                node_feat_dim + edge_feat_dim + time_feat_dim + self.random_feature_dim,
                output_dim * 2,
            ),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        ).to(device)
        self.mlp_mixers = nn.ModuleList(
            [
                MLPMixer(
                    num_tokens=num_neighbors,
                    num_channels=output_dim,
                    token_dim_expansion_factor=0.5,
                    channel_dim_expansion_factor=4.0,
                    dropout=dropout,
                ).to(device)
                for _ in range(num_layers)
            ]
        )

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
        src, dst = edge_index[0], edge_index[1]
        node_ids = torch.cat([src, dst], dim=0)
        num_src = src.shape[0]

        src = src.repeat(2)
        dst = dst.repeat(2)
        edge_time = edge_time.repeat(2)
        neighbor_node_features = X[neighbours, :]
        neighbor_node_features[neighbours == PADDED_NODE_ID] = 0

        neighbours_time_feats = self.time_encoder(
            torch.log((edge_time.unsqueeze(1) - neighbours_time) + 1)
        )
        neighbours_time_feats[(neighbours == PADDED_NODE_ID)] = 0

        if self.random_projections is not None:
            concat_neighbor_random_features = self.random_projections(
                src=neighbours.reshape(-1).repeat(2),
                dst=torch.cat(
                    [
                        src.repeat_interleave(self.num_neighbors),
                        dst.repeat_interleave(self.num_neighbors),
                    ],
                    dim=0,
                ),
            )

            neighbor_random_features = torch.cat(
                [
                    concat_neighbor_random_features[
                        : len(node_ids) * self.num_neighbors
                    ],
                    concat_neighbor_random_features[
                        len(node_ids) * self.num_neighbors :
                    ],
                ],
                dim=1,
            ).reshape(len(node_ids), self.num_neighbors, -1)
            neighbor_combine_features = torch.cat(
                [
                    neighbor_node_features,
                    neighbours_time_feats,
                    neighbours_edge_feat,
                    neighbor_random_features,
                ],
                dim=2,
            )
        else:
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbours_time_feats, neighbours_edge_feat],
                dim=2,
            )

        embeddings = self.projection_layer(neighbor_combine_features)
        embeddings.masked_fill(
            (neighbours == PADDED_NODE_ID)[:, :, None].to(self.device), 0
        )
        for mlp_mixer in self.mlp_mixers:
            embeddings = mlp_mixer(embeddings)
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings[:num_src], embeddings[num_src:]

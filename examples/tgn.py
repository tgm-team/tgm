from collections import defaultdict

import numpy as np
import torch
from torch import nn

from opendg.nn import TemporalAttention, Time2Vec


class TGN(torch.nn.Module):
    def __init__(
        self,
        neighbor_finder,
        node_features,
        edge_features,
        num_layers=2,
        n_heads=2,
        dropout=0.1,
        message_dimension=100,
        memory_dimension=500,
        n_neighbors=None,
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.neighbor_finder = neighbor_finder
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32))
        self.node_dim = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.edge_dim = self.edge_raw_features.shape[1]
        self.embed_dim = self.node_dim
        self.n_neighbors = n_neighbors
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.time_encoder = Time2Vec(time_dim=self.node_dim)
        self.memory = None
        self.memory_dimension = memory_dimension

        raw_message_dimension = (
            2 * self.memory_dimension + self.edge_dim + self.time_encoder.dimension
        )
        message_dimension = raw_message_dimension
        self.memory = Memory(
            n_nodes=self.n_nodes,
            memory_dimension=self.memory_dimension,
            input_dimension=message_dimension,
            message_dimension=message_dimension,
        )
        self.message_aggregator = LastMessageAggregator()
        self.message_function = IdentityMessageFunction(
            raw_message_dimension=raw_message_dimension,
            message_dimension=message_dimension,
        )
        self.memory_updater = GRUMemoryUpdater(
            self.memory, message_dimension, self.memory_dimension
        )
        self.embedding_module = GraphAttentionEmbedding(
            node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            num_layers=self.num_layers,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            time_dim=self.node_dim,
            embed_dim=self.embed_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.link_predictor = LinkPredictor(self.node_dim)

    def compute_temporal_embeddings(
        self,
        src_ndoes,
        dst_nodes,
        neg_nodes,
        edge_times,
        edge_idxs,
        n_neighbors=20,
    ):
        n_samples = len(src_ndoes)
        nodes = np.concatenate([src_ndoes, dst_nodes, neg_nodes])
        positives = np.concatenate([src_ndoes, dst_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None
        # Update memory for all nodes with messages stored in previous batches
        memory, last_update = self.get_updated_memory(
            list(range(self.n_nodes)), self.memory.messages
        )

        ### Compute differences between the time the memory of a node was last updated,
        ### and the time for which we want to compute the embedding of a node
        src_time_diffs = torch.LongTensor(edge_times) - last_update[src_ndoes].long()
        dst_time_diffs = torch.LongTensor(edge_times) - last_update[dst_nodes].long()
        neg_time_diffs = torch.LongTensor(edge_times) - last_update[neg_nodes].long()

        time_diffs = torch.cat([src_time_diffs, dst_time_diffs, neg_time_diffs], dim=0)

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            src_ndoes=nodes,
            timestamps=timestamps,
            num_layers=self.num_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs,
        )
        z_src = node_embedding[:n_samples]
        z_dst = node_embedding[n_samples : 2 * n_samples]
        z_neg = node_embedding[2 * n_samples :]

        if self.memory_update_at_start:
            # Persist the updates to the memory only for sources and destinations
            self.update_memory(positives, self.memory.messages)
            assert torch.allclose(
                memory[positives], self.memory.get_memory(positives), atol=1e-5
            ), 'Something wrong in how the memory was updated'

            # Remove messages for the positives since we have already updated the memory using them
            self.memory.clear_messages(positives)

        unique_src, src_id_to_messages = self.get_raw_messages(
            src_ndoes,
            z_src,
            dst_nodes,
            z_dst,
            edge_times,
            edge_idxs,
        )
        unique_dst, dst_id_to_messages = self.get_raw_messages(
            dst_nodes,
            z_dst,
            src_ndoes,
            z_src,
            edge_times,
            edge_idxs,
        )
        self.memory.store_raw_messages(unique_src, src_id_to_messages)
        self.memory.store_raw_messages(unique_dst, dst_id_to_messages)
        return z_src, z_dst, z_neg

    def compute_edge_probabilities(
        self,
        src_ndoes,
        dst_nodes,
        neg_nodes,
        edge_times,
        edge_idxs,
        n_neighbors=20,
    ):
        n_samples = len(src_ndoes)
        z_src, z_dst, z_neg = self.compute_temporal_embeddings(
            src_ndoes,
            dst_nodes,
            neg_nodes,
            edge_times,
            edge_idxs,
            n_neighbors,
        )
        score = self.link_predictor(
            torch.cat([z_src, z_src], dim=0),
            torch.cat([z_dst, z_neg]),
        ).squeeze(dim=0)
        pos_out = score[:n_samples].sigmoid()
        neg_out = score[n_samples:].sigmoid()
        return pos_out, neg_out

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_msg, unique_timestamps = self.message_aggregator.aggregate(
            nodes, messages
        )
        if len(unique_nodes) > 0:
            unique_msg = self.message_function.compute_message(unique_msg)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(
            unique_nodes, unique_msg, timestamps=unique_timestamps
        )

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_msg, unique_timestamps = self.message_aggregator.aggregate(
            nodes, messages
        )
        if len(unique_nodes) > 0:
            unique_msg = self.message_function.compute_message(unique_msg)
        return self.memory_updater.get_updated_memory(
            unique_nodes, unique_msg, timestamps=unique_timestamps
        )

    def get_raw_messages(
        self,
        src_ndoes,
        z_src,
        dst_nodes,
        z_dst,
        edge_times,
        edge_idxs,
    ):
        edge_times = torch.from_numpy(edge_times).float()
        edge_features = self.edge_raw_features[edge_idxs]
        src_memory = (
            self.memory.get_memory(src_ndoes)
            if not self.use_source_embedding_in_message
            else z_src
        )
        dst_memory = (
            self.memory.get_memory(dst_nodes)
            if not self.use_destination_embedding_in_message
            else z_dst
        )
        src_time_delta = edge_times - self.memory.last_update[src_ndoes]
        src_time_delta_encoding = self.time_encoder(
            src_time_delta.unsqueeze(dim=1)
        ).view(len(src_ndoes), -1)

        src_message = torch.cat(
            [
                src_memory,
                dst_memory,
                edge_features,
                src_time_delta_encoding,
            ],
            dim=1,
        )
        messages = defaultdict(list)
        unique_src = np.unique(src_ndoes)
        for i in range(len(src_ndoes)):
            messages[src_ndoes[i]].append((src_message[i], edge_times[i]))
        return unique_src, messages


class LastMessageAggregator(torch.nn.Module):
    def aggregate(self, node_ids, messages):
        unique_node_ids = np.unique(node_ids)
        unique_msg, unique_timestamps, to_update_node_ids = [], [], []
        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_msg.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])
        unique_msg = torch.stack(unique_msg) if len(to_update_node_ids) > 0 else []
        unique_timestamps = (
            torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
        )
        return to_update_node_ids, unique_msg, unique_timestamps


class IdentityMessageFunction(nn.Module):
    def compute_message(self, raw_messages):
        return raw_messages


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        neighbor_finder,
        time_encoder,
        num_layers,
        node_dim,
        edge_dim,
        time_dim,
        embed_dim,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=node_dim if i == 0 else embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    out_dim=embed_dim,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def compute_embedding(
        self,
        memory,
        src_ndoes,
        timestamps,
        num_layers,
        n_neighbors=20,
    ):
        assert num_layers >= 0

        source_nodes_torch = torch.from_numpy(src_ndoes).long()
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float(), dim=1)

        # query node always has the start time -> time span == 0
        src_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))
        src_node_features = self.node_features[source_nodes_torch, :]
        src_node_features = memory[src_ndoes, :] + src_node_features

        if num_layers == 0:
            return src_node_features
        src_node_features = self.compute_embedding(
            memory,
            src_ndoes,
            timestamps,
            num_layers=num_layers - 1,
            n_neighbors=n_neighbors,
        )

        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
            src_ndoes, timestamps, n_neighbors=n_neighbors
        )

        neighbors_torch = torch.from_numpy(neighbors).long()
        edge_idxs = torch.from_numpy(edge_idxs).long()
        edge_deltas = timestamps[:, np.newaxis] - edge_times
        edge_deltas_torch = torch.from_numpy(edge_deltas).float()
        neighbors = neighbors.flatten()
        neighbor_embeddings = self.compute_embedding(
            memory,
            neighbors,
            np.repeat(timestamps, n_neighbors),
            num_layers=num_layers - 1,
            n_neighbors=n_neighbors,
        )

        effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbor_embeddings = neighbor_embeddings.view(
            len(src_ndoes), effective_n_neighbors, -1
        )
        edge_time_embeddings = self.time_encoder(edge_deltas_torch)
        edge_features = self.edge_features[edge_idxs, :]
        mask = neighbors_torch == 0
        return self.attn[num_layers - 1](
            src_node_features,
            src_nodes_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            edge_features,
            mask,
        )


class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory, message_dimension, memory_dimension):
        super().__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.memory_updater = nn.GRUCell(
            input_size=message_dimension, hidden_size=memory_dimension
        )

    def update_memory(self, unique_node_ids, unique_msg, timestamps):
        if len(unique_node_ids) <= 0:
            return
        assert (
            (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item()
        ), 'Trying to update memory to time in the past'
        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps
        updated_memory = self.memory_updater(unique_msg, memory)
        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_msg, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        assert (
            (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item()
        ), 'Trying to update memory to time in the past'
        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(
            unique_msg, updated_memory[unique_node_ids]
        )
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps
        return updated_memory, updated_last_update


class Memory(nn.Module):
    def __init__(
        self,
        n_nodes,
        memory_dimension,
        input_dimension,
        message_dimension=None,
        combination_method='sum',
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.combination_method = combination_method
        self.__init_memory__()

    def __init_memory__(self):
        # Treat memory as parameter so that it is saved and loaded together with the model
        self.memory = nn.Parameter(
            torch.zeros((self.n_nodes, self.memory_dimension)),
            requires_grad=False,
        )
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes), requires_grad=False)
        self.messages = defaultdict(list)

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = (
            memory_backup[0].clone(),
            memory_backup[1].clone(),
        )
        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_()
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))
            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_out = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_out(h).sigmoid().view(-1)

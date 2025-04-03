from collections import defaultdict

import numpy as np
import torch
from torch import nn

from opendg.nn import TemporalAttention, Time2Vec


class TGN(torch.nn.Module):
    def __init__(self, node_feats, edge_feats, num_layers=2, memory_dim=500):
        super().__init__()
        self.num_layers = num_layers
        self.node_raw_features = torch.from_numpy(node_feats.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_feats.astype(np.float32))
        self.node_dim = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.edge_dim = self.edge_raw_features.shape[1]
        self.embed_dim = self.node_dim
        self.time_encoder = Time2Vec(time_dim=self.embed_dim)
        self.memory_dim = memory_dim
        self.msg_agg = LastMessageAggregator()
        self.msg_func = IdentityMessageFunction()

        msg_dim = 2 * self.memory_dim + self.edge_dim + self.time_encoder.time_dim
        self.memory = Memory(n_nodes=self.n_nodes, memory_dim=self.memory_dim)
        self.memory_updater = GRUMemoryUpdater(self.memory, msg_dim, self.memory_dim)
        self.gat = GraphAttentionEmbedding(
            node_feats=self.node_raw_features,
            edge_feats=self.edge_raw_features,
            time_encoder=self.time_encoder,
            num_layers=self.num_layers,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            time_dim=self.node_dim,
            embed_dim=self.embed_dim,
        )

    def forward(self, src, dst, neg, time, edge_idxs):
        unique_nodes, unique_msg, unique_times = self.msg_agg(
            list(range(self.n_nodes)), self.memory.msgs
        )
        if len(unique_nodes) > 0:
            unique_msg = self.msg_func(unique_msg)
        memory, last_update = self.memory_updater.get_updated_memory(
            unique_nodes, unique_msg, unique_times
        )

        src_time_diff = torch.LongTensor(time) - last_update[src].long()
        dst_time_diff = torch.LongTensor(time) - last_update[dst].long()
        neg_time_diff = torch.LongTensor(time) - last_update[neg].long()

        pos_out, neg_out = self.gat(
            memory=memory,
            src=np.concatenate([src, dst, neg]),
            time=np.concatenate([time, time, time]),
            time_diffs=torch.cat([src_time_diff, dst_time_diff, neg_time_diff], dim=0),
        )

        # Persist the updates to the memory only for sources and destinations
        pos = np.concatenate([src, dst])
        unique_nodes, unique_msg, unique_times = self.msg_agg(pos, self.memory.msg)
        if len(unique_nodes) > 0:
            unique_msg = self.msg_func(unique_msg)
        self.memory_updater.update(unique_nodes, unique_msg, time=unique_times)
        assert torch.allclose(
            memory[pos], self.memory.get_memory(pos), atol=1e-5
        ), 'Something wrong in how the memory was updated'
        # Remove msgs for the pos since we have already updated the memory using them
        self.memory.clear_msgs(pos)
        unique_src, src_to_msgs = self._get_raw_msgs(src, dst, time, edge_idxs)
        unique_dst, dst_to_msgs = self._get_raw_msgs(dst, src, time, edge_idxs)
        self.memory.store_raw_msgs(unique_src, src_to_msgs)
        self.memory.store_raw_msgs(unique_dst, dst_to_msgs)

        return pos_out, neg_out

    def _get_raw_msgs(self, src, dst, time, edge_idxs):
        time = torch.from_numpy(time).float()
        edge_feats = self.edge_raw_features[edge_idxs]
        src_memory = self.memory.get_memory(src)
        dst_memory = self.memory.get_memory(dst)
        time_delta = time - self.memory.last_update[src]
        time_feat = self.time_encoder(time_delta.unsqueeze(dim=1)).view(len(src), -1)

        src_msg = torch.cat([src_memory, dst_memory, edge_feats, time_feat], dim=1)
        msgs = defaultdict(list)
        unique_src = np.unique(src)
        for i in range(len(src)):
            msgs[src[i]].append((src_msg[i], time[i]))
        return unique_src, msgs


class LastMessageAggregator(torch.nn.Module):
    def forward(self, node_ids, msgs):
        unique_node_ids = np.unique(node_ids)
        unique_msg, unique_times, to_update_node_ids = [], [], []
        for node_id in unique_node_ids:
            if len(msgs[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_msg.append(msgs[node_id][-1][0])
                unique_times.append(msgs[node_id][-1][1])
        unique_msg = torch.stack(unique_msg) if len(to_update_node_ids) > 0 else []
        unique_times = torch.stack(unique_times) if len(to_update_node_ids) > 0 else []
        return to_update_node_ids, unique_msg, unique_times


class IdentityMessageFunction(nn.Module):
    def forward(self, raw_messages):
        return raw_messages


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
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
        self.node_feats, self.edge_feats = node_feats, edge_feats
        self.time_encoder = time_encoder
        self.link_predictor = LinkPredictor(self.node_dim)
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

    def forward(self, memory, src, time, n_nbrs=20):
        # TODO: Go back to recursive embedding for multi-hop
        hop = 0
        src_torch = torch.from_numpy(src).long()
        node_feat = memory[src, :] + self.node_feats[src_torch, :]
        time_torch = torch.unsqueeze(torch.from_numpy(time).float(), dim=1)
        time_feat = self.time_encoder(torch.zeros_like(time_torch))
        nbrs, edge_idxs, time = self.nbr_finder.get_nbrs(src, time, n_nbrs)
        edge_idxs = torch.from_numpy(edge_idxs).long()
        time_delta = torch.from_numpy(time[:, None] - time).float()

        nbrs_torch = torch.from_numpy(nbrs).long()
        nbr_feat = memory[nbrs, :] + self.node_feats[nbrs_torch, :]
        nbr_feat = nbr_feat.view(len(src), n_nbrs, -1)
        z = self.attn[hop](
            node_feat=node_feat,
            time_feat=time_feat,
            nbr_node_feat=nbr_feat,
            nbr_time_feat=self.time_encoder(time_delta),
            edge_feat=self.edge_feats[edge_idxs, :],
            nbr_mask=nbrs_torch == 0,
        )
        z_src, z_dst, z_neg = z.chunk(3, dim=0)
        pos_out = self.link_predictor(z_src, z_dst)
        neg_out = self.link_predictor(z_src, z_neg)
        return pos_out, neg_out


class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory, msg_dim, memory_dim):
        super().__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dim)
        self.msg_dim = msg_dim
        self.memory_updater = nn.GRUCell(input_size=msg_dim, hidden_size=memory_dim)

    def update(self, unique_node_ids, unique_msg, time):
        if len(unique_node_ids) <= 0:
            return
        assert (
            (self.memory.get_last_update(unique_node_ids) <= time).all().item()
        ), 'Trying to update memory to time in the past'
        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = time
        updated_memory = self.memory_updater(unique_msg, memory)
        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_msg, time):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        assert (
            (self.memory.get_last_update(unique_node_ids) <= time).all().item()
        ), 'Trying to update memory to time in the past'
        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(
            unique_msg, updated_memory[unique_node_ids]
        )
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = time
        return updated_memory, updated_last_update


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dim):
        super().__init__()
        self.memory = nn.Parameter(
            torch.zeros((n_nodes, memory_dim)), requires_grad=False
        )
        self.last_update = nn.Parameter(torch.zeros(n_nodes), requires_grad=False)
        self.msgs = defaultdict(list)

    def store_raw_msgs(self, nodes, node_id_to_msg):
        for node in nodes:
            self.msgs[node].extend(node_id_to_msg[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        msgs_clone = {}
        for k, v in self.msgs.items():
            msgs_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        return self.memory.data.clone(), self.last_update.data.clone(), msgs_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = (
            memory_backup[0].clone(),
            memory_backup[1].clone(),
        )
        self.msgs = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.msgs[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_()
        for k, v in self.msgs.items():
            new_node_msg = []
            for msg in v:
                new_node_msg.append((msg[0].detach(), msg[1]))
            self.msgs[k] = new_node_msg

    def clear_msgs(self, nodes):
        for node in nodes:
            self.msgs[node] = []


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

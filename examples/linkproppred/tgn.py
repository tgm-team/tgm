import argparse
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph, RecipeRegistry
from tgm.constants import (
    METRIC_TGB_LINKPROPPRED,
    PADDED_NODE_ID,
    RECIPE_TGB_LINK_PRED,
)
from tgm.hooks import NeighborSamplerHook, RecencyNeighborHook
from tgm.loader import DGDataLoader
from tgm.nn import TemporalAttention, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGN LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
parser.add_argument(
    '--n-nbrs',
    type=int,
    nargs='+',
    default=[10],
    help='num sampled nbrs at each hop',
)
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=172, help='attention dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['uniform', 'recency'],
    help='sampling strategy',
)
parser.add_argument(
    '--capture-gpu', action=argparse.BooleanOptionalAction, help='record peak gpu usage'
)
parser.add_argument(
    '--capture-cprofile', action=argparse.BooleanOptionalAction, help='record cprofiler'
)


class MergeLayer(nn.Module):
    def __init__(self, in_dim1: int, in_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim1 + in_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.fc1(torch.cat([x1, x2], dim=1))
        h = h.relu()
        return self.fc2(h)


class TGN(torch.nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        num_nodes: int,
    ):
        super().__init__()
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.msg_agg = LastMessageAggregator()
        self.nodes = list(range(num_nodes))

        msg_dim = 2 * embed_dim + edge_dim + time_dim
        self.memory = Memory(n_nodes=num_nodes, memory_dim=embed_dim)
        self.memory_updater = GRUMemoryUpdater(self.memory, msg_dim, embed_dim)
        self.gat = GraphAttentionEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self, batch: DGBatch, static_node_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agg_msgs = self.msg_agg(self.nodes, self.memory.msgs)
        memory, _ = self.memory_updater.get_updated_memory(*agg_msgs)
        # TODO: I think this is only needed for multi-hop?
        # Difference between the time the memory of a node was last updated,
        # and the time for which we want to compute the embedding of a node
        # batch.time[batch.src] -= last_update[batch.src].long()
        # batch.time[batch.dst] -= last_update[batch.dst].long()
        # batch.time[batch.neg] -= last_update[batch.neg].long()

        z = self.gat(batch, static_node_feats, memory=memory)
        self._update_memory(batch)
        return z

    def _update_memory(self, batch: DGBatch) -> None:
        device = batch.src.device

        def _get_raw_msgs(src, dst, time):
            edge_feats = batch.edge_feats
            src_memory = self.memory.get_memory(src)
            dst_memory = self.memory.get_memory(dst)
            time_delta = time.to(device) - self.memory.last_update[src]
            time_feat = self.time_encoder(time_delta.unsqueeze(dim=1)).view(
                len(src), -1
            )

            src_msg = torch.cat([src_memory, dst_memory, edge_feats, time_feat], dim=1)
            msgs = defaultdict(list)
            unique_src = np.unique(src.cpu())
            for i in range(len(src)):
                msgs[src[i]].append((src_msg[i], time[i]))
            return unique_src, msgs

        # Persist the updates to the memory only for sources and destinations
        pos = torch.cat([batch.src, batch.dst]).cpu().numpy()
        agg_msgs = self.msg_agg(pos, self.memory.msgs)
        self.memory_updater.update(*agg_msgs)

        # Remove msgs for the pos since we have already updated the memory using them
        self.memory.clear_msgs(pos)
        unique_src, src_to_msgs = _get_raw_msgs(batch.src, batch.dst, batch.time)
        unique_dst, dst_to_msgs = _get_raw_msgs(batch.dst, batch.src, batch.time)
        self.memory.store_raw_msgs(unique_src, src_to_msgs)
        self.memory.store_raw_msgs(unique_dst, dst_to_msgs)


class LastMessageAggregator(torch.nn.Module):
    def forward(
        self, node_ids: torch.Tensor, msgs: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        unique_nids = np.unique(node_ids)
        unique_msg, unique_times, to_update_node_ids = [], [], []
        for node_id in unique_nids:
            if len(msgs[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_msg.append(msgs[node_id][-1][0])
                unique_times.append(msgs[node_id][-1][1])
        unique_msg = torch.stack(unique_msg) if len(to_update_node_ids) > 0 else []
        unique_times = torch.stack(unique_times) if len(to_update_node_ids) > 0 else []
        return to_update_node_ids, unique_msg, unique_times


class Memory(nn.Module):
    def __init__(self, n_nodes: int, memory_dim: int, device: str = 'cpu') -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.n_nodes = n_nodes
        self.device = device
        self.msgs = defaultdict(list)

        self.reset()

    def set_device(self, device: str) -> None:
        self.device = device

    def reset(self) -> None:
        self.memory = nn.Parameter(
            torch.zeros((self.n_nodes, self.memory_dim), device=self.device),
            requires_grad=False,
        )
        self.last_update = nn.Parameter(
            torch.zeros(self.n_nodes, device=self.device), requires_grad=False
        )

    def store_raw_msgs(
        self, nodes: torch.Tensor, node_id_to_msg: Dict[int, torch.Tensor]
    ) -> None:
        for node in nodes:
            self.msgs[node].extend(node_id_to_msg[node])

    def get_memory(self, node_idxs: torch.Tensor) -> torch.Tensor:
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs: torch.Tensor, values: torch.Tensor) -> None:
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs: torch.Tensor) -> torch.Tensor:
        return self.last_update[node_idxs]

    def backup_memory(self) -> Tuple[torch.Tensor, ...]:
        msgs_clone = {}
        for k, v in self.msgs.items():
            msgs_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        return self.memory.data.clone(), self.last_update.data.clone(), msgs_clone

    def restore_memory(self, memory_backup: Tuple[torch.Tensor, ...]) -> None:
        self.memory.data, self.last_update.data = (
            memory_backup[0].clone(),
            memory_backup[1].clone(),
        )
        self.msgs = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.msgs[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self) -> None:
        self.memory.detach_()
        for k, v in self.msgs.items():
            new_node_msg = []
            for msg in v:
                new_node_msg.append((msg[0].detach(), msg[1]))
            self.msgs[k] = new_node_msg

    def clear_msgs(self, nodes: torch.Tensor) -> None:
        for node in nodes:
            self.msgs[node] = []


class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory: Memory, msg_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.memory = memory
        self.memory_updater = nn.GRUCell(input_size=msg_dim, hidden_size=memory_dim)

    def update(
        self, unique_nids: torch.Tensor, unique_msg: torch.Tensor, time: int
    ) -> None:
        if len(unique_nids) <= 0:
            return
        memory = self.memory.get_memory(unique_nids)
        self.memory.last_update[unique_nids] = time
        updated_memory = self.memory_updater(unique_msg, memory)
        self.memory.set_memory(unique_nids, updated_memory)

    def get_updated_memory(
        self, unique_nids: torch.Tensor, unique_msg: torch.Tensor, time: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(unique_nids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_nids] = self.memory_updater(
            unique_msg, updated_memory[unique_nids]
        )
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_nids] = time
        return updated_memory, updated_last_update


class GraphAttentionEmbedding(nn.Module):
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
        self, batch: DGBatch, static_node_feat: torch.Tensor, memory: Memory
    ) -> torch.Tensor:
        device = batch.src.device
        z = {j: {} for j in range(self.num_layers + 1)}  # z[j][i] = z of nbr^i at hop j

        # Layer 0 (leaf nodes): z[0][i] = static_node_feat
        z[0][0] = static_node_feat[batch.nids[0]]
        z[0][0] += memory[batch.nids[0]]
        for i in range(1, self.num_layers + 1):
            z[0][i] = static_node_feat[batch.nbr_nids[i - 1].flatten()]
            z[0][i] += memory[batch.nbr_nids[i - 1].flatten()]

        # Layers 1..H: aggregate z[j][i] = agg(z[j - 1][i], z[j - 1][i + 1])
        for j in range(1, self.num_layers + 1):
            for i in range(self.num_layers - j + 1):
                num_nodes = z[j - 1][i].size(0)
                num_nbr = batch.nbr_nids[j - 1].shape[-1]
                out = self.attn[j - 1](
                    node_feat=z[j - 1][i],
                    time_feat=self.time_encoder(torch.zeros(num_nodes, device=device)),
                    nbr_node_feat=z[j - 1][i + 1].reshape(num_nodes, num_nbr, -1),
                    edge_feat=batch.nbr_feats[i],
                    valid_nbr_mask=batch.nbr_nids[i] != PADDED_NODE_ID,
                    nbr_time_feat=self.time_encoder(
                        batch.times[i][:, None] - batch.nbr_times[i]
                    ),
                )
                z[j][i] = self.merge_layers[j - 1](out, z[0][i])

        return z[self.num_layers][0]


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).sigmoid().view(-1)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0

    encoder.memory.reset()  # Re-init memory of the model at the start of each epoch

    for batch in tqdm(loader):
        opt.zero_grad()

        z = encoder(batch, static_node_feats)
        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)

        encoder.memory.detach_memory()  # Detach memory to avoid BPTT
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        z = encoder(batch, static_node_feats)
        id_map = {nid.item(): i for i, nid in enumerate(batch.nids[0])}
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            src_idx = torch.tensor([id_map[n.item()] for n in src_ids], device=z.device)
            dst_idx = torch.tensor([id_map[n.item()] for n in dst_ids], device=z.device)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            y_pred = decoder(z_src, z_dst)

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

    return float(np.mean(perf_list))


args = parser.parse_args()
seed_everything(args.seed)

from pathlib import Path

from experiments import save_experiment_results_and_exit, setup_experiment

results = setup_experiment(args, Path(__file__))

evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.zeros((test_dg.num_nodes, 1), device=args.device)

if args.sampling == 'uniform':
    nbr_hook = NeighborSamplerHook(num_nbrs=args.n_nbrs)
elif args.sampling == 'recency':
    nbr_hook = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=test_dg.num_nodes,  # Assuming node ids at test set > train/val set
        edge_feats_dim=test_dg.edge_feats_dim,
    )
else:
    raise ValueError(f'Unknown sampling type: {args.sampling}')


_, dst, _ = train_dg.edges

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
hm.register_shared(nbr_hook)
registered_keys = hm.keys
train_key, val_key, test_key = registered_keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

encoder = TGN(
    node_dim=static_node_feats.shape[1],
    edge_dim=train_dg.edge_feats_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
    num_nodes=test_dg.num_nodes,
).to(args.device)
encoder.memory.set_device(args.device)
decoder = LinkPredictor(dim=args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        start_time = time.perf_counter()
        loss = train(train_loader, static_node_feats, encoder, decoder, opt)
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate(val_key):
        start_time = time.perf_counter()
        val_mrr = eval(val_loader, static_node_feats, encoder, decoder, evaluator)
        end_time = time.perf_counter()
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} Validation {METRIC_TGB_LINKPROPPRED}={val_mrr:.4f}'
    )
    results[f'val_{METRIC_TGB_LINKPROPPRED}'] = val_mrr
    results['train_latency_s'] = latency
    results['val_latency_s'] = end_time - start_time
    save_experiment_results_and_exit(results)

    # Clear memory state between epochs, except last epoch
    if epoch < args.epochs:
        hm.reset_state()
        encoder.memory.clear_msgs(list(range(test_dg.num_nodes)))


with hm.activate(test_key):
    test_mrr = eval(test_loader, static_node_feats, encoder, decoder, evaluator)
    print(f'Test {METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')

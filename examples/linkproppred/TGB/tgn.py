import argparse
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import (
    DGHook,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader
from tgm.nn import TemporalAttention, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGN TGB Example',
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
    default=[20],
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


class TGN(torch.nn.Module):
    def __init__(
        self,
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
            edge_dim=edge_dim,
            time_dim=time_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            time_encoder=self.time_encoder,
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
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        time_encoder: Time2Vec,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.time_encoder = time_encoder
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        batch: DGBatch,
        static_node_feats: torch.Tensor,
        memory: Memory,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = batch.src.device
        z = torch.zeros(len(batch.unique_nids), self.embed_dim, device=device)

        for hop in reversed(range(self.num_layers)):
            seed_nodes = batch.nids[hop]
            nbrs = batch.nbr_nids[hop]
            nbr_mask = batch.nbr_mask[hop].bool()
            if seed_nodes.numel() == 0:
                continue

            # TODO: Check and read static node features
            node_feat = static_node_feats[seed_nodes]
            node_time_feat = self.time_encoder(torch.zeros_like(seed_nodes))

            # If next next hops embeddings exist, use them instead of raw features
            nbr_feat = static_node_feats[nbrs]
            if hop < self.num_layers - 1:
                valid_nbrs = nbrs[nbr_mask]
                nbr_feat[nbr_mask] = z[batch.global_to_local(valid_nbrs)]

            delta_time = batch.times[hop][:, None] - batch.nbr_times[hop]
            delta_time = delta_time.masked_fill(~nbr_mask, 0)
            nbr_time_feat = self.time_encoder(delta_time)

            out = self.attn[hop](
                node_feat=node_feat + memory[seed_nodes],
                time_feat=node_time_feat,
                edge_feat=batch.nbr_feats[hop],
                nbr_node_feat=nbr_feat,
                nbr_time_feat=nbr_time_feat,
                nbr_mask=nbr_mask,
            )
            z[batch.global_to_local(seed_nodes)] = out
        return z


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    # Reinitialize memory of the model at the start of each epoch
    encoder.memory.reset()
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        z = encoder(batch, static_node_feats)

        z_src = z[batch.global_to_local(batch.src)]
        z_dst = z[batch.global_to_local(batch.dst)]
        z_neg = z[batch.global_to_local(batch.neg)]

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
        # Detach memory so we don't backpropagate to the start of time
        encoder.memory.detach_memory()
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    eval_metric: str,
    evaluator: Evaluator,
) -> dict:
    encoder.eval()
    decoder.eval()
    perf_list = []
    for batch in tqdm(loader):
        z = encoder(batch, static_node_feats)

        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            z_src = z[batch.global_to_local(src_ids)]
            z_dst = z[batch.global_to_local(dst_ids)]
            y_pred = decoder(z_src, z_dst)

            input_dict = {
                'y_pred_pos': y_pred[0].detach().cpu().numpy(),
                'y_pred_neg': y_pred[1:].detach().cpu().numpy(),
                'eval_metric': [eval_metric],
            }

            perf_list.append(evaluator.eval(input_dict)[eval_metric])
    metric_dict = {}
    metric_dict[eval_metric] = float(np.mean(perf_list))
    return metric_dict


args = parser.parse_args()
seed_everything(args.seed)

# loading negative sample from TGB
dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.embed_dim), device=args.device
    )


def _init_hooks(
    dg: DGraph, sampling_type: str, neg_sampler: object, split_mode: str
) -> List[DGHook]:
    if sampling_type == 'uniform':
        nbr_hook = NeighborSamplerHook(num_nbrs=args.n_nbrs)
    elif sampling_type == 'recency':
        nbr_hook = RecencyNeighborHook(
            num_nbrs=args.n_nbrs,
            num_nodes=dg.num_nodes,
            edge_feats_dim=dg.edge_feats_dim,
        )
    else:
        raise ValueError(f'Unknown sampling type: {args.sampling}')

    # Always produce negative edge prior to neighbor sampling for link prediction
    if split_mode in ['val', 'test']:
        neg_hook = TGBNegativeEdgeSamplerHook(neg_sampler, split_mode=split_mode)
    else:
        _, dst, _ = dg.edges
        min_dst, max_dst = int(dst.min()), int(dst.max())
        neg_hook = NegativeEdgeSamplerHook(low=min_dst, high=max_dst)
    return [neg_hook, nbr_hook]


test_loader = DGDataLoader(
    test_dg,
    hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'test'),
    batch_size=args.bsize,
)


encoder = TGN(
    edge_dim=train_dg.edge_feats_dim or args.embed_dim,
    time_dim=args.time_dim,
    embed_dim=static_node_feats.shape[1],
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
    # TODO: Need a clean way to clear nbr state across epochs
    train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'train'),
        batch_size=args.bsize,
    )
    val_loader = DGDataLoader(
        val_dg,
        hook=_init_hooks(test_dg, args.sampling, neg_sampler, 'val'),
        batch_size=args.bsize,
    )
    start_time = time.perf_counter()
    loss = train(train_loader, static_node_feats, encoder, decoder, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(
        val_loader, static_node_feats, encoder, decoder, eval_metric, evaluator
    )
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

    # Clear memory state between epochs
    encoder.memory.clear_msgs(list(range(test_dg.num_nodes)))


test_results = eval(
    test_loader, static_node_feats, encoder, decoder, eval_metric, evaluator
)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

import argparse
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from torch import Tensor
from torch_geometric.nn import AntiSymmetricConv, TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_geometric.utils import scatter
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import (
    METRIC_TGB_LINKPROPPRED,
    PADDED_NODE_ID,
    RECIPE_TGB_LINK_PRED,
)
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry
from tgm.nn import LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='CTAN LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs')
parser.add_argument('--n-layers', type=int, default=3, help='number of GNN layers')
parser.add_argument(
    '--epsilon', type=float, default=0.5, help='discretization step size'
)
parser.add_argument('--gamma', type=float, default=0.1, help='diffusion strength')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--n-nbrs',
    type=int,
    nargs='+',
    default=[32],
    help='num sampled nbrs at each hop',
)
parser.add_argument('--time-dim', type=int, default=256, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=256, help='attention dimension')
parser.add_argument('--memory-dim', type=int, default=256, help='memory dimension')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


# class LastAggregator(torch.nn.Module):
#    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
#        out = msg.new_zeros((dim_size, msg.size(-1)))
#
#        if index.numel() > 0:
#            scores = torch.full((dim_size, t.size(0)), float('-inf'), device=t.device)
#            scores[index, torch.arange(t.size(0), device=t.device)] = t.float()
#            argmax = scores.argmax(dim=1)
#            valid = scores.max(dim=1).values > float('-inf')
#            out[valid] = msg[argmax[valid]]
#
#        return out
#
#
# class SimpleMemory(torch.nn.Module):
#    def __init__(
#        self, num_nodes: int, memory_dim: int, aggr_module: Callable, init_time: int = 0
#    ) -> None:
#        super().__init__()
#        self.aggr_module = aggr_module
#        self.init_time = init_time
#        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
#        self.register_buffer(
#            'last_update', torch.full((num_nodes,), init_time, dtype=torch.long)
#        )
#        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long))
#
#    def update_state(self, src, pos_dst, t, src_emb: Tensor, pos_dst_emb: Tensor):
#        idx = torch.cat([src, pos_dst], dim=0)
#        _idx = idx.unique()
#        self._assoc[_idx] = torch.arange(_idx.size(0), device=_idx.device)
#
#        t = torch.cat([t, t], dim=0)
#        last_update = scatter(t, self._assoc[idx], 0, _idx.size(0), reduce='max')
#
#        emb = torch.cat([src_emb, pos_dst_emb], dim=0)
#        aggr = self.aggr_module(emb, self._assoc[idx], t, _idx.size(0))
#
#        self.last_update[_idx] = last_update
#        self.memory[_idx] = aggr.detach()
#
#    def reset_state(self):
#        self.memory.zero_()
#        self.last_update.fill_(self.init_time)
#
#    def detach(self):
#        self.memory.detach_()
#
#    def forward(self, n_id):
#        return self.memory[n_id], self.last_update[n_id]

from torch_geometric.nn.inits import zeros, ones
from torch_geometric.utils import scatter
from torch_scatter import scatter_max


class SimpleMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        aggr_module: Callable,
        init_time: int = 0,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.init_time = init_time
        self.aggr_module = aggr_module

        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        last_update = torch.ones(self.num_nodes, dtype=torch.long) * init_time
        self.register_buffer('last_update', last_update)
        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long))

    def update_state(self, src, pos_dst, t, src_emb, pos_dst_emb):
        idx = torch.cat([src, pos_dst], dim=0)
        _idx = idx.unique()
        self._assoc[_idx] = torch.arange(_idx.size(0), device=_idx.device)

        t = torch.cat([t, t], dim=0)
        last_update = scatter(t, self._assoc[idx], 0, _idx.size(0), reduce='max')

        emb = torch.cat([src_emb, pos_dst_emb], dim=0)
        aggr = self.aggr_module(emb, self._assoc[idx], t, _idx.size(0))

        self.last_update[_idx] = last_update
        self.memory[_idx] = aggr.detach()

    def reset_state(self):
        zeros(self.memory)
        ones(self.last_update)
        self.last_update *= self.init_time

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id):
        return self.memory[n_id], self.last_update[n_id]


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class CTAN(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        memory_dim: int,
        time_dim: int,
        node_dim: int = 0,
        num_iters: int = 1,
        epsilon: float = 1.0,
        gamma: float = 0.1,
        mean_delta_t: float = 0.0,
        std_delta_t: float = 1.0,
    ):
        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.time_enc = TimeEncoder(time_dim)
        self.enc_x = nn.Linear(memory_dim + node_dim, memory_dim)

        phi = TransformerConv(
            memory_dim, memory_dim, edge_dim=edge_dim + time_dim, root_weight=False
        )
        self.aconv = AntiSymmetricConv(
            memory_dim, phi, num_iters=num_iters, epsilon=epsilon, gamma=gamma
        )

    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.aconv.reset_parameters()
        self.enc_x.reset_parameters()

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = (last_update[edge_index[0]] - t).abs()
        rel_t = ((rel_t - self.mean_delta_t) / self.std_delta_t).to(x.dtype)
        enc_x = self.enc_x(x)
        edge_attr = torch.cat([msg, self.time_enc(rel_t)], dim=-1)
        z = self.aconv(enc_x, edge_index, edge_attr)
        z = torch.tanh(z)
        return z


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    memory: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    memory.train()
    encoder.train()
    decoder.train()
    total_loss = 0

    memory.reset_state()

    for batch in tqdm(loader):
        opt.zero_grad()

        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID

        #! run my own deduplication
        all_nids = torch.cat([batch.src, batch.dst, batch.neg, nbr_nodes[nbr_mask]])
        batch.unique_nids = torch.unique(all_nids, sorted=True)  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(batch.unique_nids, x)  # type: ignore

        num_nbrs = len(nbr_nodes) // (len(batch.src) + len(batch.dst) + len(batch.neg))
        src_nodes = torch.cat(
            [
                batch.src.repeat_interleave(num_nbrs),
                batch.dst.repeat_interleave(num_nbrs),
                batch.neg.repeat_interleave(num_nbrs),
            ]
        )
        nbr_edge_index = torch.stack(
            [
                batch.global_to_local(src_nodes[nbr_mask]),
                batch.global_to_local(nbr_nodes[nbr_mask]),
            ]
        )

        nbr_times = batch.nbr_times[0].flatten()[nbr_mask]
        nbr_feats = batch.nbr_feats[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        z = torch.cat([z, static_node_feats[batch.unique_nids]], dim=-1)
        z = encoder(z, last_update, nbr_edge_index, nbr_times, nbr_feats)

        inv_src = batch.global_to_local(batch.src)
        inv_dst = batch.global_to_local(batch.dst)
        inv_neg = batch.global_to_local(batch.neg)
        pos_out = decoder(z[inv_src], z[inv_dst])
        neg_out = decoder(z[inv_src], z[inv_neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        # Update memory with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.time, z[inv_src], z[inv_dst])

        loss.backward()
        opt.step()
        total_loss += float(loss)

        memory.detach()

    return total_loss


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    memory: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    memory.eval()
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID

        #! run my own deduplication
        all_nids = torch.cat([batch.src, batch.dst, batch.neg, nbr_nodes[nbr_mask]])
        batch.unique_nids = torch.unique(all_nids, sorted=True)  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(batch.unique_nids, x)  # type: ignore

        num_nbrs = len(nbr_nodes) // (len(batch.src) + len(batch.dst) + len(batch.neg))
        src_nodes = torch.cat(
            [
                batch.src.repeat_interleave(num_nbrs),
                batch.dst.repeat_interleave(num_nbrs),
                batch.neg.repeat_interleave(num_nbrs),
            ]
        )
        nbr_edge_index = torch.stack(
            [
                batch.global_to_local(src_nodes[nbr_mask]),
                batch.global_to_local(nbr_nodes[nbr_mask]),
            ]
        )
        nbr_times = batch.nbr_times[0].flatten()[nbr_mask]
        nbr_feats = batch.nbr_feats[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        z = torch.cat([z, static_node_feats[batch.unique_nids]], dim=-1)
        z = encoder(z, last_update, nbr_edge_index, nbr_times, nbr_feats)

        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            inv_src = batch.global_to_local(src_ids)
            inv_dst = batch.global_to_local(dst_ids)
            y_pred = decoder(z[inv_src], z[inv_dst]).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        # Update memory with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.time, z[inv_src], z[inv_dst])

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


def compute_delta_t_stats(train_dg: DGraph) -> Tuple[float, float]:
    last_timestamp = {}
    delta_times = []

    for src, dst, t in zip(*train_dg.edges):
        src, dst, t = src.item(), dst.item(), t.item()

        dt_src = t - last_timestamp.get(src, train_dg.start_time)
        dt_dst = t - last_timestamp.get(dst, train_dg.start_time)
        delta_times.extend([dt_src, dt_dst])

        last_timestamp[src] = t
        last_timestamp[dst] = t

    return np.mean(delta_times), np.std(delta_times)


mean_delta_t, std_delta_t = compute_delta_t_stats(train_dg)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.zeros((test_dg.num_nodes, 1), device=args.device)

nbr_hook = RecencyNeighborHook(
    num_nbrs=args.n_nbrs,
    num_nodes=test_dg.num_nodes,  # Assuming node ids at test set > train/val set
    seed_nodes_keys=['src', 'dst', 'neg'],
    seed_times_keys=['time', 'time', 'neg_time'],
)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

memory = SimpleMemory(
    num_nodes=test_dg.num_nodes,
    memory_dim=args.memory_dim,
    aggr_module=LastAggregator(),
    init_time=train_dg.start_time,
).to(args.device)
encoder = CTAN(
    edge_dim=train_dg.edge_feats_dim,
    memory_dim=args.memory_dim,
    time_dim=args.time_dim,
    node_dim=static_node_feats.shape[1],
    num_iters=args.n_layers,
    mean_delta_t=mean_delta_t,
    std_delta_t=std_delta_t,
    epsilon=args.epsilon,
    gamma=args.gamma,
).to(args.device)
decoder = LinkPredictor(
    node_dim=args.memory_dim, hidden_dim=args.memory_dim, merge_op='sum'
).to(args.device)
opt = torch.optim.Adam(
    set(memory.parameters()) | set(encoder.parameters()) | set(decoder.parameters()),
    lr=args.lr,
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, static_node_feats, memory, encoder, decoder, opt)

    log_metric('Loss', loss, epoch=epoch)
    if epoch % 10 == 1:
        with hm.activate(val_key):
            val_mrr = eval(
                val_loader, static_node_feats, memory, encoder, decoder, evaluator
            )
        log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate(test_key):
    test_mrr = eval(test_loader, static_node_feats, memory, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

import argparse
import copy
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from torch import Tensor
from torch.nn import GRUCell
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import (
    HookManager,
    NodeEventTemporalSubgraphHook,
    RecencyNeighborHook,
)
from tgm.nn import NodePredictor, Time2Vec
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGN NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=100, help='attention dimension')
parser.add_argument('--memory-dim', type=int, default=100, help='memory dimension')
parser.add_argument(
    '--n-nbrs',
    type=int,
    nargs='+',
    default=[10],
    help='num sampled nbrs at each hop',
)
parser.add_argument(
    '--time-gran',
    type=str,
    default=None,
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.time_dim
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        out = msg.new_zeros((dim_size, msg.size(-1)))

        if index.numel() > 0:
            scores = torch.full((dim_size, t.size(0)), float('-inf'), device=t.device)
            scores[index, torch.arange(t.size(0), device=t.device)] = t.float()
            argmax = scores.argmax(dim=1)
            valid = scores.max(dim=1).values > float('-inf')
            out[valid] = msg[argmax[valid]]

        return out


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]


class TGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = Time2Vec(time_dim=time_dim)
        self.memory_updater = GRUCell(message_module.out_channels, memory_dim)

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.empty(num_nodes, dtype=torch.long))
        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.w.weight.device

    def reset_parameters(self):
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, _ = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, _ = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module
        )

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.memory_updater(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx.long(), 0, dim_size, reduce='max')[n_id]
        return memory, last_update

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
    ):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0).to(self.device)
        dst = torch.cat(dst, dim=0).to(self.device)
        t = torch.cat(t, dim=0).to(self.device)
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, src, dst

    def train(self, mode: bool = True):
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    memory: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    memory.train()
    encoder.train()
    decoder.train()
    total_loss = 0

    perf_list = []
    memory.reset_state()
    for batch in tqdm(loader):
        opt.zero_grad()
        y_labels = batch.dynamic_node_feats
        if y_labels is not None:
            sg_edge_index = torch.stack(
                [
                    batch.sg_global_to_local(batch.sg_src),
                    batch.sg_global_to_local(batch.sg_dst),
                ]
            )
            z, last_update = memory(batch.sg_unique_nids)
            z = encoder(
                z, last_update, sg_edge_index, batch.sg_time, batch.sg_edge_feats
            )

            inv_src = batch.sg_global_to_local(batch.node_ids)
            y_pred = decoder(z[inv_src])
            loss = F.cross_entropy(y_pred, y_labels)
            loss.backward()
            opt.step()
            total_loss += float(loss)

            input_dict = {
                'y_true': y_labels,
                'y_pred': y_pred,
                'eval_metric': [METRIC_TGB_NODEPROPPRED],
            }
            perf = evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED]
            perf_list.append(perf)

        # Update memory with ground-truth state.
        if len(batch.src) > 0:
            memory.update_state(
                batch.src, batch.dst, batch.time, batch.edge_feats.float()
            )
        memory.detach()

    return total_loss, float(np.mean(perf_list))


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
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
        y_labels = batch.dynamic_node_feats
        if y_labels is not None:
            sg_edge_index = torch.stack(
                [
                    batch.sg_global_to_local(batch.sg_src),
                    batch.sg_global_to_local(batch.sg_dst),
                ]
            )

            z, last_update = memory(batch.sg_unique_nids)
            z = encoder(
                z, last_update, sg_edge_index, batch.sg_time, batch.sg_edge_feats
            )

            inv_src = batch.sg_global_to_local(batch.node_ids)
            y_pred = decoder(z[inv_src])

            input_dict = {
                'y_true': y_labels,
                'y_pred': y_pred,
                'eval_metric': [METRIC_TGB_NODEPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

        # Update memory with ground-truth state.
        if len(batch.src) > 0:
            memory.update_state(
                batch.src, batch.dst, batch.time, batch.edge_feats.float()
            )

    return float(np.mean(perf_list))


seed_everything(args.seed)

if args.time_gran is not None:
    full_data = DGData.from_tgb(args.dataset)
    full_graph = DGraph(full_data)
    train_data, val_data, test_data = full_data.split()
    train_data = train_data.discretize(args.time_gran)
    val_data = val_data.discretize(args.time_gran)
    test_data = test_data.discretize(args.time_gran)
    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)
else:
    train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

evaluator = Evaluator(name=args.dataset)
num_classes = train_dg.dynamic_node_feats_dim

nbr_hook = RecencyNeighborHook(
    num_nbrs=args.n_nbrs,
    num_nodes=test_dg.num_nodes,  # Assuming node ids at test set > train/val set
    seed_nodes_keys=['node_ids'],
    seed_times_keys=['node_times'],
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register_shared(nbr_hook)
hm.register_shared(NodeEventTemporalSubgraphHook())

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

memory = TGNMemory(
    test_dg.num_nodes,
    test_dg.edge_feats_dim,
    args.memory_dim,
    args.time_dim,
    message_module=IdentityMessage(
        test_dg.edge_feats_dim, args.memory_dim, args.time_dim
    ),
    aggregator_module=LastAggregator(),
).to(args.device)
encoder = GraphAttentionEmbedding(
    in_channels=args.memory_dim,
    out_channels=args.embed_dim,
    msg_dim=test_dg.edge_feats_dim,
    time_enc=memory.time_enc,
).to(args.device)
decoder = NodePredictor(
    in_dim=args.embed_dim, out_dim=num_classes, hidden_dim=args.embed_dim
).to(args.device)
opt = torch.optim.Adam(
    set(memory.parameters()) | set(encoder.parameters()) | set(decoder.parameters()),
    lr=args.lr,
)

for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        loss, train_metric = train(train_loader, memory, encoder, decoder, opt)

    with hm.activate('val'):
        val_metric = eval(val_loader, memory, encoder, decoder, evaluator)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Train {METRIC_TGB_NODEPROPPRED}', train_metric, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_metric, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_metric = eval(test_loader, memory, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_metric, epoch=args.epochs)

import argparse
import copy
import time
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch import Tensor
from torch.nn import GRUCell, Linear, RNNCell
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import scatter
from torch_scatter import scatter_max
from tqdm import tqdm

from tgm import DGData, DGraph
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader
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
    default=[10],
    help='num sampled nbrs at each hop',
)
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=100, help='attention dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['uniform', 'recency'],
    help='sampling strategy',
)


# -------------------------------------------------------
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
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
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
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
        memory_updater_cell: str = 'gru',
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        if memory_updater_cell == 'gru':  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == 'rnn':  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('_assoc', torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
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
        msg_s, t_s, src_s, dst_s = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(
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
        last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id]
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
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
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


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


# -------------------------------------------------------


def train(loader: DGDataLoader, opt: torch.optim.Optimizer):
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.

    total_loss = 0
    num_edges = 0
    for batch in tqdm(loader):
        opt.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.time, batch.edge_feats
        neg_dst = batch.neg

        # n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # n_id, edge_index, e_id = neighbor_loader(n_id)

        seed_nodes = batch.nids[0].repeat_interleave(10)
        nbr_nodes = batch.nbr_nids[0].flatten()

        seed_nodes = batch.global_to_local(seed_nodes)
        nbr_nodes = batch.global_to_local(nbr_nodes)
        nbr_edge_index = torch.stack([seed_nodes, nbr_nodes])

        nbr_times = batch.nbr_times[0].flatten()
        nbr_feats = batch.nbr_feats[0].flatten(0, -2).float()

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](batch.unique_nids)
        z = model['gnn'](
            z,
            last_update,
            nbr_edge_index,
            nbr_times,
            nbr_feats,
        )

        pos_out = model['link_pred'](
            z[batch.global_to_local(src)], z[batch.global_to_local(pos_dst)]
        )
        neg_out = model['link_pred'](
            z[batch.global_to_local(src)], z[batch.global_to_local(neg_dst)]
        )

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        msg = msg.float()
        model['memory'].update_state(src, pos_dst, t, msg)

        loss.backward()
        opt.step()

        model['memory'].detach()

        total_loss += float(loss) * batch.src.size(0)
        num_edges += batch.src.size(0)

    return total_loss / num_edges


@torch.no_grad()
def eval(loader, eval_metric: str, evaluator: Evaluator) -> dict:
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []
    for batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            batch.src,
            batch.dst,
            batch.time,
            batch.edge_feats,
        )

        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src = batch.src[idx].repeat(len(dst))

            seed_nodes = batch.nids[0].repeat_interleave(10)
            nbr_nodes = batch.nbr_nids[0].flatten()

            seed_nodes = batch.global_to_local(seed_nodes)
            nbr_nodes = batch.global_to_local(nbr_nodes)
            nbr_edge_index = torch.stack([seed_nodes, nbr_nodes])

            nbr_times = batch.nbr_times[0].flatten()
            nbr_feats = batch.nbr_feats[0].flatten(0, -2)

            z, last_update = model['memory'](batch.unique_nids)
            z = model['gnn'](z, last_update, nbr_edge_index, nbr_times, nbr_feats)
            y_pred = model['link_pred'](
                z[batch.global_to_local(src)], z[batch.global_to_local(dst)]
            )

            input_dict = {
                'y_pred_pos': np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                'y_pred_neg': np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])

        # Update memory and neighbor loader with ground-truth state.
        pos_msg = pos_msg.float()
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)

    metric_dict = {}
    metric_dict[eval_metric] = float(torch.tensor(perf_list).mean())
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
    static_node_feats = torch.zeros((test_dg.num_nodes, 1), device=args.device)

MEM_DIM = 100
memory = TGNMemory(
    test_dg.num_nodes,
    test_dg.edge_feats_dim,
    MEM_DIM,
    args.time_dim,
    message_module=IdentityMessage(test_dg.edge_feats_dim, MEM_DIM, args.time_dim),
    aggregator_module=LastAggregator(),
).to(args.device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=args.embed_dim,
    msg_dim=test_dg.edge_feats_dim,
    time_enc=memory.time_enc,
).to(args.device)

link_pred = LinkPredictor(in_channels=args.embed_dim).to(args.device)

model = {'memory': memory, 'gnn': gnn, 'link_pred': link_pred}
opt = torch.optim.Adam(
    set(model['memory'].parameters())
    | set(model['gnn'].parameters())
    | set(model['link_pred'].parameters()),
    lr=args.lr,
)

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
hm = HookManager(keys=['train', 'val', 'test'])
hm.register('train', NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max())))
hm.register('val', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val'))
hm.register('test', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='test'))
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)


for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        start_time = time.perf_counter()
        loss = train(train_loader, opt)
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate('val'):
        val_results = eval(val_loader, eval_metric, evaluator)
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
            + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
        )
    if epoch < args.epochs:
        hm.reset_state()
        model['memory'].clear_msgs(list(range(test_dg.num_nodes)))

with hm.activate('test'):
    test_results = eval(test_loader, eval_metric, evaluator)
    print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

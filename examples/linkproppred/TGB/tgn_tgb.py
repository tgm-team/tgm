import copy
from typing import Callable, Dict, List, Tuple

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

from tgm import DGraph
from tgm.hooks import (
    DGHook,
    NegativeEdgeSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader
from tgm.util.seed import seed_everything


# -------------------------------------------------------
class LinkPredictor(torch.nn.Module):
    """Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        # h = F.log_softmax(h, dim=-1)
        return h


class GraphAttentionEmbedding(torch.nn.Module):
    """Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        print(edge_index[0].shape, t.shape)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        print(rel_t_enc.shape, msg.shape)
        input()
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
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

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
        # self.gru = GRUCell(message_module.out_channels, memory_dim)
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
        r"""Resets all learnable parameters of the module."""
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
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp.
        """
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`.
        """
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
        # TODO: THIS IS DIFFERENT
        aggr = aggr.float()
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
        """Sets the module in training mode."""
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
        # TODO: THIS IS DIFFERENT
        t = t.float()
        return self.lin(t.view(-1, 1)).cos()


# -------------------------------------------------------


def train(loader: DGDataLoader, opt: torch.optim.Optimizer):
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.

    total_loss = 0
    num_edges = 0
    i = 0
    for batch in loader:
        opt.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.time, batch.edge_feats
        neg_dst = batch.neg

        if i > 0:
            hop = 0
            seed_nodes = batch.nids[hop]
            nbr_nodes = batch.nbr_nids[hop]

            seed_nodes_flat = seed_nodes.repeat_interleave(10)
            nbr_nodes_flat = nbr_nodes.flatten()

            nbr_edge_index = torch.stack([seed_nodes_flat, nbr_nodes_flat])
            nbr_times = batch.nbr_times[hop].flatten()
            nbr_feats = batch.nbr_feats[hop].flatten(0, -2)

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
            loss += F.binary_cross_entropy_with_logits(
                neg_out, torch.zeros_like(neg_out)
            )

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)

        if i > 0:
            loss.backward()
            opt.step()

        model['memory'].detach()

        if i > 0:
            total_loss += float(loss) * batch.src.size(0)
        num_edges += batch.src.size(0)

        i += 1

    return total_loss / num_edges


@torch.no_grad()
def test(loader, neg_sampler, split_mode):
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []
    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(
            pos_src, pos_dst, pos_t, split_mode=split_mode
        )

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                'y_pred_pos': np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                'y_pred_neg': np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                'eval_metric': [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())
    return perf_metrics


DATA = 'tgbl-wiki'
LR = 1e-4
BATCH_SIZE = 200
K_VALUE = 10
NUM_EPOCH = 30
SEED = 1
MEM_DIM = 100
TIME_DIM = 100
EMB_DIM = 100
NUM_NEIGHBORS = 10
MODEL_NAME = 'TGN'

seed_everything(SEED)

device = torch.device('cpu')

dataset = PyGLinkPropPredDataset(name=DATA, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=DATA)
dataset.load_val_ns()
dataset.load_test_ns()

train_dg = DGraph(DATA, split='train', device=device)
val_dg = DGraph(DATA, split='val', device=device)
test_dg = DGraph(DATA, split='test', device=device)

# TODO: Read from graph
NUM_NODES, NODE_FEAT_DIM = test_dg.num_nodes, EMB_DIM
STATIC_NODE_FEAT = torch.zeros((NUM_NODES, NODE_FEAT_DIM), device=device)
EDGE_FEAT_DIM = test_dg.edge_feats_dim

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler


memory = TGNMemory(
    NUM_NODES,
    EDGE_FEAT_DIM,
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(EDGE_FEAT_DIM, MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=EDGE_FEAT_DIM,
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {'memory': memory, 'gnn': gnn, 'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters())
    | set(model['gnn'].parameters())
    | set(model['link_pred'].parameters()),
    lr=LR,
)

# Helper vector to map global node indices to local ones.
assoc = torch.empty(NUM_NODES, dtype=torch.long, device=device)


def _init_hooks(dg: DGraph, neg_sampler: object, split_mode: str) -> List[DGHook]:
    nbr_hook = RecencyNeighborHook(
        num_nbrs=[NUM_NEIGHBORS],
        num_nodes=dg.num_nodes,
        edge_feats_dim=EDGE_FEAT_DIM,
    )

    # Always produce negative edge prior to neighbor sampling for link prediction
    if split_mode in ['val', 'test']:
        neg_hook = TGBNegativeEdgeSamplerHook(neg_sampler, split_mode=split_mode)
    else:
        _, dst, _ = dg.edges
        min_dst, max_dst = int(dst.min()), int(dst.max())
        neg_hook = NegativeEdgeSamplerHook(low=min_dst, high=max_dst)
    return [neg_hook, nbr_hook]


val_perf_list = []
for epoch in range(1, NUM_EPOCH + 1):
    train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(test_dg, neg_sampler, 'train'),
        batch_size=BATCH_SIZE,
    )
    val_loader = DGDataLoader(
        val_dg,
        hook=_init_hooks(test_dg, neg_sampler, 'val'),
        batch_size=BATCH_SIZE,
    )

    loss = train(train_loader, optimizer)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    perf_metric_val = test(val_loader, neg_sampler, split_mode='val')
    print(f'\tValidation {metric}: {perf_metric_val: .4f}')
    val_perf_list.append(perf_metric_val)

perf_metric_test = test(test_loader, neg_sampler, split_mode='test')
print(f'\tTest: {metric}: {perf_metric_test: .4f}')

import argparse
from typing import Callable

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

from tgm import DGBatch, DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import NeighborSamplerHook, RecencyNeighborHook, RecipeRegistry
from tgm.nn import LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='CTAN LinkPropPred Example',
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
    default=[20, 20],
    help='num sampled nbrs at each hop',
)
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=172, help='attention dimension')
parser.add_argument('--memory-dim', type=int, default=100, help='memory dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['uniform', 'recency'],
    help='sampling strategy',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


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


class SimpleMemory(torch.nn.Module):
    def __init__(
        self, num_nodes: int, memory_dim: int, aggregator_module: Callable
    ) -> None:
        super().__init__()
        self.aggr_module = aggregator_module
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(num_nodes, dtype=torch.long))
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
        self.memory.zero_()
        self.last_update.zero_()

    def detach(self):
        self.memory.detach_()

    def forward(self, n_id):
        return self.memory[n_id], self.last_update[n_id]


class CTAN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        edge_dim: int,
        memory_dim: int,
        time_dim: int,
        node_dim: int = 0,
        num_iters: int = 1,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.memory = SimpleMemory(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            aggregator_module=LastAggregator(),
        )
        self.time_enc = TimeEncoder(time_dim)
        self.enc_x = nn.Linear(memory_dim + node_dim, memory_dim)

        phi = TransformerConv(
            memory_dim, memory_dim, edge_dim=edge_dim + time_dim, root_weight=False
        )
        self.aconv = AntiSymmetricConv(memory_dim, phi, num_iters=num_iters)

    def reset_parameters(self):
        self.memory.reset_state()
        self.time_enc.reset_parameters()
        self.aconv.reset_parameters()
        self.enc_x.reset_parameters()

    def detach_memory(self):
        self.memory.detach_state()

    def zero_grad_memory(self):
        self.memory.zero_grad_state()

    def backup_memory(self):
        return self.memory.backup()

    def restore_memory(self, backup):
        self.memory.restore(backup)

    def update(self, src, pos_dst, t, msg, src_emb, pos_dst_emb):
        self.memory.update_state(src, pos_dst, t, src_emb, pos_dst_emb)

    def forward(
        self, batch: DGBatch, static_node_features: torch.Tensor
    ) -> torch.Tensor:
        batch, n_id, msg, t, edge_index = None, None, None, None, None
        z, last_update = self.memory(n_id)
        z = torch.cat([z, static_node_features[n_id]], dim=-1)
        enc_z = self.enc_x(z)
        rel_t = (last_update[edge_index[0]] - t).abs().to(z.dtype)
        edge_attr = torch.cat([msg, self.time_enc(rel_t)], dim=-1)
        z_out = self.aconv(enc_z, edge_index, edge_attr=edge_attr)
        return z_out


@log_gpu
@log_latency
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

    for batch in tqdm(loader):
        opt.zero_grad()

        z = encoder(batch, static_node_feats)

        # if self.predict_dst:
        #    assert n_neg is None
        #    pos_out = self.readout(z[id_mapper[dst]])
        #    neg_out = None
        # else:
        #    out = self.readout(z[id_mapper[src]], z[id_mapper[dst]])
        #    pos_out = out[:-n_neg]
        #    neg_out = out[-n_neg:]

        # emb_src = z[id_mapper[src[:-n_neg]]]
        # emb_pos_dst = z[id_mapper[dst[:-n_neg]]]
        # return pos_out, neg_out, emb_src, emb_pos_dst

        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@log_gpu
@log_latency
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
            y_pred = decoder(z_src, z_dst).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

    return float(np.mean(perf_list))


seed_everything(args.seed)
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
    nbr_hook = NeighborSamplerHook(
        num_nbrs=args.n_nbrs,
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['time', 'time', 'neg_time'],
    )
elif args.sampling == 'recency':
    nbr_hook = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=test_dg.num_nodes,  # Assuming node ids at test set > train/val set
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['time', 'time', 'neg_time'],
    )
else:
    raise ValueError(f'Unknown sampling type: {args.sampling}')


hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

encoder = CTAN(
    num_nodes=test_dg.num_nodes,
    edge_dim=train_dg.edge_feats_dim,
    memory_dim=args.memory_dim,
    time_dim=args.time_dim,
    node_dim=static_node_feats.shape[1],
).to(args.device)
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, static_node_feats, encoder, decoder, opt)

    with hm.activate(val_key):
        val_mrr = eval(val_loader, static_node_feats, encoder, decoder, evaluator)
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate(test_key):
    test_mrr = eval(test_loader, static_node_feats, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

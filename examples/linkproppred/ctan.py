import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
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
from tgm.nn.encoder import CTAN, CTANMemory, LastAggregator
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
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
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
    static_node_x = loader.dgraph.static_node_x

    memory.reset_state()

    for batch in tqdm(loader):
        opt.zero_grad()

        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID

        all_nids = torch.cat(
            [batch.edge_src, batch.edge_dst, batch.neg, nbr_nodes[nbr_mask]]
        )
        batch.unique_nids = torch.unique(all_nids, sorted=True)  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(batch.unique_nids, x)  # type: ignore

        num_nbrs = len(nbr_nodes) // (
            len(batch.edge_src) + len(batch.edge_dst) + len(batch.neg)
        )
        src_nodes = torch.cat(
            [
                batch.edge_src.repeat_interleave(num_nbrs),
                batch.edge_dst.repeat_interleave(num_nbrs),
                batch.neg.repeat_interleave(num_nbrs),
            ]
        )
        nbr_edge_index = torch.stack(
            [
                batch.global_to_local(src_nodes[nbr_mask]),
                batch.global_to_local(nbr_nodes[nbr_mask]),
            ]
        )

        nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
        nbr_edge_x = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        z = torch.cat([z, static_node_x[batch.unique_nids]], dim=-1)
        z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

        inv_src = batch.global_to_local(batch.edge_src)
        inv_dst = batch.global_to_local(batch.edge_dst)
        inv_neg = batch.global_to_local(batch.neg)
        pos_out = decoder(z[inv_src], z[inv_dst])
        neg_out = decoder(z[inv_src], z[inv_neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        # Update memory with ground-truth state.
        memory.update_state(
            batch.edge_src, batch.edge_dst, batch.edge_time, z[inv_src], z[inv_dst]
        )

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
    memory: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    memory.eval()
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID

        all_nids = torch.cat(
            [batch.edge_src, batch.edge_dst, batch.neg, nbr_nodes[nbr_mask]]
        )
        batch.unique_nids = torch.unique(all_nids, sorted=True)  # type: ignore
        batch.global_to_local = lambda x: torch.searchsorted(batch.unique_nids, x)  # type: ignore

        num_nbrs = len(nbr_nodes) // (
            len(batch.edge_src) + len(batch.edge_dst) + len(batch.neg)
        )
        src_nodes = torch.cat(
            [
                batch.edge_src.repeat_interleave(num_nbrs),
                batch.edge_dst.repeat_interleave(num_nbrs),
                batch.neg.repeat_interleave(num_nbrs),
            ]
        )
        nbr_edge_index = torch.stack(
            [
                batch.global_to_local(src_nodes[nbr_mask]),
                batch.global_to_local(nbr_nodes[nbr_mask]),
            ]
        )
        nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
        nbr_edge_x = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        z = torch.cat([z, static_node_x[batch.unique_nids]], dim=-1)
        z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.edge_dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.edge_src[idx].repeat(len(dst_ids))

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
        memory.update_state(
            batch.edge_src, batch.edge_dst, batch.edge_time, z[inv_src], z[inv_dst]
        )

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
if full_data.static_node_x is None:
    full_data.static_node_x = torch.randn((full_data.num_nodes, 1), device=args.device)

train_data, val_data, test_data = full_data.split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


def compute_delta_t_stats(train_dg: DGraph) -> Tuple[float, float]:
    last_timestamp = {}
    delta_times = []

    for src, dst, t in zip(train_dg.edge_src, train_dg.edge_dst, train_dg.edge_time):
        src, dst, t = src.item(), dst.item(), t.item()

        dt_src = t - last_timestamp.get(src, train_dg.start_time)
        dt_dst = t - last_timestamp.get(dst, train_dg.start_time)
        delta_times.extend([dt_src, dt_dst])

        last_timestamp[src] = t
        last_timestamp[dst] = t

    return np.mean(delta_times), np.std(delta_times)


mean_delta_t, std_delta_t = compute_delta_t_stats(train_dg)

nbr_hook = RecencyNeighborHook(
    num_nbrs=args.n_nbrs,
    num_nodes=full_data.num_nodes,
    seed_nodes_keys=['edge_src', 'edge_dst', 'neg'],
    seed_times_keys=['edge_time', 'edge_time', 'neg_time'],
)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

memory = CTANMemory(
    num_nodes=test_dg.num_nodes,
    memory_dim=args.memory_dim,
    aggr_module=LastAggregator(),
    init_time=train_dg.start_time,
).to(args.device)
encoder = CTAN(
    node_dim=train_dg.static_node_x_dim,
    edge_dim=train_dg.edge_x_dim,
    time_dim=args.time_dim,
    memory_dim=args.memory_dim,
    num_iters=args.n_layers,
    mean_delta_t=mean_delta_t,
    std_delta_t=std_delta_t,
    epsilon=args.epsilon,
    gamma=args.gamma,
).to(args.device)
decoder = LinkPredictor(node_dim=args.memory_dim, merge_op='sum').to(args.device)
opt = torch.optim.Adam(
    set(memory.parameters()) | set(encoder.parameters()) | set(decoder.parameters()),
    lr=args.lr,
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, memory, encoder, decoder, opt)

    with hm.activate(val_key):
        val_mrr = eval(val_loader, memory, encoder, decoder, evaluator)
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate(test_key):
    test_mrr = eval(test_loader, memory, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

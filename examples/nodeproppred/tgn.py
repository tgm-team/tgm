import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED, PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import DeduplicationHook, HookManager, RecencyNeighborHook
from tgm.nn import NodePredictor, TGNMemory
from tgm.nn.encoder.tgn import (
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
)
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
        y_labels = batch.node_y
        if y_labels is not None:
            nbr_nodes = batch.nbr_nids[0].flatten()
            nbr_mask = nbr_nodes != PADDED_NODE_ID

            num_nbrs = len(nbr_nodes) // (len(batch.node_y_nids))
            src_nodes = torch.cat(
                [
                    batch.node_y_nids.repeat_interleave(num_nbrs),
                ]
            )
            nbr_edge_index = torch.stack(
                [
                    batch.global_to_local(src_nodes[nbr_mask]),
                    batch.global_to_local(nbr_nodes[nbr_mask]),
                ]
            ).to(dtype=torch.int64)

            nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
            nbr_edge_x = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

            z, last_update = memory(batch.unique_nids)
            z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

            inv_src = batch.global_to_local(batch.node_y_nids)
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
        if len(batch.edge_src) > 0:
            memory.update_state(
                batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float()
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
        y_labels = batch.node_y
        if y_labels is not None:
            nbr_nodes = batch.nbr_nids[0].flatten()
            nbr_mask = nbr_nodes != PADDED_NODE_ID

            num_nbrs = len(nbr_nodes) // (len(batch.node_y_nids))
            src_nodes = torch.cat(
                [
                    batch.node_y_nids.repeat_interleave(num_nbrs),
                ]
            )
            nbr_edge_index = torch.stack(
                [
                    batch.global_to_local(src_nodes[nbr_mask]),
                    batch.global_to_local(nbr_nodes[nbr_mask]),
                ]
            ).to(dtype=torch.int64)

            nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
            nbr_edge_x = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

            z, last_update = memory(batch.unique_nids)
            z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

            inv_src = batch.global_to_local(batch.node_y_nids)
            y_pred = decoder(z[inv_src])

            input_dict = {
                'y_true': y_labels,
                'y_pred': y_pred,
                'eval_metric': [METRIC_TGB_NODEPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

        # Update memory with ground-truth state.
        if len(batch.edge_src) > 0:
            memory.update_state(
                batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float()
            )

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
train_data, val_data, test_data = full_data.split()

if args.time_gran is not None:
    train_data = train_data.discretize(args.time_gran)
    val_data = val_data.discretize(args.time_gran)
    test_data = test_data.discretize(args.time_gran)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

num_classes = train_dg.node_y_dim

nbr_hook = RecencyNeighborHook(
    num_nbrs=args.n_nbrs,
    num_nodes=full_data.num_nodes,
    seed_nodes_keys=['node_y_nids'],
    seed_times_keys=['node_y_time'],
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register_shared(nbr_hook)
hm.register_shared(DeduplicationHook())

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

memory = TGNMemory(
    full_data.num_nodes,
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

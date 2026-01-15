import argparse

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
from tgm.nn import NCNPredictor, TGNMemory
from tgm.nn.encoder.tgn import (
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
)
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TNCN LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument(
    '--k',
    type=int,
    default=2,
    choices=[2, 4, 8],
    help='k-th hop common neighbour (CN) embedding extraction (select from 2/4/8)',
)
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
    '--use-cn-time-decay',
    default=False,
    action=argparse.BooleanOptionalAction,
    help='indicate whether applying decay on time',
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

    memory.reset_state()

    for batch in tqdm(loader):
        opt.zero_grad()

        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID

        #! run my own deduplication
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
        z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

        inv_src = batch.global_to_local(batch.edge_src)
        inv_dst = batch.global_to_local(batch.edge_dst)
        inv_neg = batch.global_to_local(batch.neg)
        inv_edge_idx_pos = torch.stack([inv_src, inv_dst], dim=0).long()
        inv_edge_idx_neg = torch.stack([inv_src, inv_neg], dim=0).long()
        time_info = (last_update, batch.edge_time)

        pos_out = decoder(z, nbr_edge_index, inv_edge_idx_pos, time_info=time_info)
        neg_out = decoder(z, nbr_edge_index, inv_edge_idx_neg, time_info=time_info)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        # Update memory with ground-truth state.
        memory.update_state(
            batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float()
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

    for batch in tqdm(loader):
        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID

        #! run my own deduplication
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
        z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.edge_dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.edge_src[idx].repeat(len(dst_ids))

            inv_src = batch.global_to_local(src_ids)
            inv_dst = batch.global_to_local(dst_ids)
            inv_edge_idx = torch.stack([inv_src, inv_dst], dim=0)
            time_info = (
                last_update,
                batch.edge_time.repeat(len(inv_src)),
            )  # We can move this outside of the inner loop

            y_pred = decoder(
                z,
                nbr_edge_index,
                inv_edge_idx,
                time_info=time_info,
            ).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        # Update memory with ground-truth state.
        memory.update_state(
            batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float()
        )

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
train_data, val_data, test_data = full_data.split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

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

memory = TGNMemory(
    full_data.num_nodes,
    test_dg.edge_x_dim,
    args.memory_dim,
    args.time_dim,
    message_module=IdentityMessage(test_dg.edge_x_dim, args.memory_dim, args.time_dim),
    aggregator_module=LastAggregator(),
).to(args.device)
encoder = GraphAttentionEmbedding(
    in_channels=args.memory_dim,
    out_channels=args.embed_dim,
    msg_dim=test_dg.edge_x_dim,
    time_enc=memory.time_enc,
).to(args.device)
decoder = NCNPredictor(
    in_channels=args.embed_dim,
    hidden_dim=args.embed_dim,
    out_channels=1,
    k=args.k,
    cn_time_decay=args.use_cn_time_decay,
).to(args.device)
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

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, NeighborSamplerHook, RecencyNeighborHook
from tgm.nn import TGAT, NodePredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
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


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        opt.zero_grad()
        y_labels = batch.node_y
        if y_labels is not None:
            z = encoder(
                static_node_x,
                batch.seed_nids,
                batch.seed_times,
                batch.nbr_nids,
                batch.nbr_edge_x,
                batch.nbr_edge_time,
            )
            y_pred = decoder(z)

            loss = F.cross_entropy(y_pred, y_labels)
            loss.backward()
            opt.step()
            total_loss += float(loss)

    return total_loss


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        y_labels = batch.node_y
        if y_labels is not None:
            z = encoder(
                static_node_x,
                batch.seed_nids,
                batch.seed_times,
                batch.nbr_nids,
                batch.nbr_edge_x,
                batch.nbr_edge_time,
            )
            y_pred = decoder(z)
            input_dict = {
                'y_true': y_labels,
                'y_pred': y_pred,
                'eval_metric': [METRIC_TGB_NODEPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

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

num_classes = train_dg.node_y_dim

if args.sampling == 'uniform':
    nbr_hook = NeighborSamplerHook(
        num_nbrs=args.n_nbrs,
        seed_nodes_keys=['node_y_nids'],
        seed_times_keys=['node_y_time'],
    )
elif args.sampling == 'recency':
    nbr_hook = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=full_data.num_nodes,  # Assuming node ids at test set > train/val set
        seed_nodes_keys=['node_y_nids'],
        seed_times_keys=['node_y_time'],
    )
else:
    raise ValueError(f'Unknown sampling type: {args.sampling}')


hm = HookManager(keys=['train', 'val', 'test'])
hm.register_shared(nbr_hook)
train_key, val_key, test_key = hm.keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

encoder = TGAT(
    node_dim=train_dg.static_node_x_dim,
    edge_dim=train_dg.edge_x_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
).to(args.device)
decoder = NodePredictor(
    in_dim=args.embed_dim, out_dim=num_classes, hidden_dim=args.embed_dim
).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

best_val = 0.0

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, encoder, decoder, opt)

    with hm.activate(val_key):
        val_mrr = eval(val_loader, encoder, decoder, evaluator)
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_mrr, epoch=epoch)
    if val_mrr > best_val:
        best_val = val_mrr
        with hm.activate(test_key):
            test_mrr = eval(test_loader, encoder, decoder, evaluator)
        log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_mrr, epoch=args.epochs)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

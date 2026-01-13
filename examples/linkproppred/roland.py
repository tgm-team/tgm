import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph, TimeDeltaDG
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecipeRegistry
from tgm.nn import ROLAND, LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='ROLAND LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--embed-dim', type=int, default=256, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout for ROLAND')
parser.add_argument(
    '--tau', type=float, default=0.5, help='tau for updating embeddings'
)
parser.add_argument(
    '--update', type=str, default='learnable', help='embedding update mechanism'
)
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='h',
    help='time granularity to operate on for snapshots',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class RecurrentGCN(torch.nn.Module):
    def __init__(
        self,
        input_channel: int,
        nhid: int,
        num_nodes: int,
        dropout: float = 0.0,
        update: str = 'learnable',
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.recurrent = ROLAND(input_channel, nhid, num_nodes, dropout, update, tau)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.tensor,
        last_embeddings: List[torch.Tensor] | None = None,
        num_current_edges: int | None = None,
        num_previous_edges: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)

        h_0 = self.recurrent(
            node_feat,
            edge_index,
            last_embeddings,
            num_current_edges,
            num_previous_edges,
        )
        return h_0


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    conversion_rate: int,
    last_embeddings: list,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_x = loader.dgraph.static_node_x

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    prev_num_edge = None
    curr_num_edge = snapshot_batch.src.numel()

    z = encoder(
        snapshot_batch,
        static_node_x,
        last_embeddings,
        num_previous_edges=prev_num_edge,
        num_current_edges=curr_num_edge,
    )

    z[0], z[1] = z[0].detach(), z[1].detach()

    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out = decoder(z[-1][batch.src], z[-1][batch.dst])
        neg_out = decoder(z[-1][batch.src], z[-1][batch.neg])

        loss = F.mse_loss(pos_out, torch.ones_like(pos_out))
        loss += F.mse_loss(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss) / batch.src.shape[0]

        # update the model if the prediction batch has moved to next snapshot.
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                prev_num_edge = curr_num_edge
                curr_num_edge = snapshot_batch.src.numel()
                z = encoder(
                    snapshot_batch,
                    static_node_x,
                    last_embeddings,
                    num_previous_edges=prev_num_edge,
                    num_current_edges=curr_num_edge,
                )
                last_embeddings = z
                z[0], z[1] = z[0].detach(), z[1].detach()
            except StopIteration:
                pass

    return total_loss, last_embeddings


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    z: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
    conversion_rate: int,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    prev_num_edge = None
    curr_num_edge = snapshot_batch.src.numel()

    for batch in tqdm(loader):
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = decoder(z[-1][query_src], z[-1][query_dst])
            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        # update the model if the prediction batch has moved to next snapshot.
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                prev_num_edge = curr_num_edge
                curr_num_edge = snapshot_batch.src.numel()
                z = encoder(
                    snapshot_batch,
                    static_node_x,
                    z,
                    num_previous_edges=prev_num_edge,
                    num_current_edges=curr_num_edge,
                )
            except StopIteration:
                pass

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
if full_data.static_node_x is None:
    full_data.static_node_x = torch.randn(
        (full_data.num_nodes, args.node_dim), device=args.device
    )

train_data, val_data, test_data = full_data.split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

snapshot_td = TimeDeltaDG(args.snapshot_time_gran)
conversion_rate = int(snapshot_td.convert(train_dg.time_delta))

train_data_discretized = train_data.discretize(args.snapshot_time_gran)
val_data_discretized = val_data.discretize(args.snapshot_time_gran)
test_data_discretized = test_data.discretize(args.snapshot_time_gran)

train_snapshots = DGraph(train_data_discretized, device=args.device)
val_snapshots = DGraph(val_data_discretized, device=args.device)
test_snapshots = DGraph(test_data_discretized, device=args.device)

_, dst, _ = train_dg.edge_events

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
registered_keys = hm.keys
train_key, val_key, test_key = registered_keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

train_snapshots_loader = DGDataLoader(
    train_snapshots, batch_unit=args.snapshot_time_gran, on_empty='raise'
)
val_snapshots_loader = DGDataLoader(
    val_snapshots, batch_unit=args.snapshot_time_gran, on_empty='raise'
)
test_snapshots_loader = DGDataLoader(
    test_snapshots, batch_unit=args.snapshot_time_gran, on_empty='raise'
)

encoder = RecurrentGCN(
    input_channel=train_dg.static_node_x_dim,
    num_nodes=full_data.num_nodes,
    nhid=args.embed_dim,
    dropout=args.dropout,
    update=args.update,
    tau=args.tau,
).to(args.device)
decoder = LinkPredictor(args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    last_embeddings = [
        torch.zeros(full_data.num_nodes, args.embed_dim, device=args.device),
        torch.zeros(full_data.num_nodes, args.embed_dim, device=args.device),
    ]
    with hm.activate(train_key):
        start_time = time.perf_counter()
        loss, last_embeddings = train(
            train_loader,
            train_snapshots_loader,
            encoder,
            decoder,
            opt,
            conversion_rate,
            last_embeddings,
        )
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate(val_key):
        val_mrr = eval(
            val_loader,
            val_snapshots_loader,
            last_embeddings,
            encoder,
            decoder,
            evaluator,
            conversion_rate,
        )
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

with hm.activate(test_key):
    test_mrr = eval(
        test_loader,
        test_snapshots_loader,
        last_embeddings,
        encoder,
        decoder,
        evaluator,
        conversion_rate,
    )
    print(f'Test {METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')

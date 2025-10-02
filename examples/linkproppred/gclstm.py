import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph, RecipeRegistry
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.loader import DGDataLoader
from tgm.nn import LinkPredictor
from tgm.nn.recurrent import GCLSTM
from tgm.timedelta import TimeDeltaDG
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCLSTM LinkPropPred Example',
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
    def __init__(self, node_dim: int, embed_dim: int, K: int = 1) -> None:
        super().__init__()
        self.recurrent = GCLSTM(in_channels=node_dim, out_channels=embed_dim, K=K)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None  # type: ignore

        h_0, c_0 = self.recurrent(node_feat, edge_index, edge_weight, h, c)
        z = F.relu(h_0)
        z = self.linear(z)
        return z, h_0, c_0


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
    conversion_rate: int,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)
    z, h_0, c_0 = encoder(snapshot_batch, static_node_feats)
    z, h_0, c_0 = z.detach(), h_0.detach(), c_0.detach()

    for batch in tqdm(loader):
        opt.zero_grad()

        pos_out = decoder(z[batch.src], z[batch.dst])
        neg_out = decoder(z[batch.src], z[batch.neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss) / batch.src.shape[0]

        # update the model if the prediction batch has moved to next snapshot.
        while batch.time[-1] > (snapshot_batch.time[-1] + 1) * conversion_rate:
            try:
                snapshot_batch = next(snapshots_iterator)
                z, h_0, c_0 = encoder(snapshot_batch, static_node_feats, h_0, c_0)
                z, h_0, c_0 = z.detach(), h_0.detach(), c_0.detach()  # Truncate BPTT
            except StopIteration:
                pass

    return total_loss, z, h_0, c_0


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    z: torch.Tensor,
    h_0: torch.Tensor,
    c_0: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
    conversion_rate: int,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    snapshots_iterator = iter(snapshots_loader)
    snapshot_batch = next(snapshots_iterator)

    for batch in tqdm(loader):
        neg_batch_list = batch.neg_batch_list
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = batch.src[idx].repeat(len(neg_batch) + 1)
            query_dst = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])

            y_pred = decoder(z[query_src], z[query_dst]).sigmoid()
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
                z, h_0, c_0 = encoder(snapshot_batch, static_node_feats, h_0, c_0)
            except StopIteration:
                pass

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
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

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

train_snapshots_loader = DGDataLoader(
    train_snapshots, batch_unit=args.snapshot_time_gran
)
val_snapshots_loader = DGDataLoader(val_snapshots, batch_unit=args.snapshot_time_gran)
test_snapshots_loader = DGDataLoader(test_snapshots, batch_unit=args.snapshot_time_gran)


if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

encoder = RecurrentGCN(
    node_dim=static_node_feats.shape[1], embed_dim=args.embed_dim
).to(args.device)
decoder = LinkPredictor(
    node_dim=args.embed_dim, out_dim=1, hids_sizes=args.embed_dim
).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss, z, h_0, c_0 = train(
            train_loader,
            train_snapshots_loader,
            static_node_feats,
            encoder,
            decoder,
            opt,
            conversion_rate,
        )

    with hm.activate(val_key):
        val_mrr = eval(
            val_loader,
            val_snapshots_loader,
            static_node_feats,
            z,
            h_0,
            c_0,
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
        static_node_feats,
        z,
        h_0,
        c_0,
        encoder,
        decoder,
        evaluator,
        conversion_rate,
    )
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

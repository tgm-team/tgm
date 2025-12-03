import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph, TimeDeltaDG
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecipeRegistry
from tgm.nn import EvolveGCNO, LinkPredictor
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

# from tgm.nn import EvolveGCNH, EvolveGCNO, LinkPredictor


torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser(
    description='EvolveGCN LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
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
parser.add_argument(
    '--mode',
    type=str,
    default='o',
    choices=['o', 'h'],
    help='To use EvolveGCNO or EvolveGCNH',
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class RecurrentGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        mode: str,
        hidden_dim: int,
        num_nodes: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ['o', 'h']:
            raise ValueError(
                'EvolveGCN mode must be either o or h , currently received,', mode
            )
        if mode == 'o':
            self.recurrent = EvolveGCNO(
                in_channels=in_channels,
                improved=improved,
                cached=cached,
                normalize=normalize,
                add_self_loops=add_self_loops,
            )

        # if mode == 'h':
        #     self.recurrent = EvolveGCNH(
        #         num_nodes=num_nodes,
        #         in_channels=in_channels,
        #         improved=improved,
        #         cached=cached,
        #         normalize=normalize,
        #         add_self_loops=add_self_loops,
        #     )
        self.linear = torch.nn.Linear(in_channels, hidden_dim)

    # batch: DGBatch,
    def forward(
        self,
        edge_index,
        node_feat: torch.tensor,
    ) -> torch.Tensor:
        # if edge_index is None:
        #     edge_index = torch.stack([batch.src, batch.dst], dim=0)
        # edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else torch.ones(edge_index.size(1), 1).to(args.device)
        edge_weight = torch.ones(edge_index.size(1), 1).to(args.device)
        h = self.recurrent(node_feat.to(torch.float), edge_index, edge_weight)
        z = F.relu(h)
        z = self.linear(z)
        return z


# @log_gpu
# @log_latency
# def train(
#     loader: DGDataLoader,
#     snapshots_loader: DGDataLoader,
#     static_node_feats: torch.Tensor,
#     encoder: nn.Module,
#     decoder: nn.Module,
#     opt: torch.optim.Optimizer,
#     conversion_rate: int,
#     dst_min: int,
#     dst_max: int,
# ):
#     opt.zero_grad()
#     encoder.train()
#     decoder.train()
#     total_loss = 0
#     all_snapshots = []
#     h = None
#     for batch in snapshots_loader:
#         all_snapshots.append(torch.stack([batch.src, batch.dst], dim=0))

#     for i in tqdm(range(len(all_snapshots))):
#         if i == 0:  # first snapshot, feed the current snapshot
#             cur_index = all_snapshots[0].long()
#             encoder.recurrent.weight = None
#             h = encoder(cur_index, static_node_feats)
#         else:  # subsequent snapshot, feed the previous snapshot
#             prev_index = all_snapshots[i - 1]
#             h = encoder(prev_index, static_node_feats)

#         pos_index = all_snapshots[i]
#         pos_index = pos_index.long()

#         neg_dst = torch.randint(
#             dst_min,
#             dst_max,
#             (pos_index.shape[1],),
#             dtype=torch.long,
#             device=args.device,
#         )

#         pos_out = decoder(h[pos_index[0]], h[pos_index[1]])
#         neg_out = decoder(h[pos_index[0]], h[neg_dst])

#         total_loss += F.binary_cross_entropy_with_logits(
#             pos_out, torch.ones_like(pos_out)
#         )
#         total_loss += F.binary_cross_entropy_with_logits(
#             neg_out, torch.zeros_like(neg_out)
#         )

#     total_loss.backward()
#     opt.step()

#     # cutoff the graph of the hidden state between epochs
#     encoder.recurrent.weight = encoder.recurrent.weight.detach()
#     loss = float(total_loss.item()) / len(all_snapshots)
#     h = h.detach()
#     return loss, h


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
    dst_min: int,
    dst_max: int,
):
    encoder.train()
    decoder.train()
    total_loss = 0
    prev_index = 0
    h = None
    all_snapshots = []
    loss = 0
    for batch in snapshots_loader:
        all_snapshots.append(torch.stack([batch.src, batch.dst], dim=0))

    for i in tqdm(range(len(all_snapshots))):
        opt.zero_grad()
        if i == 0:  # first snapshot, feed the current snapshot
            encoder.recurrent.weight = None
            cur_index = all_snapshots[0].long()
            h = encoder(cur_index, static_node_feats)
        else:  # subsequent snapshot, feed the previous snapshot
            prev_index = all_snapshots[i - 1]
            h = encoder(prev_index, static_node_feats)
        h = h.detach()
        pos_index = all_snapshots[i].long().detach()

        neg_dst = torch.randint(
            dst_min,
            dst_max,
            (pos_index.shape[1],),
            dtype=torch.long,
            device=args.device,
        )

        pos_out = decoder(h[pos_index[0]], h[pos_index[1]])
        neg_out = decoder(h[pos_index[0]], h[neg_dst])

        loss += F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) / pos_index.shape[1]
        loss = 0

    encoder.recurrent.weight = encoder.recurrent.weight.detach()
    h = h.detach()
    return total_loss, h


"""
for snapshot_idx in range(train_data['time_length']):
                # neg_edges = negative_sampling(pos_index, num_nodes=num_nodes, num_neg_samples=(pos_index.size(1)*1), force_undirected = True)
                if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                    cur_index = snapshot_list[snapshot_idx]
                    cur_index = cur_index.long().to(args.device)
                    # TODO, also need to support edge attributes correctly in TGX
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h = model(node_feat, cur_index, edge_attr)
                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    if ('edge_attr' not in train_data):
                        edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h = model(node_feat, prev_index, edge_attr)

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)

                neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_index.shape[1],),
                        dtype=torch.long,
                        device=args.device,
                    )
                pos_out = link_pred(h[pos_index[0]], h[pos_index[1]])
                neg_out = link_pred(h[pos_index[0]], h[neg_dst])

                total_loss += criterion(pos_out, torch.ones_like(pos_out))
                total_loss += criterion(neg_out, torch.zeros_like(neg_out))

            total_loss.backward()
            optimizer.step()
            num_snapshots = train_data['time_length']

"""


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    snapshots_loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    z: torch.Tensor,
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
                z = encoder(
                    torch.stack([snapshot_batch.src, snapshot_batch.dst], dim=0),
                    static_node_feats,
                )
                # z = encoder(snapshot_batch, static_node_feats)
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
_, dst, _ = train_dg.edges

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
    in_channels=static_node_feats.shape[1],
    mode=args.mode,
    hidden_dim=args.embed_dim,
    num_nodes=static_node_feats.shape[0],  # required for EvolveGCNH
).to(args.device)
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)

opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

dst_min = int(dst.min())
dst_max = int(dst.max())

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss, z = train(
            train_loader,
            train_snapshots_loader,
            static_node_feats,
            encoder,
            decoder,
            opt,
            conversion_rate,
            dst_min=dst_min,
            dst_max=dst_max,
        )
        print('training loss is', loss)
        log_metric('Loss', loss, epoch=epoch)

    if epoch % 10 == 0:
        with hm.activate(val_key):
            val_mrr = eval(
                val_loader,
                val_snapshots_loader,
                static_node_feats,
                z,
                encoder,
                decoder,
                evaluator,
                conversion_rate,
            )
        log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)


with hm.activate(test_key):
    test_mrr = eval(
        test_loader,
        test_snapshots_loader,
        static_node_feats,
        z,
        encoder,
        decoder,
        evaluator,
        conversion_rate,
    )
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

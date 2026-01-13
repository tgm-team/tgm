import argparse
import copy
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry
from tgm.nn import DyGFormer, LinkPredictor, Time2Vec
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='DyGFormers TGB Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--max_sequence_length',
    type=int,
    default=32,
    help='maximal length of the input sequence of each node',
)
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--time_dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed_dim', type=int, default=172, help='attention dimension')
parser.add_argument('--node_dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--channel-embedding-dim',
    type=int,
    default=50,
    help='dimension of each channel embedding',
)
parser.add_argument('--patch-size', type=int, default=1, help='patch size')
parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
parser.add_argument(
    '--num_heads', type=int, default=2, help='number of heads used in attention layer'
)
parser.add_argument(
    '--num-channels',
    type=int,
    default=4,
    help='number of channels used in attention layer',
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class DyGFormer_LinkPrediction(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        channel_embedding_dim: int,
        output_dim: int = 172,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        num_channels: int = 4,
        time_encoder: Callable[..., nn.Module] = Time2Vec,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.encoder = DyGFormer(
            node_feat_dim,
            edge_feat_dim,
            time_feat_dim,
            channel_embedding_dim,
            output_dim,
            patch_size,
            num_layers,
            num_heads,
            dropout,
            max_input_sequence_length,
            num_channels,
            time_encoder,
            device,
        )
        # @TODO: Make encoder/decoder to be explicit
        self.decoder = LinkPredictor(
            node_dim=args.embed_dim, hidden_dim=args.embed_dim
        ).to(args.device)

    def forward(
        self, batch: DGBatch, static_node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = batch.src
        dst = batch.dst
        neg = batch.neg
        time = batch.edge_event_time
        nbr_nids = batch.nbr_nids[0]
        nbr_times = batch.nbr_times[0]
        nbr_feats = batch.nbr_feats[0]
        src_nbr_idx = batch.seed_node_nbr_mask['src']
        dst_nbr_idx = batch.seed_node_nbr_mask['dst']
        neg_nbr_idx = batch.seed_node_nbr_mask['neg']
        pos_batch_size = dst.shape[0]
        neg_batch_size = neg.shape[0]

        # positive edge
        edge_idx_pos = torch.stack((src, dst), dim=0)
        pos_nbr_idx = torch.cat([src_nbr_idx, dst_nbr_idx])
        z_src_pos, z_dst_pos = self.encoder(
            static_node_feat,
            edge_idx_pos,
            time,
            nbr_nids[pos_nbr_idx],
            nbr_times[pos_nbr_idx],
            nbr_feats[pos_nbr_idx],
        )
        pos_out = self.decoder(z_src_pos, z_dst_pos)

        neg_nbr_nids = nbr_nids[
            neg_nbr_idx
        ]  # @TODO: Assume that batch.neg doesn't have duplicated records
        neg_nbr_times = nbr_times[neg_nbr_idx]
        neg_nbr_feats = nbr_feats[neg_nbr_idx]
        src_nbr_nids = nbr_nids[src_nbr_idx]
        src_nbr_times = nbr_times[src_nbr_idx]
        src_nbr_feats = nbr_feats[src_nbr_idx]

        if src.shape[0] != neg_batch_size:
            src = torch.repeat_interleave(src, repeats=neg_batch_size, dim=0)
            time = torch.repeat_interleave(time, repeats=neg_batch_size, dim=0)
            src_nbr_nids = torch.repeat_interleave(
                src_nbr_nids, repeats=neg_batch_size, dim=0
            )
            src_nbr_times = torch.repeat_interleave(
                src_nbr_times, repeats=neg_batch_size, dim=0
            )
            src_nbr_feats = torch.repeat_interleave(
                src_nbr_feats, repeats=neg_batch_size, dim=0
            )
            neg_nbr_nids = neg_nbr_nids.repeat(pos_batch_size, 1)
            neg_nbr_times = neg_nbr_times.repeat(pos_batch_size, 1)
            neg_nbr_feats = neg_nbr_feats.repeat(pos_batch_size, 1, 1)
            neg = neg.repeat(pos_batch_size)
        else:
            src_nbr_nids = nbr_nids[src_nbr_idx]
            src_nbr_times = nbr_times[src_nbr_idx]
            src_nbr_feats = nbr_feats[src_nbr_idx]

        edge_idx_neg = torch.stack((src, neg), dim=0)

        # negative edge
        z_src_neg, z_dst_neg = self.encoder(
            static_node_feat,
            edge_idx_neg,
            time,
            torch.cat([src_nbr_nids, neg_nbr_nids], dim=0),
            torch.cat([src_nbr_times, neg_nbr_times], dim=0),
            torch.cat([src_nbr_feats, neg_nbr_feats], dim=0),
        )
        neg_out = self.decoder(z_src_neg, z_dst_neg)

        return pos_out, neg_out


@log_gpu
@log_latency
def train(loader: DGDataLoader, model: nn.Module, opt: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch, static_node_x)

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
    evaluator: Evaluator,
    loader: DGDataLoader,
    model: nn.Module,
) -> float:
    model.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        copy_batch = copy.deepcopy(batch)
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            idx = torch.tensor([idx], device=args.device)
            copy_batch.src = batch.src[idx]
            copy_batch.dst = batch.dst[idx]
            copy_batch.time = batch.edge_event_time[idx]
            copy_batch.neg = neg_batch
            neg_idx = (batch.neg == neg_batch[:, None]).nonzero(as_tuple=True)[1]

            # Update nbr map to only indices that are used
            copy_batch.seed_node_nbr_mask['src'] = batch.seed_node_nbr_mask['src'][idx]
            copy_batch.seed_node_nbr_mask['dst'] = batch.seed_node_nbr_mask['dst'][idx]
            copy_batch.seed_node_nbr_mask['neg'] = batch.seed_node_nbr_mask['neg'][
                neg_idx
            ]

            pos_out, neg_out = model(copy_batch, static_node_x)
            pos_out, neg_out = pos_out.sigmoid(), neg_out.sigmoid()

            input_dict = {
                'y_pred_pos': pos_out,
                'y_pred_neg': neg_out,
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

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

nbr_hook = RecencyNeighborHook(
    num_nbrs=[args.max_sequence_length - 1],  # 1 remaining for seed node itself
    num_nodes=full_data.num_nodes,
    seed_nodes_keys=['src', 'dst', 'neg'],
    seed_times_keys=['edge_event_time', 'edge_event_time', 'neg_time'],
)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
hm.register_shared(nbr_hook)
train_key, val_key, test_key = hm.keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

model = DyGFormer_LinkPrediction(
    node_feat_dim=train_dg.static_node_x_dim,
    edge_feat_dim=train_dg.edge_x_dim,
    time_feat_dim=args.time_dim,
    channel_embedding_dim=args.channel_embedding_dim,
    output_dim=args.embed_dim,
    max_input_sequence_length=args.max_sequence_length,
    dropout=args.dropout,
    num_heads=args.num_heads,
    num_channels=args.num_channels,
    num_layers=args.num_layers,
    device=args.device,
    patch_size=args.patch_size,
).to(args.device)

opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, model, opt)
    with hm.activate(val_key):
        val_mrr = eval(evaluator, val_loader, model)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    # Clear memory state between epochs, except last epoch
    if epoch < args.epochs:
        hm.reset_state()

with hm.activate(test_key):
    test_mrr = eval(evaluator, test_loader, model)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

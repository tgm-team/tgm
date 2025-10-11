import argparse
import copy
import logging
import os
from pathlib import Path
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
from tgm.nn import LinkPredictor, RandomProjectionModule, Time2Vec, TPNet
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TPNet LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument(
    '--num-neighbors',
    type=int,
    default=32,
    help='number of recency temporal neighbors of each node',
)
parser.add_argument(
    '--rp-num-layers',
    type=int,
    default=2,
    help='the number of layer of random projection module',
)
parser.add_argument(
    '--rp-time-decay-weight',
    type=float,
    default=0.000001,
    help='the first weight of the time decay',
)
parser.add_argument(
    '--enforce-dim',
    type=int,
    default=None,
    help='enforced dimension of random projections',
)
parser.add_argument(
    '--rp-dim-factor',
    type=int,
    default=10,
    help='the dim factor of random feature w.r.t. the node num',
)
parser.add_argument(
    '--not-use-matrix',
    default=False,
    action='store_true',
    help='if False, explicitly maintain the temporal walk matrices',
)
parser.add_argument(
    '--not-concat-src-dst',
    default=False,
    action='store_true',
    help='if True, Random projection avoids concat src and dst in computation',
)
parser.add_argument('--node-dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument(
    '--embed-dim', type=int, default=172, help='node representation dimension'
)
parser.add_argument('--num-layers', type=int, default=2, help='number of model layers')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)
logger = logging.getLogger(f'tgm.{Path(__file__).stem}')


class TPNet_LinkPrediction(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        num_neighbors: int,
        random_projection_module: RandomProjectionModule | None = None,
        device: str = 'cpu',
        time_encoder: Callable[..., nn.Module] = Time2Vec,
    ) -> None:
        super().__init__()
        self.encoder = TPNet(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            time_feat_dim=time_feat_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            random_projections=random_projection_module,
            device=device,
            time_encoder=time_encoder,
        )
        self.rp_module = random_projection_module.to(device)
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
        time = batch.time
        nbr_nids = batch.nbr_nids[0]
        nbr_times = batch.nbr_times[0]
        nbr_feats = batch.nbr_feats[0]
        pos_batch_size = dst.shape[0]
        neg_batch_size = neg.shape[0]

        # positive edge
        edge_idx_pos = torch.stack((src, dst), dim=0)
        z_src_pos, z_dst_pos = self.encoder(
            static_node_feat,
            edge_idx_pos,
            time,
            nbr_nids[: pos_batch_size * 2],
            nbr_times[: pos_batch_size * 2],
            nbr_feats[: pos_batch_size * 2],
        )
        pos_out = self.decoder(z_src_pos, z_dst_pos)

        neg_nbr_nids = nbr_nids[
            -neg_batch_size:
        ]  # @TODO: Assume that batch.neg doesn't have duplicated records
        neg_nbr_times = nbr_times[-neg_batch_size:]
        neg_nbr_feats = nbr_feats[-neg_batch_size:]
        src_nbr_nids = nbr_nids[:pos_batch_size]
        src_nbr_times = nbr_times[:pos_batch_size]
        src_nbr_feats = nbr_feats[:pos_batch_size]

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
            src_nbr_nids = nbr_nids[:pos_batch_size]
            src_nbr_times = nbr_times[:pos_batch_size]
            src_nbr_feats = nbr_feats[:pos_batch_size]

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
        self.rp_module.update(batch.src, batch.dst, time=batch.time)

        return pos_out, neg_out


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    static_node_feat: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch, static_node_feat)

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
    static_node_feat: torch.Tensor,
) -> float:
    model.eval()
    perf_list = []
    max_eval_batches_per_epoch = os.getenv('TGM_CI_MAX_EVAL_BATCHES_PER_EPOCH')

    for batch_num, batch in enumerate(tqdm(loader)):
        copy_batch = copy.deepcopy(batch)
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            copy_batch.src = batch.src[idx].unsqueeze(0)
            copy_batch.dst = batch.dst[idx].unsqueeze(0)
            copy_batch.time = batch.time[idx].unsqueeze(0)
            copy_batch.neg = neg_batch
            neg_idx = (batch.neg == neg_batch[:, None]).nonzero(as_tuple=True)[1]

            # A tensor of index of src, dst and negative nodes to retrieve neighbor information
            all_idx = torch.cat(
                [
                    torch.Tensor([idx]).to(neg_batch.device),  # src idx
                    torch.Tensor([idx + batch.src.shape[0]]).to(
                        neg_batch.device
                    ),  # dst idx
                    neg_idx,
                ],
                dim=0,
            ).long()
            copy_batch.nbr_nids = [batch.nbr_nids[0][all_idx]]
            copy_batch.nbr_times = [batch.nbr_times[0][all_idx]]
            copy_batch.nbr_feats = [batch.nbr_feats[0][all_idx]]

            pos_out, neg_out = model(copy_batch, static_node_feat)
            pos_out, neg_out = pos_out.sigmoid(), neg_out.sigmoid()

            input_dict = {
                'y_pred_pos': pos_out,
                'y_pred_neg': neg_out,
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        if max_eval_batches_per_epoch and batch_num >= int(max_eval_batches_per_epoch):
            logger.warning(
                f'Exiting evaluation prematurely since max_eval_batches_per_epoch ({max_eval_batches_per_epoch}) was reached. Reported performance may not reflect ground truth on the entire dataset.'
            )
            break

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

if train_dg.static_node_feats is not None:
    static_node_feat = train_dg.static_node_feats
else:
    static_node_feat = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )


hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
hm.register_shared(
    RecencyNeighborHook(
        num_nbrs=[args.num_neighbors],
        num_nodes=test_dg.num_nodes,
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['time', 'time', 'neg_time'],
    )
)
train_key, val_key, test_key = hm.keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

random_projection_module = RandomProjectionModule(
    num_nodes=test_dg.num_nodes,
    num_layer=args.rp_num_layers,
    time_decay_weight=args.rp_time_decay_weight,
    beginning_time=train_dg.start_time,
    use_matrix=not args.not_use_matrix,
    enforce_dim=args.enforce_dim,
    num_edges=train_dg.num_edges,
    dim_factor=args.rp_dim_factor,
    concat_src_dst=not args.not_concat_src_dst,
    device=args.device,
)

model = TPNet_LinkPrediction(
    node_feat_dim=static_node_feat.shape[1],
    edge_feat_dim=train_dg.edge_feats_dim,
    time_feat_dim=args.time_dim,
    output_dim=args.embed_dim,
    dropout=args.dropout,
    num_layers=args.num_layers,
    num_neighbors=args.num_neighbors,
    random_projection_module=random_projection_module,
    device=args.device,
    time_encoder=Time2Vec,
)

opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, model, opt, static_node_feat)
    with hm.activate(val_key):
        val_mrr = eval(evaluator, val_loader, model, static_node_feat)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    # Clear memory state between epochs, except last epoch
    if epoch < args.epochs:
        hm.reset_state()
        model.rp_module.reset_random_projections()

with hm.activate(test_key):
    test_mrr = eval(evaluator, test_loader, model, static_node_feat)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

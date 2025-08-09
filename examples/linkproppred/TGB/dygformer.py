import argparse
import copy
import time
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm.graph import DGBatch, DGraph
from tgm.hooks import (
    DGHook,
    NegativeEdgeSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader
from tgm.nn import DyGFormer, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='DyGFormers TGB Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--max_sequence_length',
    type=int,
    default=40,
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
parser.add_argument('--patch-size', type=int, default=8, help='patch size')
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


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_out = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_out(h).sigmoid().view(-1)


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
        self.decoder = LinkPredictor(output_dim)

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
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
            STATIC_NODE_FEAT,
            edge_idx_pos,
            time,
            nbr_nids[: pos_batch_size * 2],
            nbr_times[: pos_batch_size * 2],
            nbr_feats[: pos_batch_size * 2],
        )
        pos_out = self.decoder(z_src_pos, z_dst_pos)

        neg_nbr_nids = nbr_nids[-neg_batch_size:]
        neg_nbr_times = nbr_times[-neg_batch_size:]
        neg_nbr_feats = nbr_feats[-neg_batch_size:]

        if src.shape[0] != neg_batch_size:
            src = torch.repeat_interleave(src, repeats=neg_batch_size, dim=0)
            time = torch.repeat_interleave(time, repeats=neg_batch_size, dim=0)
            src_nbr_nids = torch.repeat_interleave(
                nbr_nids[:pos_batch_size], repeats=neg_batch_size, dim=0
            )
            src_nbr_times = torch.repeat_interleave(
                nbr_times[:pos_batch_size], repeats=neg_batch_size, dim=0
            )
            src_nbr_feats = torch.repeat_interleave(
                nbr_feats[:pos_batch_size], repeats=neg_batch_size, dim=0
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
            STATIC_NODE_FEAT,
            edge_idx_neg,
            time,
            torch.cat([src_nbr_nids, neg_nbr_nids], dim=0),
            torch.cat([src_nbr_times, neg_nbr_times], dim=0),
            torch.cat([src_nbr_feats, neg_nbr_feats], dim=0),
        )
        neg_out = self.decoder(z_src_neg, z_dst_neg)

        return pos_out, neg_out


def _init_hooks(dg: DGraph, neg_sampler: object, split_mode='train') -> List[DGHook]:
    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.max_sequence_length - 1],  # 1 remaining for seed node itself
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )

    # Always produce negative edge prior to neighbor sampling for link prediction
    if split_mode in ['val', 'test']:
        neg_hook = TGBNegativeEdgeSamplerHook(neg_sampler, split_mode=split_mode)
    else:
        _, dst, _ = dg.edges
        min_dst, max_dst = int(dst.min()), int(dst.max())
        neg_hook = NegativeEdgeSamplerHook(low=min_dst, high=max_dst)

    return [neg_hook, nbr_hook]


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    evaluator: Evaluator,
    loader: DGDataLoader,
    model: nn.Module,
    eval_metric: str,
) -> dict:
    model.eval()
    perf_list = []
    for batch in tqdm(loader):
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            copy_batch = copy.deepcopy(batch)

            copy_batch.src = batch.src[idx].unsqueeze(0)
            copy_batch.dst = batch.dst[idx].unsqueeze(0)
            copy_batch.time = batch.time[idx].unsqueeze(0)
            copy_batch.neg = neg_batch
            neg_idx = (batch.neg == neg_batch[:, None]).nonzero(as_tuple=True)[1]

            all_idx = torch.cat(
                [
                    torch.Tensor([idx]).to(neg_batch.device),
                    torch.Tensor([idx + batch.src.shape[0]]).to(neg_batch.device),
                    neg_idx,
                ],
                dim=0,
            ).long()
            copy_batch.nbr_nids = [batch.nbr_nids[0][all_idx]]
            copy_batch.nbr_times = [batch.nbr_times[0][all_idx]]
            copy_batch.nbr_feats = [batch.nbr_feats[0][all_idx]]

            pos_out, neg_out = model(copy_batch)

            input_dict = {
                'y_pred_pos': pos_out.detach().cpu().numpy(),
                'y_pred_neg': neg_out.detach().cpu().numpy(),
                'eval_metric': [eval_metric],
            }
            perf_list.append(evaluator.eval(input_dict)[eval_metric])

    metric_dict = {}
    metric_dict[eval_metric] = float(np.mean(perf_list))
    return metric_dict


args = parser.parse_args()
seed_everything(args.seed)

dataset = PyGLinkPropPredDataset(name=args.dataset, root='datasets')
eval_metric = dataset.eval_metric
neg_sampler = dataset.negative_sampler

neg_sampler = dataset.negative_sampler
evaluator = Evaluator(name=args.dataset)
dataset.load_val_ns()
dataset.load_test_ns()

dgraph = DGraph(args.dataset)
train_dg = DGraph(args.dataset, split='train', device=args.device)
val_dg = DGraph(args.dataset, split='val', device=args.device)
test_dg = DGraph(args.dataset, split='test', device=args.device)

# TODO: Read from graph
num_nodes = dgraph.num_nodes
edge_feats_dim = dgraph.edge_feats_dim
label_dim = train_dg.dynamic_node_feats_dim
STATIC_NODE_FEAT = torch.randn((num_nodes, args.node_dim), device=args.device)


test_loader = DGDataLoader(
    test_dg,
    batch_size=args.bsize,
    hook=_init_hooks(dg=test_dg, neg_sampler=neg_sampler, split_mode='test'),
)


model = DyGFormer_LinkPrediction(
    node_feat_dim=args.node_dim,
    edge_feat_dim=edge_feats_dim,
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
    # TODO: Need a clean way to clear nbr state across epochs
    train_loader = DGDataLoader(
        train_dg,
        hook=_init_hooks(dg=train_dg, neg_sampler=neg_sampler, split_mode='train'),
        batch_size=args.bsize,
    )
    val_loader = DGDataLoader(
        val_dg,
        hook=_init_hooks(dg=test_dg, neg_sampler=neg_sampler, split_mode='val'),
        batch_size=args.bsize,
    )

    start_time = time.perf_counter()
    val_results = eval(evaluator, val_loader, model, eval_metric)
    loss = train(train_loader, model, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, encoder, decoder, eval_metric, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import (
    METRIC_TGB_LINKPROPPRED,
    PADDED_NODE_ID,
    RECIPE_TGB_LINK_PRED,
)
from tgm.data import DGData, DGDataLoader
from tgm.hooks import NeighborSamplerHook, RecencyNeighborHook, RecipeRegistry
from tgm.nn import LinkPredictor, TemporalAttention, Time2Vec
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
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


class MergeLayer(nn.Module):
    def __init__(self, in_dim1: int, in_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim1 + in_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h = self.fc1(torch.cat([x1, x2], dim=1))
        h = h.relu()
        return self.fc2(h)


class TGAT(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """In this implementation, the node embedding dimension must be the same as hidden embedding dimension."""
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.time_encoder = Time2Vec(time_dim=time_dim)

        self.attn, self.merge_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers):
            self.attn.append(
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=node_dim if i == 0 else embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    dropout=dropout,
                )
            )
            self.merge_layers.append(
                MergeLayer(
                    in_dim1=self.attn[-1].out_dim,
                    in_dim2=node_dim,
                    hidden_dim=embed_dim,
                    output_dim=embed_dim,
                )
            )

    def forward(self, batch: DGBatch, static_node_feat: torch.Tensor) -> torch.Tensor:
        device = batch.src.device
        z = {j: {} for j in range(self.num_layers + 1)}  # z[j][i] = z of nbr^i at hop j

        # Layer 0 (leaf nodes): z[0][i] = static_node_feat
        z[0][0] = static_node_feat[batch.nids[0]]
        for i in range(1, self.num_layers + 1):
            z[0][i] = static_node_feat[batch.nbr_nids[i - 1].flatten()]

        # Layers 1..H: aggregate z[j][i] = agg(z[j - 1][i], z[j - 1][i + 1])
        for j in range(1, self.num_layers + 1):
            for i in range(self.num_layers - j + 1):
                num_nodes = z[j - 1][i].size(0)
                num_nbr = batch.nbr_nids[j - 1].shape[-1]
                out = self.attn[j - 1](
                    node_feat=z[j - 1][i],
                    time_feat=self.time_encoder(torch.zeros(num_nodes, device=device)),
                    nbr_node_feat=z[j - 1][i + 1].reshape(num_nodes, num_nbr, -1),
                    edge_feat=batch.nbr_feats[i],
                    valid_nbr_mask=batch.nbr_nids[i] != PADDED_NODE_ID,
                    nbr_time_feat=self.time_encoder(
                        batch.times[i][:, None] - batch.nbr_times[i]
                    ),
                )
                z[j][i] = self.merge_layers[j - 1](out, z[0][i])

        return z[self.num_layers][0]


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

        z = encoder(batch, static_node_x)
        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

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
        z = encoder(batch, static_node_x)
        id_map = {nid.item(): i for i, nid in enumerate(batch.nids[0])}
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            src_idx = torch.tensor([id_map[n.item()] for n in src_ids], device=z.device)
            dst_idx = torch.tensor([id_map[n.item()] for n in dst_ids], device=z.device)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            y_pred = decoder(z_src, z_dst).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

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

if args.sampling == 'uniform':
    nbr_hook = NeighborSamplerHook(
        num_nbrs=args.n_nbrs,
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['edge_event_time', 'edge_event_time', 'neg_time'],
    )
elif args.sampling == 'recency':
    nbr_hook = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=full_data.num_nodes,
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['edge_event_time', 'edge_event_time', 'neg_time'],
    )
else:
    raise ValueError(f'Unknown sampling type: {args.sampling}')


hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(nbr_hook)

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
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, encoder, decoder, opt)

    with hm.activate(val_key):
        val_mrr = eval(val_loader, encoder, decoder, evaluator)
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate(test_key):
    test_mrr = eval(test_loader, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

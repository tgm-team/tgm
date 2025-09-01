import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
)
from tgm.loader import DGDataLoader
from tgm.nn import TemporalAttention, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGAT Example',
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


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).sigmoid().view(-1)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        z = encoder(batch, static_node_feats)
        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    metrics: Metric,
) -> dict:
    encoder.eval()
    decoder.eval()
    for batch in tqdm(loader):
        z = encoder(batch, static_node_feats)
        z_src, z_dst, z_neg = torch.chunk(z, 3)
        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        y_pred = torch.cat([pos_out, neg_out], dim=0).float()
        y_true = (
            torch.cat(
                [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
            )
            .long()
            .to(y_pred.device)
        )
        indexes = torch.zeros(y_pred.size(0), dtype=torch.long, device=y_pred.device)
        metrics(y_pred, y_true, indexes=indexes)
    return metrics.compute()


args = parser.parse_args()
seed_everything(args.seed)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.zeros((test_dg.num_nodes, 1), device=args.device)

# Neighbor Sampler is shared across loaders
if args.sampling == 'uniform':
    nbr_hook = NeighborSamplerHook(num_nbrs=args.num_nbrs)
elif args.sampling == 'recency':
    nbr_hook = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=test_dg.num_nodes,  # Assuming node ids at test set > train/val set
        edge_feats_dim=test_dg.edge_feats_dim,
    )
else:
    raise ValueError(f'Unknown sampling type: {args.sampling}')

_, train_dst, _ = train_dg.edges
_, val_dst, _ = val_dg.edges
_, test_dst, _ = test_dg.edges
train_neg_hook = NegativeEdgeSamplerHook(
    low=int(train_dst.min()), high=int(train_dst.max())
)
val_neg_hook = NegativeEdgeSamplerHook(low=int(val_dst.min()), high=int(val_dst.max()))
test_neg_hook = NegativeEdgeSamplerHook(
    low=int(test_dst.min()), high=int(test_dst.max())
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register('train', train_neg_hook)
hm.register('val', val_neg_hook)
hm.register('test', test_neg_hook)
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

encoder = TGAT(
    node_dim=static_node_feats.shape[1],
    edge_dim=train_dg.edge_feats_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
).to(args.device)
decoder = LinkPredictor(dim=args.embed_dim).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        start_time = time.perf_counter()
        loss = train(train_loader, static_node_feats, encoder, decoder, opt)
        end_time = time.perf_counter()
        latency = end_time - start_time

    with hm.activate('test'):
        val_results = eval(val_loader, static_node_feats, encoder, decoder, val_metrics)
        val_metrics.reset()
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
            + ' '.join(f'{k}={v.item():.4f}' for k, v in val_results.items())
        )

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_results = eval(test_loader, static_node_feats, encoder, decoder, test_metrics)
    print(' '.join(f'{k}={v.item():.4f}' for k, v in test_results.items()))

import argparse
import time
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm.graph import DGBatch, DGraph
from tgm.hooks import DGHook, NegativeEdgeSamplerHook, RecencyNeighborHook
from tgm.loader import DGDataLoader
from tgm.nn import DyGFormer, Time2Vec
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='DyGFormers Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
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
        nbr_nids = batch.nbr_nids[0]
        nbr_times = batch.nbr_times[0]
        nbr_feats = batch.nbr_feats[0]
        edge_idx_pos = torch.stack((src, dst), dim=0)
        edge_idx_neg = torch.stack((src, neg), dim=0)
        batch_size = src.shape[0]

        # positive edge
        z_src_pos, z_dst_pos = self.encoder(
            STATIC_NODE_FEAT,
            edge_idx_pos,
            batch.time,
            nbr_nids[: batch_size * 2],
            nbr_times[: batch_size * 2],
            nbr_feats[: batch_size * 2],
        )
        pos_out = self.decoder(z_src_pos, z_dst_pos)

        # negative edge
        z_src_neg, z_dst_neg = self.encoder(
            STATIC_NODE_FEAT,
            edge_idx_neg,
            batch.time,
            torch.cat(
                [nbr_nids[:batch_size], nbr_nids[-batch_size:]], dim=0
            ),  # Get neighbour inf of src and neg nodes
            torch.cat([nbr_times[:batch_size], nbr_times[-batch_size:]], dim=0),
            torch.cat([nbr_feats[:batch_size], nbr_feats[-batch_size:]], dim=0),
        )
        neg_out = self.decoder(z_src_neg, z_dst_neg)

        return pos_out, neg_out


def _init_hooks(dg: DGraph) -> List[DGHook]:
    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.max_sequence_length - 1],  # Keep 1 slot for seed node itself
        num_nodes=dg.num_nodes,
        edge_feats_dim=dg.edge_feats_dim,
    )
    _, dst, _ = dg.edges
    min_dst, max_dst = int(dst.min()), int(dst.max())
    neg_hook = NegativeEdgeSamplerHook(low=min_dst, high=max_dst)

    return [neg_hook, nbr_hook]


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    metrics: Metric,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch)

        loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))

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

        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss, metrics.compute()


@torch.no_grad()
def eval(
    evaluator: Evaluator,
    loader: DGDataLoader,
    model: nn.Module,
    metrics: Metric,
) -> dict:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out = model(batch)

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


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)

    dgraph = DGraph(args.dataset)
    train_dg = DGraph(
        args.dataset,
        split='train',
        device=args.device,
    )
    val_dg = DGraph(
        args.dataset,
        split='val',
        device=args.device,
    )
    test_dg = DGraph(
        args.dataset,
        split='test',
        device=args.device,
    )

    num_nodes = dgraph.num_nodes
    edge_feats_dim = dgraph.edge_feats_dim
    label_dim = train_dg.dynamic_node_feats_dim

    train_loader = DGDataLoader(
        train_dg,
        batch_size=args.bsize,
        hook=_init_hooks(dg=train_dg),
    )
    val_loader = DGDataLoader(
        val_dg,
        batch_size=args.bsize,
        hook=_init_hooks(dg=val_dg),
    )
    test_loader = DGDataLoader(
        test_dg,
        batch_size=args.bsize,
        hook=_init_hooks(dg=test_dg),
    )

    # TODO: add static node features to DGraph
    STATIC_NODE_FEAT = torch.randn((num_nodes, args.node_dim), device=args.device)

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
    evaluator = Evaluator(name=args.dataset)
    metrics = [BinaryAveragePrecision(), BinaryAUROC()]
    train_metrics = MetricCollection(metrics, prefix='Train')
    val_metrics = MetricCollection(metrics, prefix='Validation')
    test_metrics = MetricCollection(metrics, prefix='Test')

    for epoch in range(1, args.epochs + 1):
        # TODO: Need a clean way to clear nbr state across epochs
        train_loader = DGDataLoader(
            train_dg,
            batch_size=args.bsize,
            hook=_init_hooks(dg=train_dg),
        )
        val_loader = DGDataLoader(
            val_dg,
            batch_size=args.bsize,
            hook=_init_hooks(dg=val_dg),
        )

        start_time = time.perf_counter()
        loss, train_results = train(train_loader, model, opt, train_metrics)
        end_time = time.perf_counter()
        latency = end_time - start_time

        val_results = eval(evaluator, val_loader, model, val_metrics)
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
            + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
            + ' '.join(f'{k}={v:.4f}' for k, v in train_results.items())
        )

    test_results = eval(evaluator, test_loader, model, test_metrics)
    print('Test:', ' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

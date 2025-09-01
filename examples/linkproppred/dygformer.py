import argparse
import time
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm.graph import DGBatch, DGData, DGraph
from tgm.hooks import HookManager, NegativeEdgeSamplerHook, RecencyNeighborHook
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

parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
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

    full_data = DGData.from_tgb(args.dataset)
    full_graph = DGraph(full_data)
    num_nodes = full_graph.num_nodes
    edge_feats_dim = full_graph.edge_feats_dim

    train_data, val_data, test_data = full_data.split()

    train_data = train_data.discretize(args.time_gran)
    val_data = val_data.discretize(args.time_gran)
    test_data = test_data.discretize(args.time_gran)

    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

    train_neg_hook = NegativeEdgeSamplerHook(
        low=int(train_dg.edges[1].min()), high=int(train_dg.edges[1].max())
    )
    val_neg_hook = NegativeEdgeSamplerHook(
        low=int(val_dg.edges[1].min()), high=int(val_dg.edges[1].max())
    )
    test_neg_hook = NegativeEdgeSamplerHook(
        low=int(test_dg.edges[1].min()), high=int(test_dg.edges[1].max())
    )

    nbr_hook = RecencyNeighborHook(
        num_nbrs=[args.max_sequence_length - 1],  # Keep 1 slot for seed node itself
        num_nodes=num_nodes,
        edge_feats_dim=edge_feats_dim,
    )

    hm = HookManager(keys=['train', 'val', 'test'])
    hm.register_shared(nbr_hook)
    hm.register('train', train_neg_hook)
    hm.register('val', val_neg_hook)
    hm.register('test', test_neg_hook)

    train_loader = DGDataLoader(train_dg, batch_size=args.bsize, hook_manager=hm)
    val_loader = DGDataLoader(val_dg, batch_size=args.bsize, hook_manager=hm)
    test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

    if train_dg.static_node_feats is not None:
        STATIC_NODE_FEAT = train_dg.static_node_feats
    else:
        STATIC_NODE_FEAT = torch.randn(
            (test_dg.num_nodes, args.node_dim), device=args.device
        )

    model = DyGFormer_LinkPrediction(
        node_feat_dim=STATIC_NODE_FEAT.shape[1],
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
        with hm.activate('train'):
            start_time = time.perf_counter()
            loss, train_results = train(train_loader, model, opt, train_metrics)
            end_time = time.perf_counter()
            latency = end_time - start_time
        with hm.activate('val'):
            val_results = eval(evaluator, val_loader, model, val_metrics)
            val_metrics.reset()
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
            + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
            + ' '
            + ' '.join(f'{k}={v:.4f}' for k, v in train_results.items())
        )
        if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
            hm.reset_state()

    with hm.activate('test'):
        test_results = eval(evaluator, test_loader, model, test_metrics)
        print('Test:', ' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

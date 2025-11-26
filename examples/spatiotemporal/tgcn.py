import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric_temporal.dataset
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader, TemporalRatioSplit
from tgm.nn import TGCN
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGCN SpatioTemporal Regression Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument(
    '--dataset',
    type=str,
    default='metr_la',
    help='Dataset name',
    choices=['metr_la', 'pems_bay'],
)
parser.add_argument('--bsize', type=int, default=64, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--embed-dim', type=int, default=32, help='embedding dimension')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.cached_edge_index = None

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.tensor,
        h: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = self._get_cached_edge_index(batch)
        edge_weight = batch.edge_feats

        B, N, _ = node_feat.shape
        node_feat = node_feat.reshape(B * N, -1)
        h_0 = self.recurrent(node_feat, edge_index, edge_weight, h)
        z = F.relu(h_0)
        z = self.linear(z).reshape(B, N, -1)
        return z, h_0

    def _get_cached_edge_index(self, batch: DGBatch) -> torch.Tensor:
        if self.cached_edge_index is None:
            self.cached_edge_index = torch.stack([batch.src, batch.dst], dim=0)
        return self.cached_edge_index


def masked_mae_loss(y_pred, y_true):
    mask = y_true != 0
    loss = torch.abs(y_pred[mask] - y_true[mask])
    return loss.mean() if loss.numel() else torch.tensor(0.0, device=y_pred.device)


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    encoder: nn.Module,
    opt: torch.optim.Optimizer,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    encoder.train()
    total_loss, h_0 = 0, None

    for batch in tqdm(loader):
        opt.zero_grad()

        node_feats, y_true = (
            batch.dynamic_node_feats[..., :-1, :],
            batch.dynamic_node_feats[..., -1, :],
        )

        out_seq = []
        for t in range(node_feats.shape[-1]):
            out_t, h_0 = encoder(batch, node_feats[:, :, :, t], h_0)
            out_seq.append(out_t)
        y_pred = torch.cat(out_seq, dim=-1)

        loss = masked_mae_loss((y_pred * std) + mean, (y_true * std) + mean)
        loss.backward()
        opt.step()
        total_loss += float(loss)

        h_0 = h_0.detach()

    return total_loss, h_0


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    h_0: torch.Tensor,
    encoder: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    encoder.eval()
    maes = []

    for batch in tqdm(loader):
        node_feats, y_true = (
            batch.dynamic_node_feats[..., :-1, :],
            batch.dynamic_node_feats[..., -1, :],
        )

        out_seq = []
        for t in range(node_feats.shape[-1]):
            out_t, h_0 = encoder(batch, node_feats[:, :, :, t], h_0)
            out_seq.append(out_t)
        y_pred = torch.cat(out_seq, dim=-1)
        mae = masked_mae_loss((y_pred * std) + mean, (y_true * std) + mean)
        maes.append(float(mae))

    return np.mean(maes)


seed_everything(args.seed)

pyg_temporal_loaders = {
    'metr_la': lambda index: torch_geometric_temporal.dataset.METRLADatasetLoader(
        index=index
    ),
    'pems_bay': lambda index: torch_geometric_temporal.dataset.PemsBayDatasetLoader(
        index=index
    ),
}

# Load dataset
if args.dataset in pyg_temporal_loaders:
    # TODO: Hide behind IO if we want to natively support this
    signal = pyg_temporal_loaders[args.dataset](index=False).get_dataset()
    _, _, _, _, _, means, stds = pyg_temporal_loaders[args.dataset](
        index=True
    ).get_index_dataset()
    means, stds = means[0].item(), stds[0].item()  # Predicting 1d signal
else:
    raise ValueError(f'Unknown PyG-Temporal dataset: {args.dataset}')

data = DGData.from_pyg_temporal(signal)
split = TemporalRatioSplit(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
train_data, val_data, test_data = split.apply(data)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(
    train_dg, batch_unit=train_dg.time_delta.unit, batch_size=args.bsize, drop_last=True
)
val_loader = DGDataLoader(
    val_dg, batch_unit=val_dg.time_delta.unit, batch_size=args.bsize, drop_last=True
)
test_loader = DGDataLoader(
    test_dg, batch_unit=train_dg.time_delta.unit, batch_size=args.bsize, drop_last=True
)

# Dynamic node features are concatenating of node signals and 1-dim target
node_dim = train_dg.dynamic_node_feats_dim[1] - 1
encoder = RecurrentGCN(node_dim=node_dim, embed_dim=args.embed_dim).to(args.device)
opt = torch.optim.Adam(encoder.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    loss, h_0 = train(train_loader, encoder, opt, means, stds)
    val_mae = eval(val_loader, h_0, encoder, means, stds)
    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation MAE', val_mae, epoch=epoch)

mae = eval(test_loader, h_0, encoder, means, stds)
log_metric(f'Test MAE', mae, epoch=epoch)

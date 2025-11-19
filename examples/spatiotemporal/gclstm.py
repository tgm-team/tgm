import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric_temporal.dataset
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader, TemporalRatioSplit
from tgm.nn import GCLSTM
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GC-LSTM SpatioTemporal Regression Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument(
    '--dataset',
    type=str,
    default='chickenpox',
    help='Dataset name',
    choices=['chickenpox', 'encovid', 'metr_la', 'pems_bay'],
)
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--embed-dim', type=int, default=32, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
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
        self.linear = nn.Linear(embed_dim, 1)

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
    loader: DGDataLoader, encoder: nn.Module, opt: torch.optim.Optimizer
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    encoder.train()
    opt.zero_grad()
    loss, h_0, c_0 = 0, None, None

    for batch in tqdm(loader):
        node_feats, y = (
            batch.dynamic_node_feats[:, :-1],
            batch.dynamic_node_feats[:, -1],
        )
        y_pred, h_0, c_0 = encoder(batch, node_feats, h_0, c_0)
        loss += F.mse_loss(y_pred, y)

    loss /= len(loader)
    loss.backward()
    opt.step()
    return float(loss), h_0, c_0


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader, h_0: torch.Tensor, c_0: torch.Tensor, encoder: nn.Module
) -> float:
    encoder.eval()
    loss = 0

    for batch in tqdm(loader):
        node_feats, y = (
            batch.dynamic_node_feats[:, :-1],
            batch.dynamic_node_feats[:, -1],
        )
        y_pred, h_0, c_0 = encoder(batch, node_feats, h_0, c_0)
        loss += F.mse_loss(y_pred, y)

    loss /= len(loader)
    return float(loss)


seed_everything(args.seed)

pyg_temporal_loaders = {
    'chickenpox': lambda: torch_geometric_temporal.dataset.ChickenpoxDatasetLoader(),
    'encovid': lambda: torch_geometric_temporal.dataset.EnglandCovidDatasetLoader(),
    'metr_la': lambda: torch_geometric_temporal.dataset.METRLADatasetLoader(),
    'pems_bay': lambda: torch_geometric_temporal.dataset.PemsBayDatasetLoader(),
}

# Load dataset
if args.dataset in pyg_temporal_loaders:
    signal = pyg_temporal_loaders[args.dataset]().get_dataset()
else:
    raise ValueError(f'Unknown PyG-Temporal dataset: {args.dataset}')

data = DGData.from_pyg_temporal(signal)
split = TemporalRatioSplit(train_ratio=0.2, val_ratio=0.0, test_ratio=0.8)
train_data, test_data = split.apply(data)

train_dg = DGraph(train_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(train_dg, batch_unit=train_dg.time_delta.unit)
test_loader = DGDataLoader(test_dg, batch_unit=train_dg.time_delta.unit)

# Dynamic node features are concatenating of node signals and 1-dim target
node_dim = train_dg.dynamic_node_feats_dim - 1
encoder = RecurrentGCN(node_dim=node_dim, embed_dim=args.embed_dim).to(args.device)
opt = torch.optim.Adam(encoder.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    loss, h_0, c_0 = train(train_loader, encoder, opt)
    log_metric('Loss', loss, epoch=epoch)

mse = eval(test_loader, h_0, c_0, encoder)
log_metric(f'Test MSE', mse, epoch=epoch)

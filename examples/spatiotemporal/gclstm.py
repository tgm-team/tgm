import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric_temporal.dataset

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader, TemporalRatioSplit
from tgm.nn import GCLSTM
from tgm.util.logging import enable_logging
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GC-LSTM SpatioTemporal Regression Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='chickenpox', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int, K: int = 1) -> None:
        super().__init__()
        self.recurrent = GCLSTM(in_channels=node_dim, out_channels=embed_dim, K=K)
        self.linear = nn.Linear(embed_dim, embed_dim)

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


seed_everything(args.seed)

pyg_temporal_loaders = {
    'chickenpox': lambda: torch_geometric_temporal.dataset.ChickenpoxDatasetLoader(),
    'advection-diffusion': lambda: torch_geometric_temporal.dataset.AdvectionDiffusionDatasetLoader(),
    'encovid': lambda: torch_geometric_temporal.dataset.EnglandCovidDatasetLoader(),
    'metr_la': lambda: torch_geometric_temporal.dataset.METRLADatasetLoader(),
    'montevideo_bus': lambda: torch_geometric_temporal.dataset.MontevideoBusDatasetLoader(),
    'mtm': lambda: torch_geometric_temporal.dataset.MTMDatasetLoader(),
    'pedalme': lambda: torch_geometric_temporal.dataset.PedalMeDatasetLoader(),
    'pems': lambda: torch_geometric_temporal.dataset.PemsDatasetLoader(),
    'pemsAllLA': lambda: torch_geometric_temporal.dataset.PemsAllLADatasetLoader(),
    'pems_bay': lambda: torch_geometric_temporal.dataset.PemsBayDatasetLoader(),
    'si_diffusion': lambda: torch_geometric_temporal.dataset.SIDiffusionDatasetLoader(),
    'twitter_tennis': lambda: torch_geometric_temporal.dataset.TwitterTennisDatasetLoader(),
    'wave_equation': lambda: torch_geometric_temporal.dataset.WaveEquationDatasetLoader(),
    'wikimath': lambda: torch_geometric_temporal.dataset.WikiMathsDatasetLoader(),
    'windmilllarge': lambda: torch_geometric_temporal.dataset.WindmillOutputLargeDatasetLoader(),
    'windmillmedium': lambda: torch_geometric_temporal.dataset.WindmillOutputMediumDatasetLoader(),
    'windmillsmall': lambda: torch_geometric_temporal.dataset.WindmillOutputSmallDatasetLoader(),
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

train_loader = DGDataLoader(train_dg)
test_loader = DGDataLoader(test_dg)

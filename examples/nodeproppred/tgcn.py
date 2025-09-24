import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.graph import DGBatch, DGData, DGraph
from tgm.loader import DGDataLoader
from tgm.nn.recurrent import TGCN
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGCN NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=256, help='node feat dimension if not provided'
)
parser.add_argument(
    '--snapshot-time-gran',
    type=str,
    default='Y',
    help='time granularity to operate on for snapshots',
)
parser.add_argument(
    '--capture-gpu', action=argparse.BooleanOptionalAction, help='record peak gpu usage'
)
parser.add_argument(
    '--capture-cprofile', action=argparse.BooleanOptionalAction, help='record cprofiler'
)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.tensor,
        h: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None  # type: ignore

        h_0 = self.recurrent(node_feat, edge_index, edge_weight, h)
        z = F.relu(h_0)
        z = self.linear(z)
        return z, h_0


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, z_node: torch.Tensor) -> torch.Tensor:
        h = self.fc1(z_node)
        h = h.relu()
        return self.fc2(h)


def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> Tuple[float, torch.Tensor]:
    encoder.train()
    decoder.train()
    total_loss = 0
    h_0 = None

    for batch in tqdm(loader):
        opt.zero_grad()
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z, h_0 = encoder(batch, static_node_feats)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        loss = F.cross_entropy(y_pred, y_true)
        loss.backward()
        opt.step()
        total_loss += float(loss)

        h_0 = h_0.detach()

    return total_loss, h_0


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    h_0: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        y_true = batch.dynamic_node_feats
        if y_true is None:
            continue

        z, h_0 = encoder(batch, static_node_feats, h_0)
        z_node = z[batch.node_ids]
        y_pred = decoder(z_node)

        input_dict = {
            'y_true': y_true,
            'y_pred': y_pred,
            'eval_metric': [METRIC_TGB_NODEPROPPRED],
        }
        perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

    return float(np.mean(perf_list))


args = parser.parse_args()
seed_everything(args.seed)
from pathlib import Path

from experiments import save_experiment_results_and_exit, setup_experiment

results = setup_experiment(args, Path(__file__))

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

train_loader = DGDataLoader(train_dg, batch_unit=args.snapshot_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.snapshot_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.snapshot_time_gran)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

evaluator = Evaluator(name=args.dataset)
num_classes = train_dg.dynamic_node_feats_dim

encoder = RecurrentGCN(
    node_dim=static_node_feats.shape[1], embed_dim=args.embed_dim
).to(args.device)
decoder = NodePredictor(in_dim=args.embed_dim, out_dim=num_classes).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss, h_0 = train(train_loader, static_node_feats, encoder, decoder, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    start_time = time.perf_counter()
    val_ndcg = eval(val_loader, static_node_feats, h_0, encoder, decoder, evaluator)
    end_time = time.perf_counter()
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} Validation {METRIC_TGB_NODEPROPPRED}={val_ndcg:.4f}'
    )
    results[f'val_{METRIC_TGB_NODEPROPPRED}'] = val_ndcg
    results['train_latency_s'] = latency
    results['val_latency_s'] = end_time - start_time
    save_experiment_results_and_exit(results)

test_ndcg = eval(test_loader, static_node_feats, h_0, encoder, decoder, evaluator)
print(f'Test {METRIC_TGB_NODEPROPPRED}={test_ndcg:.4f}')

r"""python -u tgcn.py --dataset tgbn-trade --time-gran Y --batch-time-gran Y
python -u tgcn.py --dataset tgbn-genre --time-gran s --batch-time-gran D\
example commands to run this script.
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm.graph import DGBatch, DGData, DGraph
from tgm.loader import DGDataLoader
from tgm.nn.recurrent import TGCN
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGCN Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--time-gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--batch-time-gran',
    type=str,
    default='D',
    help='time granularity to operate on for snapshots',
)


class TGCN_Model(nn.Module):
    def __init__(self, node_dim: int, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = RecurrentGCN(node_dim=node_dim, embed_dim=embed_dim)
        self.decoder = NodePredictor(in_dim=embed_dim, out_dim=num_classes)

    def forward(
        self,
        batch: DGBatch,
        node_feat: torch.Tensor,
        h_0: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None  # type: ignore
        z, h_0 = self.encoder(node_feat, edge_index, edge_weight, h_0)
        z_node = z[batch.global_to_local(batch.node_ids)]  # type: ignore
        pred = self.decoder(z_node)
        return pred, h_0


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.recurrent = TGCN(in_channels=node_dim, out_channels=embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        h: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, ...]:
        h_0 = self.recurrent(x, edge_index, edge_weight, h)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_node = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        return h


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    node_feat: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.train()
    total_loss = 0
    h_0 = None
    criterion = torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader):
        opt.zero_grad()
        label = batch.dynamic_node_feats
        if label is None:
            continue
        pred, h_0 = model(batch, node_feat, h_0)
        loss = criterion(pred, label)
        loss.backward()
        opt.step()
        total_loss += float(loss)
        h_0 = h_0.detach()
    return total_loss, h_0


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    model: nn.Module,
    node_feat: torch.Tensor,
    h_0: torch.Tensor | None = None,
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    eval_metric = 'ndcg'
    total_score = 0
    for batch in tqdm(loader):
        label = batch.dynamic_node_feats
        if label is None:
            continue
        pred, h_0 = model(batch, node_feat, h_0)
        np_pred = pred.cpu().detach().numpy()
        np_true = label.cpu().detach().numpy()
        input_dict = {
            'y_true': np_true,
            'y_pred': np_pred,
            'eval_metric': [eval_metric],
        }
        result_dict = evaluator.eval(input_dict)
        score = result_dict[eval_metric]
        total_score += score
    metric_dict = {}
    metric_dict[eval_metric] = float(total_score) / len(loader)
    return metric_dict, h_0


args = parser.parse_args()
seed_everything(args.seed)

train_data = DGData.from_tgb(args.dataset, split='train').discretize(args.time_gran)
train_dg = DGraph(train_data, train_data.time_delta, device=args.device)

val_data = DGData.from_tgb(args.dataset, split='val').discretize(args.time_gran)
val_dg = DGraph(val_data, val_data.time_delta, device=args.device)

test_data = DGData.from_tgb(args.dataset, split='test').discretize(args.time_gran)
test_dg = DGraph(test_data, test_data.time_delta, device=args.device)

full_data = DGData.from_tgb(args.dataset)
num_nodes = DGraph(full_data).num_nodes
label_dim = train_dg.dynamic_node_feats_dim
evaluator = Evaluator(name=args.dataset)

train_loader = DGDataLoader(train_dg, batch_unit=args.batch_time_gran)
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran)
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran)

# TODO: add static node features to DGraph
args.node_dim = args.embed_dim
static_node_feats = torch.randn((test_dg.num_nodes, args.node_dim), device=args.device)
model = TGCN_Model(
    node_dim=args.node_dim, embed_dim=args.embed_dim, num_classes=label_dim
).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss, h_0 = train(train_loader, model, opt, static_node_feats)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results, h_0 = eval(val_loader, model, static_node_feats, h_0)
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results, h_0 = eval(test_loader, model, static_node_feats, h_0)
print('Test:', ' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

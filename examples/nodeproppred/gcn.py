r"""python -u gcn.py --dataset tgbn-trade --time-gran Y --batch-time-gran Y
python -u gcn.py --dataset tgbn-genre --time-gran s --batch-time-gran D
example commands to run this script.
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from tgm import DGBatch, DGData, DGraph
from tgm.hooks import HookManager
from tgm.loader import DGDataLoader
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GCN Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-genre', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-layers', type=int, default=2, help='number of GCN layers')
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
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


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = GCNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_channels=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = NodePredictor(in_dim=embed_dim, out_dim=num_classes)

    def forward(
        self, batch: DGBatch, node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.stack([batch.src, batch.dst], dim=0).to(node_feat.device)
        z = self.encoder(node_feat, edge_index)
        z_node = z[batch.global_to_local(batch.node_ids)]  # type: ignore
        pred = self.decoder(z_node)
        return pred


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, embed_dim, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(embed_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(embed_dim, embed_dim, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(embed_dim))
        self.convs.append(GCNConv(embed_dim, out_channels, cached=True))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
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
    static_node_feats: torch.Tensor,
    model: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader):
        opt.zero_grad()
        label = batch.dynamic_node_feats
        if label is None:
            continue
        pred = model(batch, static_node_feats)
        loss = criterion(pred, label)
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    model: nn.Module,
    evaluator: Evaluator,
) -> dict:
    model.eval()
    eval_metric = 'ndcg'
    total_score = 0
    for batch in tqdm(loader):
        label = batch.dynamic_node_feats
        if label is None:
            continue
        pred = model(batch, static_node_feats)

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
    return metric_dict


args = parser.parse_args()
seed_everything(args.seed)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()

train_data = train_data.discretize(args.time_gran)
val_data = val_data.discretize(args.time_gran)
test_data = test_data.discretize(args.time_gran)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

num_nodes = DGraph(train_data).num_nodes
label_dim = train_dg.dynamic_node_feats_dim
evaluator = Evaluator(name=args.dataset)

hm = HookManager()
train_loader = DGDataLoader(train_dg, batch_unit=args.batch_time_gran, hook_manager=hm)
val_loader = DGDataLoader(val_dg, batch_unit=args.batch_time_gran, hook_manager=hm)
test_loader = DGDataLoader(test_dg, batch_unit=args.batch_time_gran, hook_manager=hm)

if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

model = GCN(
    in_channels=static_node_feats.shape[1],
    embed_dim=args.embed_dim,
    num_layers=args.n_layers,
    dropout=float(args.dropout),
    num_classes=label_dim,
).to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    start_time = time.perf_counter()
    loss = train(train_loader, static_node_feats, model, opt)
    end_time = time.perf_counter()
    latency = end_time - start_time

    val_results = eval(val_loader, static_node_feats, model, evaluator)
    print(
        f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
        + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
    )

test_results = eval(test_loader, static_node_feats, model, evaluator)
print(' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

import argparse
import time
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from tgm.graph import DGData, DGraph
from tgm.hooks import HookManager, NegativeEdgeSamplerHook, RecencyNeighborHook
from tgm.loader import DGBatch, DGDataLoader
from tgm.nn import RandomProjectionModule, Time2Vec, TPNet
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TPNet Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument(
    '--time_gran',
    type=str,
    default='s',
    help='raw time granularity for dataset',
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')

parser.add_argument(
    '--num_neighbors',
    type=int,
    default=32,
    help='number of recency temporal neighbors of each node',
)
parser.add_argument(
    '--rp_num_layers',
    type=int,
    default=2,
    help='the number of layer of random projection module',
)
parser.add_argument(
    '--rp_time_decay_weight',
    type=float,
    default=0.000001,
    help='the first weight of the time decay',
)
parser.add_argument(
    '--enforce_dim',
    type=int,
    default=None,
    help='enforced dimension of random projections',
)
parser.add_argument(
    '--rp_dim_factor',
    type=int,
    default=10,
    help='the dim factor of random feature w.r.t. the node num',
)
parser.add_argument('--node_dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--time_dim', type=int, default=100, help='time encoding dimension')
parser.add_argument(
    '--embed_dim', type=int, default=172, help='node representation dimension'
)
parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')


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


class TPNet_LinkPrediction(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        num_neighbors: int,
        random_projections: RandomProjectionModule | None = None,
        device: str = 'cpu',
        time_encoder: Callable[..., nn.Module] = Time2Vec,
    ) -> None:
        super().__init__()
        self.encoder = TPNet(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            time_feat_dim=time_feat_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_layers=num_layers,
            num_neighbors=num_neighbors,
            random_projections=random_projections,
            device=device,
            time_encoder=time_encoder,
        )
        self.rp_module = random_projection_module.to(device)
        self.decoder = LinkPredictor(output_dim).to(device)

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
        self.rp_module.update(src, dst, time=batch.time)

        return pos_out, neg_out


def train(
    loader: DGDataLoader,
    model: TPNet_LinkPrediction,
    opt: torch.optim.Optimizer,
    metrics: Metric,
) -> Tuple[float, dict]:
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

    data = DGData.from_tgb(args.dataset)
    dgraph = DGraph(data)

    num_nodes = dgraph.num_nodes
    edge_feats_dim = dgraph.edge_feats_dim

    if dgraph.static_node_feats is not None:
        STATIC_NODE_FEAT = dgraph.static_node_feats
    else:
        STATIC_NODE_FEAT = torch.randn((num_nodes, args.node_dim), device=args.device)

    node_dim = STATIC_NODE_FEAT.shape[1]

    train_data, val_data, test_data = data.split()
    train_data.discretize(args.time_gran)
    val_data.discretize(args.time_gran)
    test_data.discretize(args.time_gran)

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
        num_nbrs=[args.num_neighbors],
        num_nodes=num_nodes,
        edge_feats_dim=edge_feats_dim,
    )

    hm = HookManager(keys=['train', 'val', 'test'])
    hm.register('train', train_neg_hook)
    hm.register('val', val_neg_hook)
    hm.register('test', test_neg_hook)
    hm.register_shared(nbr_hook)

    train_loader = DGDataLoader(train_dg, batch_size=args.bsize, hook_manager=hm)
    val_loader = DGDataLoader(val_dg, batch_size=args.bsize, hook_manager=hm)
    test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

    random_projection_module = RandomProjectionModule(
        num_nodes=num_nodes,
        num_layer=args.rp_num_layers,
        time_decay_weight=args.rp_time_decay_weight,
        beginning_time=dgraph.start_time,
        enforce_dim=args.enforce_dim,
        num_edges=train_dg.num_edges,
        dim_factor=args.rp_dim_factor,
        device=args.device,
    )

    model = TPNet_LinkPrediction(
        node_feat_dim=STATIC_NODE_FEAT.shape[1],
        edge_feat_dim=edge_feats_dim,
        time_feat_dim=args.time_dim,
        output_dim=args.embed_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        random_projections=random_projection_module,
        device=args.device,
        time_encoder=Time2Vec,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
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
            val_results = eval(val_loader, model, val_metrics)
            val_metrics.reset()
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} '
            + ' '.join(f'{k}={v:.4f}' for k, v in val_results.items())
            + ' '
            + ' '.join(f'{k}={v:.4f}' for k, v in train_results.items())
        )
        if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
            hm.reset_state()
            # reinitialize the random projections of temporal walk matrices at the start of each epoch
            model.rp_module.reset_random_projections()

    with hm.activate('test'):
        test_results = eval(test_loader, model, test_metrics)
        print('Test:', ' '.join(f'{k}={v:.4f}' for k, v in test_results.items()))

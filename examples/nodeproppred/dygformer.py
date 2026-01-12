import argparse
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import METRIC_TGB_NODEPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import (
    DeduplicationHook,
    EdgeEventsSeenNodesTrackHook,
    HookManager,
    RecencyNeighborHook,
)
from tgm.nn import DyGFormer, NodePredictor, Time2Vec
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='DyGFormers NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
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
    default='Y',
    help='raw time granularity for dataset',
)
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class DyGFormer_NodePrediction(nn.Module):
    def __init__(
        self,
        num_nodes: int,
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
        self.z = torch.zeros(
            (num_nodes, output_dim), dtype=torch.float32, device=device
        )  # Maintain up-to-date node embeddings

    def _update_latest_node_embedding(
        self, batch: DGBatch, z_src: torch.Tensor, z_dst: torch.Tensor
    ):
        nodes = torch.cat([batch.src, batch.dst])
        z_all = torch.cat([z_src, z_dst])
        timestamp = torch.cat([batch.edge_time, batch.edge_time])

        chronological_order = torch.argsort(timestamp)
        nodes = nodes[chronological_order]
        z_all = z_all[chronological_order]

        unique_nodes, idx = torch.unique(nodes, return_inverse=True)
        positions = torch.arange(len(nodes), device=nodes.device)
        latest_indices = torch.zeros(
            len(unique_nodes), dtype=torch.long, device=nodes.device
        )
        latest_indices.scatter_reduce_(
            0, idx, positions, reduce='amax', include_self=False
        )

        latest_embeddings = z_all[latest_indices]

        self.z[unique_nodes] = latest_embeddings.detach()

    def forward(
        self, batch: DGBatch, static_node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = batch.src
        dst = batch.dst
        nbr_nids = batch.nbr_nids[0]
        nbr_times = batch.nbr_times[0]
        nbr_feats = batch.nbr_feats[0]
        src_nbr_idx = batch.seed_node_nbr_mask['src']
        dst_nbr_idx = batch.seed_node_nbr_mask['dst']
        edge_idx = torch.stack((src, dst), dim=0)

        src_dst_nbr_idx = torch.cat([src_nbr_idx, dst_nbr_idx])
        z_src, z_dst = self.encoder(
            static_node_feat,
            edge_idx,
            batch.edge_time,
            nbr_nids[src_dst_nbr_idx],
            nbr_times[src_dst_nbr_idx],
            nbr_feats[src_dst_nbr_idx],
        )
        self._update_latest_node_embedding(batch, z_src, z_dst)

        return self.z


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
):
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        opt.zero_grad()

        y_true = batch.node_x
        if len(batch.src) > 0:
            z = encoder(batch, static_node_x)  # [num_nodes, embed_dim]

        if y_true is not None:
            if len(batch.seen_nodes) == 0:
                continue

            z_node = z[batch.seen_nodes]

            y_pred = decoder(z_node)
            loss = F.cross_entropy(y_pred, y_true[batch.batch_nodes_mask])
            loss.backward()
            opt.step()
            total_loss += float(loss)

    return total_loss


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        y_true = batch.node_x

        if batch.src.shape[0] > 0:
            z = encoder(batch, static_node_x)
            if y_true is not None:
                z_node = z[batch.node_event_node_ids]
                y_pred = decoder(z_node)
                input_dict = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'eval_metric': [METRIC_TGB_NODEPROPPRED],
                }
                perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
if full_data.static_node_x is None:
    full_data.static_node_x = torch.randn(
        (full_data.num_nodes, args.node_dim), device=args.device
    )

train_data, val_data, test_data = full_data.split()

train_data = train_data.discretize(args.time_gran)
val_data = val_data.discretize(args.time_gran)
test_data = test_data.discretize(args.time_gran)

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


nbr_hook = RecencyNeighborHook(
    num_nbrs=[args.max_sequence_length - 1],  # Keep 1 slot for seed node itself
    num_nodes=full_data.num_nodes,
    seed_nodes_keys=['src', 'dst'],
    seed_times_keys=['edge_time', 'edge_time'],
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register('train', EdgeEventsSeenNodesTrackHook(full_data.num_nodes))
hm.register_shared(DeduplicationHook())
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, batch_size=args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, batch_size=args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

num_classes = train_dg.node_x_dim

encoder = DyGFormer_NodePrediction(
    num_nodes=full_data.num_nodes,
    node_feat_dim=train_dg.static_node_x_dim,
    edge_feat_dim=train_dg.edge_x_dim,
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

decoder = NodePredictor(
    in_dim=args.embed_dim, out_dim=num_classes, hidden_dim=args.embed_dim
).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        loss = train(train_loader, encoder, decoder, opt)
    with hm.activate('val'):
        val_ndcg = eval(val_loader, encoder, decoder, evaluator)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_ndcg, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_ndcg = eval(test_loader, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_ndcg, epoch=args.epochs)

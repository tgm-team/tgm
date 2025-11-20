import argparse
from typing import Callable, Tuple, Dict
import os

import pandas as pd
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
from tgm.nn import NodePredictor, RandomProjectionModule, Time2Vec, TPNet
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TPNet NodePropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument(
    '--num-neighbors',
    type=int,
    default=32,
    help='number of recency temporal neighbors of each node',
)
parser.add_argument(
    '--rp-num-layers',
    type=int,
    default=2,
    help='the number of layer of random projection module',
)
parser.add_argument(
    '--rp-time-decay-weight',
    type=float,
    default=0.000001,
    help='the first weight of the time decay',
)
parser.add_argument(
    '--enforce-dim',
    type=int,
    default=None,
    help='enforced dimension of random projections',
)
parser.add_argument(
    '--rp-dim-factor',
    type=int,
    default=10,
    help='the dim factor of random feature w.r.t. the node num',
)
parser.add_argument(
    '--use-matrix',
    default=True,
    action=argparse.BooleanOptionalAction,
    help='if no-use-matrix, will not explicitly maintain the temporal walk matrices',
)
parser.add_argument(
    '--concat-src-dst',
    default=True,
    action=argparse.BooleanOptionalAction,
    help='if no-concat-src-dst, Random projection avoids concat src and dst in computation',
)
parser.add_argument('--node-dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument(
    '--embed-dim', type=int, default=172, help='node representation dimension'
)
parser.add_argument('--num-layers', type=int, default=2, help='number of model layers')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

parser.add_argument(
    '--max_sequence_length',
    type=int,
    default=32,
    help='maximal length of the input sequence of each node',
)

parser.add_argument(
    '--time-gran',
    type=str,
    default='Y',
    help='raw time granularity for dataset',
)
parser.add_argument(
    '--experiment_name',
    type=str,
    default='time_granularity_effect',
    help='Name of experiment',
)

MODEL_NAME = 'TPNet'
EXPERIMENTS_ARTIFACT = 'experiments/artifact'

def save_results(experiment_id: str, results: Dict, intermediate_path: str = ''):
    partial_path = f'{EXPERIMENTS_ARTIFACT}/results/{intermediate_path}'
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)

    result_path = f'{partial_path}/{experiment_id}.csv'
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns=results.keys())
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df._append(results, ignore_index=True)
    result_df.to_csv(result_path, index=False)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class TPNet_NodePrediction(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        num_neighbors: int,
        random_projection_module: RandomProjectionModule | None = None,
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
            random_projections=random_projection_module,
            device=device,
            time_encoder=time_encoder,
        )
        self.z = torch.zeros(
            (num_nodes, output_dim), dtype=torch.float32, device=device
        )  # Maintain up-to-date node embeddings
        
        self.rp_module = random_projection_module.to(device)

    def _update_latest_node_embedding(
        self, batch: DGBatch, z_src: torch.Tensor, z_dst: torch.Tensor
    ):
        nodes = torch.cat([batch.src, batch.dst])
        z_all = torch.cat([z_src, z_dst])
        timestamp = torch.cat([batch.time, batch.time])

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
        edge_idx = torch.stack((src, dst), dim=0)
        batch_size = src.shape[0]

        z_src, z_dst = self.encoder(
            static_node_feat,
            edge_idx,
            batch.time,
            nbr_nids[: batch_size * 2],
            nbr_times[: batch_size * 2],
            nbr_feats[: batch_size * 2],
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
    static_node_feat: torch.Tensor,
):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in tqdm(loader):
        opt.zero_grad()

        y_true = batch.dynamic_node_feats
        if len(batch.src) > 0:
            z = encoder(batch, static_node_feat)  # [num_nodes, embed_dim]

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
    static_node_feat: torch.Tensor,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    for batch in tqdm(loader):
        y_true = batch.dynamic_node_feats

        if batch.src.shape[0] > 0:
            z = encoder(batch, static_node_feat)
            if y_true is not None:
                z_node = z[batch.node_ids]
                y_pred = decoder(z_node)
                input_dict = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'eval_metric': [METRIC_TGB_NODEPROPPRED],
                }
                perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_NODEPROPPRED])

    return float(np.mean(perf_list))


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

nbr_hook = RecencyNeighborHook(
    num_nbrs=[args.num_neighbors],  # Keep 1 slot for seed node itself
    num_nodes=num_nodes,
    seed_nodes_keys=['src', 'dst'],
    seed_times_keys=['time', 'time'],
)

hm = HookManager(keys=['train', 'val', 'test'])
hm.register('train', EdgeEventsSeenNodesTrackHook(num_nodes))
hm.register_shared(DeduplicationHook())
hm.register_shared(nbr_hook)

train_loader = DGDataLoader(train_dg, batch_size=args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, batch_size=args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, batch_size=args.bsize, hook_manager=hm)

if train_dg.static_node_feats is not None:
    static_node_feat = train_dg.static_node_feats
else:
    static_node_feat = torch.randn(
        (test_dg.num_nodes, args.node_dim), device=args.device
    ).to(args.device)

evaluator = Evaluator(name=args.dataset)
num_classes = train_dg.dynamic_node_feats_dim

random_projection_module = RandomProjectionModule(
    num_nodes=test_dg.num_nodes,
    num_layer=args.rp_num_layers,
    time_decay_weight=args.rp_time_decay_weight,
    beginning_time=train_dg.start_time,
    use_matrix=bool(args.use_matrix),
    enforce_dim=args.enforce_dim,
    num_edges=train_dg.num_edges,
    dim_factor=args.rp_dim_factor,
    concat_src_dst=bool(args.concat_src_dst),
    device=args.device,
).to(args.device)

encoder = TPNet_NodePrediction(
    node_feat_dim=static_node_feat.shape[1],
    edge_feat_dim=train_dg.edge_feats_dim,
    time_feat_dim=args.time_dim,
    output_dim=args.embed_dim,
    dropout=args.dropout,
    num_layers=args.num_layers,
    num_neighbors=args.num_neighbors,
    random_projection_module=random_projection_module,
    device=args.device,
    time_encoder=Time2Vec,
)

decoder = NodePredictor(
    in_dim=args.embed_dim, out_dim=num_classes, hidden_dim=args.embed_dim
).to(args.device)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate('train'):
        loss = train(train_loader, encoder, decoder, opt, static_node_feat)
    with hm.activate('val'):
        val_ndcg = eval(val_loader, encoder, decoder, evaluator, static_node_feat)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_NODEPROPPRED}', val_ndcg, epoch=epoch)

    save_results(
        f'{args.dataset}_{MODEL_NAME}_{args.seed}',
        {'epoch': epoch, 'val_ndcg': val_ndcg, 'loss': loss},
        f'epoch_log/{args.experiment_name}',
    )

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()
        encoder.rp_module.reset_random_projections()


with hm.activate('test'):
    test_ndcg = eval(test_loader, encoder, decoder, evaluator, static_node_feat)
log_metric(f'Test {METRIC_TGB_NODEPROPPRED}', test_ndcg, epoch=args.epochs)

# ==
save_results(
    f'{args.dataset}',
    {
        'dataset': args.dataset,
        'model': MODEL_NAME,
        'seed': args.seed,
        'test_mrr': test_ndcg,
    },
    args.experiment_name,
)
# ==

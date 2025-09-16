import argparse
import copy
import time
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import RecipeRegistry
from tgm.constants import METRIC_TGB_LINKPROPPRED, RECIPE_TGB_LINK_PRED
from tgm.graph import DGBatch, DGData, DGraph
from tgm.hooks import RecencyNeighborHook
from tgm.loader import DGDataLoader
from tgm.nn import RandomProjectionModule, Time2Vec, TPNet
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TPNet LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
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
    '--capture-gpu', action=argparse.BooleanOptionalAction, help='record peak gpu usage'
)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.fc1(torch.cat([z_src, z_dst], dim=1))
        h = h.relu()
        return self.fc2(h).sigmoid().view(-1)


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
        self.rp_module = random_projection_module.to(device)
        self.decoder = LinkPredictor(output_dim).to(
            device
        )  # @TODO: Make encoder/decoder to be explicit

    def forward(
        self, batch: DGBatch, static_node_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = batch.src
        dst = batch.dst
        neg = batch.neg
        time = batch.time
        nbr_nids = batch.nbr_nids[0]
        nbr_times = batch.nbr_times[0]
        nbr_feats = batch.nbr_feats[0]
        pos_batch_size = dst.shape[0]
        neg_batch_size = neg.shape[0]

        # positive edge
        edge_idx_pos = torch.stack((src, dst), dim=0)
        z_src_pos, z_dst_pos = self.encoder(
            static_node_feat,
            edge_idx_pos,
            time,
            nbr_nids[: pos_batch_size * 2],
            nbr_times[: pos_batch_size * 2],
            nbr_feats[: pos_batch_size * 2],
        )
        pos_out = self.decoder(z_src_pos, z_dst_pos)

        neg_nbr_nids = nbr_nids[
            -neg_batch_size:
        ]  # @TODO: Assume that batch.neg doesn't have duplicated records
        neg_nbr_times = nbr_times[-neg_batch_size:]
        neg_nbr_feats = nbr_feats[-neg_batch_size:]
        src_nbr_nids = nbr_nids[:pos_batch_size]
        src_nbr_times = nbr_times[:pos_batch_size]
        src_nbr_feats = nbr_feats[:pos_batch_size]

        if src.shape[0] != neg_batch_size:
            src = torch.repeat_interleave(src, repeats=neg_batch_size, dim=0)
            time = torch.repeat_interleave(time, repeats=neg_batch_size, dim=0)
            src_nbr_nids = torch.repeat_interleave(
                src_nbr_nids, repeats=neg_batch_size, dim=0
            )
            src_nbr_times = torch.repeat_interleave(
                src_nbr_times, repeats=neg_batch_size, dim=0
            )
            src_nbr_feats = torch.repeat_interleave(
                src_nbr_feats, repeats=neg_batch_size, dim=0
            )
            neg_nbr_nids = neg_nbr_nids.repeat(pos_batch_size, 1)
            neg_nbr_times = neg_nbr_times.repeat(pos_batch_size, 1)
            neg_nbr_feats = neg_nbr_feats.repeat(pos_batch_size, 1, 1)
            neg = neg.repeat(pos_batch_size)
        else:
            src_nbr_nids = nbr_nids[:pos_batch_size]
            src_nbr_times = nbr_times[:pos_batch_size]
            src_nbr_feats = nbr_feats[:pos_batch_size]

        edge_idx_neg = torch.stack((src, neg), dim=0)

        # negative edge
        z_src_neg, z_dst_neg = self.encoder(
            static_node_feat,
            edge_idx_neg,
            time,
            torch.cat([src_nbr_nids, neg_nbr_nids], dim=0),
            torch.cat([src_nbr_times, neg_nbr_times], dim=0),
            torch.cat([src_nbr_feats, neg_nbr_feats], dim=0),
        )
        neg_out = self.decoder(z_src_neg, z_dst_neg)
        self.rp_module.update(batch.src, batch.dst, time=batch.time)

        return pos_out, neg_out


def train(
    loader: DGDataLoader,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    static_node_feat: torch.Tensor,
) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch, static_node_feat)

        loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(
    evaluator: Evaluator,
    loader: DGDataLoader,
    model: nn.Module,
    static_node_feat: torch.Tensor,
) -> float:
    model.eval()
    perf_list = []
    for batch in tqdm(loader):
        copy_batch = copy.deepcopy(batch)
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            copy_batch.src = batch.src[idx].unsqueeze(0)
            copy_batch.dst = batch.dst[idx].unsqueeze(0)
            copy_batch.time = batch.time[idx].unsqueeze(0)
            copy_batch.neg = neg_batch
            neg_idx = (batch.neg == neg_batch[:, None]).nonzero(as_tuple=True)[1]

            # A tensor of index of src, dst and negative nodes to retrieve neighbor information
            all_idx = torch.cat(
                [
                    torch.Tensor([idx]).to(neg_batch.device),  # src idx
                    torch.Tensor([idx + batch.src.shape[0]]).to(
                        neg_batch.device
                    ),  # dst idx
                    neg_idx,
                ],
                dim=0,
            ).long()
            copy_batch.nbr_nids = [batch.nbr_nids[0][all_idx]]
            copy_batch.nbr_times = [batch.nbr_times[0][all_idx]]
            copy_batch.nbr_feats = [batch.nbr_feats[0][all_idx]]

            pos_out, neg_out = model(copy_batch, static_node_feat)

            input_dict = {
                'y_pred_pos': pos_out,
                'y_pred_neg': neg_out,
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

    return float(np.mean(perf_list))


args = parser.parse_args()
seed_everything(args.seed)

from pathlib import Path

from experiments import save_experiment_results_and_exit, setup_experiment
from tgm.util.perf import Usage

results = setup_experiment(args, Path(__file__))
u = Usage(gpu=args.capture_gpu).__enter__()

evaluator = Evaluator(name=args.dataset)

data = DGData.from_tgb(args.dataset)
dgraph = DGraph(data)

num_nodes = dgraph.num_nodes
edge_feats_dim = dgraph.edge_feats_dim

if dgraph.static_node_feats is not None:
    static_node_feat = dgraph.static_node_feats
else:
    static_node_feat = torch.randn((num_nodes, args.node_dim), device=args.device)

train_data, val_data, test_data = data.split()

train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

_, dst, _ = train_dg.edges
nbr_hook = RecencyNeighborHook(
    num_nbrs=[args.num_neighbors],
    num_nodes=num_nodes,
    edge_feats_dim=edge_feats_dim,
)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
hm.register_shared(nbr_hook)
registered_keys = hm.keys
train_key, val_key, test_key = registered_keys

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

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
    node_feat_dim=static_node_feat.shape[1],
    edge_feat_dim=edge_feats_dim,
    time_feat_dim=args.time_dim,
    output_dim=args.embed_dim,
    dropout=args.dropout,
    num_layers=args.num_layers,
    num_neighbors=args.num_neighbors,
    random_projection_module=random_projection_module,
    device=args.device,
    time_encoder=Time2Vec,
)

opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        start_time = time.perf_counter()
        loss = train(train_loader, model, opt, static_node_feat)
        end_time = time.perf_counter()
        latency = end_time - start_time
    with hm.activate(val_key):
        start_time = time.perf_counter()
        val_mrr = eval(evaluator, val_loader, model, static_node_feat)
        end_time = time.perf_counter()
        print(
            f'Epoch={epoch:02d} Latency={latency:.4f} Loss={loss:.4f} Validation {METRIC_TGB_LINKPROPPRED}={val_mrr:.4f}'
        )
    results[f'val_{METRIC_TGB_LINKPROPPRED}'] = val_mrr
    results['train_latency_s'] = latency
    results['val_latency_s'] = end_time - start_time
    u.__exit__()
    results['peak_gpu_gb'] = u.gpu_gb
    save_experiment_results_and_exit(results)

    # Clear memory state between epochs, except last epoch
    if epoch < args.epochs:
        hm.reset_state()
        model.rp_module.reset_random_projections()

with hm.activate(test_key):
    test_mrr = eval(evaluator, test_loader, model, static_node_feat)
    print(f'Test MRR:{METRIC_TGB_LINKPROPPRED}={test_mrr:.4f}')

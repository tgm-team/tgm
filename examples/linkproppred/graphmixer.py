import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.constants import (
    METRIC_TGB_LINKPROPPRED,
    PADDED_NODE_ID,
    RECIPE_TGB_LINK_PRED,
)
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry, StatelessHook
from tgm.nn import LinkPredictor, MLPMixer, Time2Vec
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GraphMixer LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-nbrs', type=int, default=20, help='num sampled nbrs')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=128, help='attention dimension')
parser.add_argument(
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
parser.add_argument(
    '--time-gap', type=int, default=2000, help='graphmixer time slot size'
)
parser.add_argument(
    '--token-dim-expansion',
    type=float,
    default=0.5,
    help='token dimension expansion factor in MLP sub-blocks',
)
parser.add_argument(
    '--channel-dim-expansion',
    type=float,
    default=4.0,
    help='channel dimension expansion factor in MLP sub-blocks',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


class GraphMixerEncoder(nn.Module):
    def __init__(
        self,
        time_dim: int,
        embed_dim: int,
        num_tokens: int,
        node_dim: int,
        edge_dim: int,
        num_layers: int = 2,
        token_dim_expansion: float = 0.5,
        channel_dim_expansion: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # GraphMixer time encoding function is not trainable
        self.time_encoder = Time2Vec(time_dim=time_dim)
        for param in self.time_encoder.parameters():
            param.requires_grad = False

        self.projection_layer = nn.Linear(edge_dim + time_dim, edge_dim)
        self.mlp_mixers = nn.ModuleList(
            [
                MLPMixer(
                    num_tokens=num_tokens,
                    num_channels=edge_dim,
                    token_dim_expansion_factor=token_dim_expansion,
                    channel_dim_expansion_factor=channel_dim_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(
            in_features=edge_dim + node_dim, out_features=embed_dim
        )

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        # Link Encoder
        edge_feat = batch.nbr_feats[0]
        nbr_time_feat = self.time_encoder(batch.times[0][:, None] - batch.nbr_times[0])
        z_link = self.projection_layer(torch.cat([edge_feat, nbr_time_feat], dim=-1))
        for mixer in self.mlp_mixers:
            z_link = mixer(z_link)

        valid_nbrs_mask = batch.nbr_nids[0] != PADDED_NODE_ID
        z_link = z_link * valid_nbrs_mask.unsqueeze(-1)
        z_link = z_link.sum(dim=1) / valid_nbrs_mask.sum(dim=1, keepdim=True).clamp(
            min=1
        )

        # Node Encoder
        num_nodes, feat_dim = len(batch.time_gap_nbrs), node_feat.shape[1]
        agg_feats = torch.zeros((num_nodes, feat_dim), device=node_feat.device)
        for i in range(num_nodes):
            if batch.time_gap_nbrs[i]:
                agg_feats[i] = node_feat[batch.time_gap_nbrs[i]].mean(dim=0)

        z_node = agg_feats + node_feat[torch.cat([batch.src, batch.dst, batch.neg])]
        z = self.output_layer(torch.cat([z_link, z_node], dim=1))
        return z


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in tqdm(loader):
        opt.zero_grad()

        z = encoder(batch, static_node_feats)
        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader,
    static_node_feats: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    evaluator: Evaluator,
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []

    for batch in tqdm(loader):
        z = encoder(batch, static_node_feats)
        id_map = {nid.item(): i for i, nid in enumerate(batch.nids[0])}
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.src[idx].repeat(len(dst_ids))

            src_idx = torch.tensor([id_map[n.item()] for n in src_ids], device=z.device)
            dst_idx = torch.tensor([id_map[n.item()] for n in dst_ids], device=z.device)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            y_pred = decoder(z_src, z_dst).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

train_data, val_data, test_data = DGData.from_tgb(args.dataset).split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)


class GraphMixerHook(StatelessHook):
    r"""Custom hook that gets 1-hop neighbors in a specific window.

    If N(v_i, t_s, t_e) = nbrs of v_i from [t_s, t_e], then we materialize
    N(node_ids, t - TIME_GAP, t) for all seed nodes in a given batch.
    """

    requires = {'neg'}
    produces = {'time_gap_nbrs'}

    def __init__(self, time_gap: int) -> None:
        self._time_gap = time_gap
        self._device = torch.device('cpu')

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        # Construct a the time_gap slice
        time_gap_slice = replace(dg._slice)
        time_gap_slice.start_idx = max(dg._slice.end_idx - self._time_gap, 0)
        time_gap_slice.end_time = int(batch.time.min()) - 1
        time_gap_src, time_gap_dst, _ = dg._storage.get_edges(time_gap_slice)

        nbr_index = defaultdict(list)
        for u, v in zip(time_gap_src.tolist(), time_gap_dst.tolist()):
            nbr_index[u].append(v)
            nbr_index[v].append(u)  # undirected

        seed_nodes = torch.cat([batch.src, batch.dst, batch.neg.to(dg.device)])
        batch.time_gap_nbrs = [nbr_index.get(nid, []) for nid in seed_nodes.tolist()]  # type: ignore
        return batch


hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(GraphMixerHook(args.time_gap))
hm.register_shared(
    RecencyNeighborHook(
        num_nbrs=[args.n_nbrs],
        num_nodes=test_dg.num_nodes,
        seed_nodes_keys=['src', 'dst', 'neg'],
        seed_times_keys=['time', 'time', 'neg_time'],
    )
)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)


if train_dg.static_node_feats is not None:
    static_node_feats = train_dg.static_node_feats
else:
    static_node_feats = torch.zeros(
        (test_dg.num_nodes, args.node_dim), device=args.device
    )

encoder = GraphMixerEncoder(
    node_dim=static_node_feats.shape[1],
    edge_dim=train_dg.edge_feats_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_tokens=args.n_nbrs,
    token_dim_expansion=float(args.token_dim_expansion),
    channel_dim_expansion=float(args.channel_dim_expansion),
    dropout=float(args.dropout),
).to(args.device)
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, static_node_feats, encoder, decoder, opt)

    with hm.activate(val_key):
        val_mrr = eval(val_loader, static_node_feats, encoder, decoder, evaluator)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_mrr = eval(test_loader, static_node_feats, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)

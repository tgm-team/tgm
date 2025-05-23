import argparse
import math
import time
from collections import defaultdict
from pprint import pprint
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from opendg.graph import DGBatch, DGraph
from opendg.hooks import NeighborSamplerHook
from opendg.loader import DGDataLoader
from opendg.nn import TemporalAttention, Time2Vec
from opendg.util.perf import Usage
from opendg.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='TGN Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=str, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--n-nbrs', type=int, default=[20], help='num sampled nbrs')
parser.add_argument('--time-dim', type=int, default=100, help='time encoding dimension')
parser.add_argument('--embed-dim', type=int, default=100, help='attention dimension')
parser.add_argument('--memory_dim', type=int, default=172, help='memory dimension')
parser.add_argument(
    '--sampling',
    type=str,
    default='recency',
    choices=['uniform', 'recency'],
    help='sampling strategy',
)


class TGN(torch.nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        num_layers: int,
        memory_dim: int,
    ):
        super().__init__()
        self.time_encoder = Time2Vec(time_dim=time_dim)
        self.msg_agg = LastMessageAggregator()
        self.msg_func = IdentityMessageFunction()

        msg_dim = 2 * memory_dim + edge_dim + time_dim
        self.n_nodes = 100  # TODO
        self.memory = Memory(n_nodes=self.n_nodes, memory_dim=memory_dim)
        self.memory_updater = GRUMemoryUpdater(self.memory, msg_dim, memory_dim)
        self.gat = GraphAttentionEmbedding(
            time_encoder=self.time_encoder,
            num_layers=num_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=node_dim,
            embed_dim=embed_dim,
        )

    def forward(self, batch: DGBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = list(range(self.n_nodes))
        unique_nids, unique_msg, unique_times = self.msg_agg(nodes, self.memory.msgs)
        if len(unique_nids) > 0:
            unique_msg = self.msg_func(unique_msg)
        memory, last_update = self.memory_updater.get_updated_memory(
            unique_nids, unique_msg, unique_times
        )
        batch.time[batch.src] -= last_update[batch.src].long()
        batch.time[batch.dst] -= last_update[batch.dst].long()
        batch.time[batch.neg] -= last_update[batch.neg].long()
        pos_out, neg_out = self.gat(batch, memory=memory)
        self._update(batch, memory)
        return pos_out, neg_out

    def _update(self, batch: DGBatch, memory: Memory) -> None:
        def _get_raw_msgs(src, dst, time, edge_idxs):
            time = torch.from_numpy(time).float()
            edge_feats = self.edge_raw_features[edge_idxs]
            src_memory = self.memory.get_memory(src)
            dst_memory = self.memory.get_memory(dst)
            time_delta = time - self.memory.last_update[src]
            time_feat = self.time_encoder(time_delta.unsqueeze(dim=1)).view(
                len(src), -1
            )

            src_msg = torch.cat([src_memory, dst_memory, edge_feats, time_feat], dim=1)
            msgs = defaultdict(list)
            unique_src = np.unique(src)
            for i in range(len(src)):
                msgs[src[i]].append((src_msg[i], time[i]))
            return unique_src, msgs

        # Persist the updates to the memory only for sources and destinations
        src, dst, time, edge_idxs = batch.src, batch.dst, batch.time, batch.nbr_nids
        pos = np.concatenate([src, dst])
        unique_nids, unique_msg, unique_times = self.msg_agg(pos, self.memory.msg)
        if len(unique_nids) > 0:
            unique_msg = self.msg_func(unique_msg)
        self.memory_updater.update(unique_nids, unique_msg, time=unique_times)
        assert torch.allclose(
            memory[pos], self.memory.get_memory(pos), atol=1e-5
        ), 'Something wrong in how the memory was updated'
        # Remove msgs for the pos since we have already updated the memory using them
        self.memory.clear_msgs(pos)
        unique_src, src_to_msgs = _get_raw_msgs(src, dst, time, edge_idxs)
        unique_dst, dst_to_msgs = _get_raw_msgs(dst, src, time, edge_idxs)
        self.memory.store_raw_msgs(unique_src, src_to_msgs)
        self.memory.store_raw_msgs(unique_dst, dst_to_msgs)


class LastMessageAggregator(torch.nn.Module):
    def forward(
        self, node_ids: torch.Tensor, msgs: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        unique_nids = np.unique(node_ids)
        unique_msg, unique_times, to_update_node_ids = [], [], []
        for node_id in unique_nids:
            if len(msgs[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_msg.append(msgs[node_id][-1][0])
                unique_times.append(msgs[node_id][-1][1])
        unique_msg = torch.stack(unique_msg) if len(to_update_node_ids) > 0 else []
        unique_times = torch.stack(unique_times) if len(to_update_node_ids) > 0 else []
        return to_update_node_ids, unique_msg, unique_times


class IdentityMessageFunction(nn.Module):
    def forward(self, raw_msgs: torch.Tensor) -> torch.Tensor:
        return raw_msgs


class GraphAttentionEmbedding(nn.Module):
    def __init__(
        self,
        time_encoder: Time2Vec,
        num_layers: int,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        embed_dim: int,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_encoder = time_encoder
        self.link_predictor = LinkPredictor(self.node_dim)
        self.attn = nn.ModuleList(
            [
                TemporalAttention(
                    n_heads=n_heads,
                    node_dim=node_dim if i == 0 else embed_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    out_dim=embed_dim,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self, batch: DGBatch, memory: Memory
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Go back to recursive embedding for multi-hop
        hop = 0
        node_feat = torch.zeros((*batch.nids[hop].shape, self.embed_dim))
        nbr_node_feat = torch.zeros((*batch.nbr_nids[hop].shape, self.embed_dim))

        z = self.attn[hop](
            node_feat=node_feat + memory[batch.nids[hop], :],
            nbr_node_feat=nbr_node_feat + memory[batch.nbr_nids, :],
            time_feat=self.time_encoder(torch.zeros(len(batch.nids[hop]))),
            nbr_time_feat=self.time_encoder(
                batch.nbr_times[hop] - batch.time.unsqueeze(dim=1).repeat(3, 1)
            ),
            edge_feat=batch.nbr_feats[hop],
            nbr_mask=batch.nbr_mask[hop],
        )
        z_src, z_dst, z_neg = z.chunk(3, dim=0)
        pos_out = self.link_predictor(z_src, z_dst)
        neg_out = self.link_predictor(z_src, z_neg)
        return pos_out, neg_out


class GRUMemoryUpdater(nn.Module):
    def __init__(self, memory: Memory, msg_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.memory = memory
        self.memory_updater = nn.GRUCell(input_size=msg_dim, hidden_size=memory_dim)

    def update(
        self, unique_nids: torch.Tensor, unique_msg: torch.Tensor, time: int
    ) -> None:
        if len(unique_nids) <= 0:
            return
        assert (
            (self.memory.get_last_update(unique_nids) <= time).all().item()
        ), 'Trying to update memory to time in the past'
        memory = self.memory.get_memory(unique_nids)
        self.memory.last_update[unique_nids] = time
        updated_memory = self.memory_updater(unique_msg, memory)
        self.memory.set_memory(unique_nids, updated_memory)

    def get_updated_memory(
        self, unique_nids: torch.Tensor, unique_msg: torch.Tensor, time: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(unique_nids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        assert (
            (self.memory.get_last_update(unique_nids) <= time).all().item()
        ), 'Trying to update memory to time in the past'
        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_nids] = self.memory_updater(
            unique_msg, updated_memory[unique_nids]
        )
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_nids] = time
        return updated_memory, updated_last_update


class Memory(nn.Module):
    def __init__(self, n_nodes: int, memory_dim: int) -> None:
        super().__init__()
        self.memory = nn.Parameter(
            torch.zeros((n_nodes, memory_dim)), requires_grad=False
        )
        self.last_update = nn.Parameter(torch.zeros(n_nodes), requires_grad=False)
        self.msgs = defaultdict(list)

    def store_raw_msgs(
        self, nodes: torch.Tensor, node_id_to_msg: Dict[int, torch.Tensor]
    ) -> None:
        for node in nodes:
            self.msgs[node].extend(node_id_to_msg[node])

    def get_memory(self, node_idxs: torch.Tensor) -> torch.Tensor:
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs: torch.Tensor, values: torch.Tensor) -> None:
        self.memory[node_idxs, :] = values

    def get_last_update(self, node_idxs: torch.Tensor) -> torch.Tensor:
        return self.last_update[node_idxs]

    def backup_memory(self) -> Tuple[torch.Tensor, ...]:
        msgs_clone = {}
        for k, v in self.msgs.items():
            msgs_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
        return self.memory.data.clone(), self.last_update.data.clone(), msgs_clone

    def restore_memory(self, memory_backup: Tuple[torch.Tensor, ...]) -> None:
        self.memory.data, self.last_update.data = (
            memory_backup[0].clone(),
            memory_backup[1].clone(),
        )
        self.msgs = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.msgs[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self) -> None:
        self.memory.detach_()
        for k, v in self.msgs.items():
            new_node_msg = []
            for msg in v:
                new_node_msg.append((msg[0].detach(), msg[1]))
            self.msgs[k] = new_node_msg

    def clear_msgs(self, nodes: torch.Tensor) -> None:
        for node in nodes:
            self.msgs[node] = []


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


def train(loader: DGDataLoader, model: nn.Module, opt: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        pos_out, neg_out = model(batch)
        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@torch.no_grad()
def eval(loader: DGDataLoader, model: nn.Module, metrics: Metric) -> None:
    model.eval()
    for batch in tqdm(loader):
        pos_out, neg_out = model(batch)
        y_pred = torch.cat([pos_out, neg_out], dim=0).float()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        ).long()
        indexes = torch.zeros(y_pred.size(0), dtype=torch.long)
        metrics(y_pred, y_true, indexes=indexes)
    pprint(metrics.compute())


args = parser.parse_args()
seed_everything(args.seed)

train_dg = DGraph(args.dataset, split='train')
val_dg = DGraph(args.dataset, split='valid')
test_dg = DGraph(args.dataset, split='test')

train_loader = DGDataLoader(
    train_dg,
    hook=NeighborSamplerHook(num_nbrs=args.n_nbrs),
    batch_size=args.bsize,
)
val_loader = DGDataLoader(
    val_dg,
    hook=NeighborSamplerHook(num_nbrs=args.n_nbrs),
    batch_size=args.bsize,
)
test_loader = DGDataLoader(
    test_dg,
    hook=NeighborSamplerHook(num_nbrs=args.n_nbrs),
    batch_size=args.bsize,
)

device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
model = TGAT(
    node_dim=train_dg.node_feats_dim or args.embed_dim,  # TODO: verify
    edge_dim=train_dg.edge_feats_dim or args.embed_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_layers=len(args.n_nbrs),
    n_heads=args.n_heads,
    dropout=float(args.dropout),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

metrics = [BinaryAveragePrecision(), BinaryAUROC()]
val_metrics = MetricCollection(metrics, prefix='Validation')
test_metrics = MetricCollection(metrics, prefix='Test')

with Usage(prefix='TGN Training'):
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader, model, opt)
        pprint(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        eval(val_loader, model, val_metrics)
        eval(test_loader, model, test_metrics)
        val_metrics.reset()
        test_metrics.reset()


BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim


### Extract data for training, validation and testing
(
    node_features,
    edge_features,
    full_data,
    train_data,
    val_data,
    test_data,
    new_node_val_data,
    new_node_test_data,
) = get_data(
    args.data,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=True,
)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(
    new_node_val_data.sources, new_node_val_data.destinations, seed=1
)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(
    new_node_test_data.sources, new_node_test_data.destinations, seed=3
)

# Initialize Model
tgn = TGN(
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=NUM_LAYER,
    n_heads=NUM_HEADS,
    dropout=DROP_OUT,
    use_memory=USE_MEMORY,
    message_dimension=MESSAGE_DIM,
    memory_dim=MEMORY_DIM,
    n_neighbors=NUM_NEIGHBORS,
)
criterion = torch.nn.BCELoss()
tgn = tgn.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

print('num of training instances: {}'.format(num_instance))
print('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)

new_nodes_val_aps = []
val_aps = []
epoch_times = []
total_epoch_times = []
train_losses = []

for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    tgn.memory.__init_memory__()

    # Train using only training graph
    m_loss = []

    print('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
        loss = 0
        opt.zero_grad()

        # Custom loop to allow to perform backpropagation only every a certain number of batches
        for j in range(args.backprop_every):
            batch_idx = k + j

            if batch_idx >= num_batch:
                continue

            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = (
                train_data.sources[start_idx:end_idx],
                train_data.destinations[start_idx:end_idx],
            )
            edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
            timestamps_batch = train_data.timestamps[start_idx:end_idx]

            size = len(sources_batch)
            _, negatives_batch = train_rand_sampler.sample(size)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            tgn = tgn.train()
            pos_prob, neg_prob = tgn.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negatives_batch,
                timestamps_batch,
                edge_idxs_batch,
                NUM_NEIGHBORS,
            )

            loss += criterion(pos_prob.squeeze(), pos_label) + criterion(
                neg_prob.squeeze(), neg_label
            )

        loss /= args.backprop_every

        loss.backward()
        opt.step()
        m_loss.append(loss.item())

        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Backup memory at the end of training, so later we can restore it and use it for the
    # validation on unseen nodes
    train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=val_rand_sampler,
        data=val_data,
        n_neighbors=NUM_NEIGHBORS,
    )
    val_memory_backup = tgn.memory.backup_memory()
    # Restore memory we had at the end of training to be used when validating on new nodes.
    # Also backup memory after validation so it can be used for testing (since test edges are
    # strictly later in time than validation edges)
    tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_edge_prediction(
        model=tgn,
        negative_edge_sampler=val_rand_sampler,
        data=new_node_val_data,
        n_neighbors=NUM_NEIGHBORS,
    )

    # Restore memory we had at the end of validation
    tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    print('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    print('Epoch mean loss: {}'.format(np.mean(m_loss)))
    print('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    print('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))


# Training has finished, we have loaded the best model, and we want to backup its current
# memory (which has seen validation edges) so that it can also be used when testing on unseen
# nodes
val_memory_backup = tgn.memory.backup_memory()

### Test
test_ap, test_auc = eval_edge_prediction(
    model=tgn,
    negative_edge_sampler=test_rand_sampler,
    data=test_data,
    n_neighbors=NUM_NEIGHBORS,
)

tgn.memory.restore_memory(val_memory_backup)

# Test on unseen nodes
nn_test_ap, nn_test_auc = eval_edge_prediction(
    model=tgn,
    negative_edge_sampler=nn_test_rand_sampler,
    data=new_node_test_data,
    n_neighbors=NUM_NEIGHBORS,
)

print('Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
print('Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))

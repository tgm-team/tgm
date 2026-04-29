"""Phase 4-5: TGN training and evaluation loop for rel-hm user-item-purchase.

Run:
    python -m examples.linkproppred.relbench.train [options]

The script:
  - Loads DGData via data.py (Phase 1-3).
  - Optionally builds static node features via embed.py (Phase 2).
  - Instantiates TGNMemory + GraphAttentionEmbedding + LinkPredictor.
  - Trains with binary cross-entropy loss + random negative sampling.
  - Evaluates on val/test using Average Precision (AP) and NDCG@10.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGDataLoader
from tgm.hooks import DeduplicationHook, HookManager, RecencyNeighborHook
from tgm.hooks.negatives import NegativeEdgeSamplerHook
from tgm.nn import LinkPredictor, TGNMemory
from tgm.nn.encoder.tgn import (
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
)

# ---------------------------------------------------------------------------
# Memory-efficient replacements for TGNMemory internals
# ---------------------------------------------------------------------------


class _MemoryEfficientLastAggregator(LastAggregator):
    """Replaces the O(N×M) scores matrix in LastAggregator with an O(M log M) sort.

    The original implementation allocates a [dim_size, num_messages] float tensor.
    When memory.eval() flushes all 1.48 M nodes at once, this is ~8.7 TB → OOM.
    The sort-based approach uses O(M) space only.
    """

    def forward(
        self,
        msg: torch.Tensor,
        index: torch.Tensor,
        t: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        out = msg.new_zeros((dim_size, msg.size(-1)))
        if index.numel() > 0:
            # Double stable sort: (index asc, t asc) → last element per group = max t.
            perm = torch.argsort(t, stable=True)
            perm = perm[torch.argsort(index[perm], stable=True)]
            sorted_idx = index[perm]
            node_ids, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
            last_pos = counts.cumsum(0) - 1
            out[node_ids] = msg[perm[last_pos]]
        return out


def _patch_memory_for_large_graphs(memory: TGNMemory) -> None:
    """Patch a TGNMemory instance to reduce memory usage with large node counts.

    Applies three changes without modifying tgm library code:
    1. Sparse msg_store: uses a plain dict + sentinel instead of pre-populating
       1.48 M entries (saves ~300 MB of Python objects per epoch reset).
    2. Cloned msg_store entries: stored tensors are clones, not views of batch
       tensors, so old batch GPU storage is freed between batches.
    3. Safe dict access: _compute_msg uses .get() with the sentinel.
    """
    import types

    # Create the empty sentinel once.
    _i = memory.memory.new_empty((0,), device=memory.device, dtype=torch.long)
    _msg = memory.memory.new_empty((0, memory.raw_msg_dim), device=memory.device)
    sentinel = (_i, _i, _i, _msg)
    memory._msg_sentinel = sentinel  # type: ignore[attr-defined]

    def _reset_message_store(self: TGNMemory) -> None:
        _i2 = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        _m2 = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        self._msg_sentinel = (_i2, _i2, _i2, _m2)  # type: ignore[attr-defined]
        self.msg_s_store = {}
        self.msg_d_store = {}

    def _update_msg_store(
        self: TGNMemory,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        raw_msg: torch.Tensor,
        msg_store: dict,
    ) -> None:
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            # Clone so the store doesn't hold views into the batch tensor; otherwise
            # the batch's GPU storage stays live until every node from that batch is
            # updated again (potentially thousands of batches later).
            msg_store[i] = (
                src[idx].clone(),
                dst[idx].clone(),
                t[idx].clone(),
                raw_msg[idx].clone(),
            )

    def _compute_msg(
        self: TGNMemory,
        n_id: torch.Tensor,
        msg_store: dict,
        msg_module: nn.Module,
    ) -> tuple:
        sent = self._msg_sentinel  # type: ignore[attr-defined]
        data = [msg_store.get(i, sent) for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0).to(self.device)
        dst = torch.cat(dst, dim=0).to(self.device)
        t = torch.cat(t, dim=0).to(self.device)
        raw_msg = torch.cat(raw_msg, dim=0).to(self.device)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, src, dst

    memory._reset_message_store = types.MethodType(_reset_message_store, memory)  # type: ignore[method-assign]
    memory._update_msg_store = types.MethodType(_update_msg_store, memory)  # type: ignore[method-assign]
    memory._compute_msg = types.MethodType(_compute_msg, memory)  # type: ignore[method-assign]
    # Re-initialise with the new (sparse) reset method.
    memory.reset_state()


from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything

from .data import build_relbench_hm_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='TGN rel-hm LinkPropPred',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--bsize', type=int, default=200)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--time-dim', type=int, default=100)
parser.add_argument('--embed-dim', type=int, default=100)
parser.add_argument('--memory-dim', type=int, default=100)
parser.add_argument('--n-nbrs', type=int, nargs='+', default=[10])
parser.add_argument(
    '--use-static-features',
    action='store_true',
    help='Embed article/customer tables and prepend to TGN memory',
)
parser.add_argument(
    '--joint-static',
    action='store_true',
    help='Make static node embeddings trainable (requires --use-static-features)',
)
parser.add_argument('--log-file-path', type=str, default=None)


# ---------------------------------------------------------------------------
# Phase 4.2 — Evaluation helpers
# ---------------------------------------------------------------------------


def _ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@k given a binary relevance array (first element is positive)."""
    k = min(k, len(relevance))
    ideal = np.sort(relevance)[::-1][:k]
    actual = relevance[:k]
    gains = 2**actual - 1
    ideal_gains = 2**ideal - 1
    discounts = np.log2(np.arange(2, k + 2))
    idcg = (ideal_gains / discounts).sum()
    dcg = (gains / discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_ap_ndcg(scores: np.ndarray, labels: np.ndarray, k: int = 10):
    """Return (AP, NDCG@k) given parallel score and binary-label arrays."""
    ap = float(average_precision_score(labels, scores)) if labels.sum() > 0 else 0.0
    # Sort by score descending, reorder labels accordingly
    order = np.argsort(scores)[::-1]
    ndcg = _ndcg_at_k(labels[order], k)
    return ap, ndcg


# ---------------------------------------------------------------------------
# Phase 5.1/5.2 — Model helpers
# ---------------------------------------------------------------------------


class StaticAugmentedEncoder(nn.Module):
    """Wraps GraphAttentionEmbedding, fusing static node features into memory.

    Args:
        trainable_static: If True, ``static_node_x`` is registered as an
            ``nn.Parameter`` so its values are updated during back-prop
            (jointly-trained ablation).  If False (default) it is a frozen
            buffer.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        static_node_x: torch.Tensor,
        static_dim: int,
        memory_dim: int,
        trainable_static: bool = False,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        if trainable_static:
            self.static_node_x = nn.Parameter(static_node_x)
        else:
            self.register_buffer('static_node_x', static_node_x)
        self.proj = nn.Linear(memory_dim + static_dim, memory_dim)

    def forward(
        self,
        z,
        last_update,
        nbr_edge_index,
        nbr_edge_time,
        nbr_edge_x,
        unique_nids=None,
    ):
        if unique_nids is not None:
            static = self.static_node_x[unique_nids]
            z = self.proj(torch.cat([z, static], dim=-1))
        return self.base_encoder(
            z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x
        )


# ---------------------------------------------------------------------------
# Phase 5.3 — Training loop
# ---------------------------------------------------------------------------


@log_gpu
@log_latency
def train(loader, memory, encoder, decoder, opt, use_static=False):
    memory.train()
    encoder.train()
    decoder.train()
    memory.reset_state()
    total_loss = 0.0

    for batch in tqdm(loader, desc='train'):
        opt.zero_grad()

        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID
        num_nbrs = len(nbr_nodes) // (
            len(batch.edge_src) + len(batch.edge_dst) + len(batch.neg)
        )
        src_nodes = torch.cat(
            [
                batch.edge_src.repeat_interleave(num_nbrs),
                batch.edge_dst.repeat_interleave(num_nbrs),
                batch.neg.repeat_interleave(num_nbrs),
            ]
        )
        nbr_edge_index = torch.stack(
            [
                batch.global_to_local(src_nodes[nbr_mask]),
                batch.global_to_local(nbr_nodes[nbr_mask]),
            ]
        ).to(dtype=torch.int64)
        nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
        nbr_edge_x = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        if use_static and isinstance(encoder, StaticAugmentedEncoder):
            z = encoder(
                z,
                last_update,
                nbr_edge_index,
                nbr_edge_time,
                nbr_edge_x,
                unique_nids=batch.unique_nids,
            )
        else:
            z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

        inv_src = batch.global_to_local(batch.edge_src)
        inv_dst = batch.global_to_local(batch.edge_dst)
        inv_neg = batch.global_to_local(batch.neg)
        pos_out = decoder(z[inv_src], z[inv_dst])
        neg_out = decoder(z[inv_src], z[inv_neg])

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        memory.update_state(
            batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float()
        )
        loss.backward()
        opt.step()
        memory.detach()
        total_loss += float(loss)

    return total_loss


# ---------------------------------------------------------------------------
# Phase 5.4 — Validation / test evaluation
# ---------------------------------------------------------------------------


@log_gpu
@log_latency
@torch.no_grad()
def evaluate(loader, memory, encoder, decoder, device, use_static=False):
    """Evaluate using AP, NDCG@10, and per-customer AP (RelBench protocol).

    For each batch event we score the positive dst and one random negative dst.
    Three metrics are returned:

    - *Global AP*: average precision computed over all (score, label) pairs.
    - *NDCG@10*: computed from the globally ranked scores.
    - *Per-customer AP*: for each customer, AP is computed over only that
      customer's positive/negative pairs, then averaged across customers.
      This matches the RelBench eval protocol.
    """
    memory.eval()
    encoder.eval()
    decoder.eval()

    all_scores: list = []
    all_labels: list = []
    # per_cust maps customer_id -> {'scores': [...], 'labels': [...]}
    per_cust: dict = {}

    for batch in tqdm(loader, desc='eval'):
        nbr_nodes = batch.nbr_nids[0].flatten()
        nbr_mask = nbr_nodes != PADDED_NODE_ID
        num_nbrs = len(nbr_nodes) // (
            len(batch.edge_src) + len(batch.edge_dst) + len(batch.neg)
        )
        src_nodes = torch.cat(
            [
                batch.edge_src.repeat_interleave(num_nbrs),
                batch.edge_dst.repeat_interleave(num_nbrs),
                batch.neg.repeat_interleave(num_nbrs),
            ]
        )
        nbr_edge_index = torch.stack(
            [
                batch.global_to_local(src_nodes[nbr_mask]),
                batch.global_to_local(nbr_nodes[nbr_mask]),
            ]
        ).to(dtype=torch.int64)
        nbr_edge_time = batch.nbr_edge_time[0].flatten()[nbr_mask]
        nbr_edge_x = batch.nbr_edge_x[0].flatten(0, -2).float()[nbr_mask]

        z, last_update = memory(batch.unique_nids)
        if use_static and isinstance(encoder, StaticAugmentedEncoder):
            z = encoder(
                z,
                last_update,
                nbr_edge_index,
                nbr_edge_time,
                nbr_edge_x,
                unique_nids=batch.unique_nids,
            )
        else:
            z = encoder(z, last_update, nbr_edge_index, nbr_edge_time, nbr_edge_x)

        inv_src = batch.global_to_local(batch.edge_src)
        inv_dst = batch.global_to_local(batch.edge_dst)
        inv_neg = batch.global_to_local(batch.neg)

        pos_scores = decoder(z[inv_src], z[inv_dst]).sigmoid().cpu().numpy()
        neg_scores = decoder(z[inv_src], z[inv_neg]).sigmoid().cpu().numpy()
        cust_ids = batch.edge_src.cpu().numpy()

        for cid, ps, ns in zip(cust_ids, pos_scores, neg_scores):
            all_scores.extend([float(ps), float(ns)])
            all_labels.extend([1, 0])
            cid = int(cid)
            if cid not in per_cust:
                per_cust[cid] = {'scores': [], 'labels': []}
            per_cust[cid]['scores'].extend([float(ps), float(ns)])
            per_cust[cid]['labels'].extend([1, 0])

        memory.update_state(
            batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_x.float()
        )

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    ap, ndcg = compute_ap_ndcg(scores, labels)

    # Per-customer AP (RelBench protocol)
    per_cust_aps = [
        float(average_precision_score(v['labels'], v['scores']))
        for v in per_cust.values()
        if sum(v['labels']) > 0
    ]
    mean_per_cust_ap = float(np.mean(per_cust_aps)) if per_cust_aps else 0.0

    return ap, ndcg, mean_per_cust_ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()
    enable_logging(log_file_path=args.log_file_path)
    seed_everything(args.seed)

    # --- Data ---
    static_node_x = None
    if args.use_static_features:
        from .data import load_raw_tables
        from .embed import build_static_node_features

        _, table_article, table_customer, _ = load_raw_tables()
        static_node_x = build_static_node_features(table_article, table_customer)

    full_data, meta = build_relbench_hm_data(static_node_x=static_node_x)
    train_data, val_data, test_data = full_data.split()

    train_dg = DGraph(train_data, device=args.device)
    val_dg = DGraph(val_data, device=args.device)
    test_dg = DGraph(test_data, device=args.device)

    # --- Hooks ---
    # Negatives are sampled from [0, N_art) — articles only, since all edges
    # are customer → article.
    neg_hook = NegativeEdgeSamplerHook(low=0, high=meta.n_articles)
    nbr_hook = RecencyNeighborHook(
        num_nbrs=args.n_nbrs,
        num_nodes=meta.n_nodes,
        seed_nodes_keys=['edge_src', 'edge_dst', 'neg'],
        seed_times_keys=['edge_time', 'edge_time', 'neg_time'],
    )

    hm = HookManager(keys=['train', 'val', 'test'])
    hm.register('train', neg_hook)
    hm.register('val', NegativeEdgeSamplerHook(low=0, high=meta.n_articles))
    hm.register('test', NegativeEdgeSamplerHook(low=0, high=meta.n_articles))
    hm.register_shared(nbr_hook)
    hm.register_shared(DeduplicationHook(seed_nodes_keys=['neg', 'nbr_nids']))

    train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
    val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
    test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

    # --- Model ---
    edge_x_dim = meta.edge_x_dim  # 2

    memory = TGNMemory(
        meta.n_nodes,
        edge_x_dim,
        args.memory_dim,
        args.time_dim,
        message_module=IdentityMessage(edge_x_dim, args.memory_dim, args.time_dim),
        # Use memory-efficient aggregator: avoids the O(N×M) scores matrix that
        # would be ~8.7 TB when flushing all 1.48 M nodes at memory.eval() time.
        aggregator_module=_MemoryEfficientLastAggregator(),
    ).to(args.device)
    # Patch msg_store internals for large-graph efficiency (sparse dict, clones).
    _patch_memory_for_large_graphs(memory)

    _base_encoder = GraphAttentionEmbedding(
        in_channels=args.memory_dim,
        out_channels=args.embed_dim,
        msg_dim=edge_x_dim,
        time_enc=memory.time_enc,
    ).to(args.device)

    if args.use_static_features and static_node_x is not None:
        from .embed import TARGET_DIM

        encoder = StaticAugmentedEncoder(
            base_encoder=_base_encoder,
            static_node_x=static_node_x.to(args.device),
            static_dim=TARGET_DIM,
            memory_dim=args.memory_dim,
            trainable_static=args.joint_static,
        ).to(args.device)
    else:
        encoder = _base_encoder

    decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
        args.device
    )

    opt = torch.optim.Adam(
        set(memory.parameters())
        | set(encoder.parameters())
        | set(decoder.parameters()),
        lr=args.lr,
    )

    # --- Training loop ---
    train_key, val_key, test_key = hm.keys

    for epoch in range(1, args.epochs + 1):
        with hm.activate(train_key):
            loss = train(
                train_loader,
                memory,
                encoder,
                decoder,
                opt,
                use_static=args.use_static_features,
            )

        with hm.activate(val_key):
            val_ap, val_ndcg, val_pc_ap = evaluate(
                val_loader,
                memory,
                encoder,
                decoder,
                args.device,
                use_static=args.use_static_features,
            )

        log_metric('Loss', loss, epoch=epoch)
        log_metric('Val AP', val_ap, epoch=epoch)
        log_metric('Val NDCG@10', val_ndcg, epoch=epoch)
        log_metric('Val Per-Customer AP', val_pc_ap, epoch=epoch)
        logger.info(
            'Epoch %d — loss=%.4f  val_AP=%.4f  val_NDCG@10=%.4f  val_PerCustAP=%.4f',
            epoch,
            loss,
            val_ap,
            val_ndcg,
            val_pc_ap,
        )

        if epoch < args.epochs:
            hm.reset_state()

    with hm.activate(test_key):
        test_ap, test_ndcg, test_pc_ap = evaluate(
            test_loader,
            memory,
            encoder,
            decoder,
            args.device,
            use_static=args.use_static_features,
        )

    log_metric('Test AP', test_ap, epoch=args.epochs)
    log_metric('Test NDCG@10', test_ndcg, epoch=args.epochs)
    log_metric('Test Per-Customer AP', test_pc_ap, epoch=args.epochs)
    logger.info(
        'Test — AP=%.4f  NDCG@10=%.4f  PerCustAP=%.4f',
        test_ap,
        test_ndcg,
        test_pc_ap,
    )


if __name__ == '__main__':
    main()

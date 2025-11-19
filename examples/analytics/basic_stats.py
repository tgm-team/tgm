import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, StatelessHook
from tgm.util.logging import enable_logging
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Basic Batch Statistics Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--bsize', type=int, default=10, help='batch size')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)
logger = logging.getLogger('tgm').getChild(Path(__file__).stem)


class BasicBatchStatsHook(StatelessHook):
    """Compute simple batch-level statistics."""

    produces = {
        'num_edge_events',
        'num_node_events',
        'num_timestamps',
        'num_unique_nodes',
        'avg_degree',
        'num_repeated_events',
    }

    @staticmethod
    def count_edge_events(batch):
        return int(batch.src.numel()) if batch.src is not None else 0

    @staticmethod
    def count_node_events(batch):
        return int(batch.node_ids.numel()) if batch.node_ids is not None else 0

    @staticmethod
    def count_timestamps(batch):
        num_edge_ts = batch.time.numel() if batch.time is not None else 0
        num_node_ts = batch.node_times.numel() if batch.node_times is not None else 0
        return num_edge_ts + num_node_ts

    @staticmethod
    def compute_unique_nodes(batch):
        node_tensors = []
        if batch.src is not None and batch.src.numel() > 0:
            node_tensors.append(batch.src)
        if batch.dst is not None and batch.dst.numel() > 0:
            node_tensors.append(batch.dst)
        if batch.node_ids is not None and batch.node_ids.numel() > 0:
            node_tensors.append(batch.node_ids)

        if len(node_tensors) == 0:
            return 0

        all_nodes = torch.cat(node_tensors, dim=0)
        unique_nodes = torch.unique(all_nodes)
        return int(unique_nodes.numel())

    @staticmethod
    def compute_avg_degree(batch):
        src, dst = batch.src, batch.dst
        if src is None or dst is None or src.numel() == 0:
            return 0.0

        edge_nodes = torch.cat([src, dst], dim=0)
        edge_unique_nodes, edge_inverse = torch.unique(edge_nodes, return_inverse=True)
        degree_per_node = torch.bincount(
            edge_inverse, minlength=edge_unique_nodes.numel()
        )
        return float(degree_per_node.float().mean().item())

    @staticmethod
    def count_repeated_edge_events(batch):
        src, dst, time = batch.src, batch.dst, batch.time
        if src is None or dst is None or time is None or src.numel() == 0:
            return 0

        # (E, 3)
        edge_triples = torch.stack(
            [src.to(torch.long), dst.to(torch.long), time.to(torch.long)], dim=1
        )

        _, edge_counts = torch.unique(edge_triples, dim=0, return_counts=True)
        return int((edge_counts - 1).clamp(min=0).sum().item())

    @staticmethod
    def count_repeated_node_events(batch):
        if batch.node_ids is None or batch.node_ids.numel() == 0:
            return 0

        node_ids = batch.node_ids.to(torch.long)
        if batch.node_times is None:
            return 0  # cannot detect duplicates without time

        node_times = batch.node_times.to(torch.long)
        node_pairs = torch.stack([node_ids, node_times], dim=1)

        _, node_counts = torch.unique(node_pairs, dim=0, return_counts=True)
        return int((node_counts - 1).clamp(min=0).sum().item())

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.num_edge_events = self.count_edge_events(batch)
        batch.num_node_events = self.count_node_events(batch)
        batch.num_timestamps = self.count_timestamps(batch)
        batch.num_unique_nodes = self.compute_unique_nodes(batch)
        batch.avg_degree = self.compute_avg_degree(batch)

        edge_rep = self.count_repeated_edge_events(batch)
        node_rep = self.count_repeated_node_events(batch)
        batch.num_repeated_events = edge_rep + node_rep

        return batch


if __name__ == '__main__':
    seed_everything(args.seed)

    dg = DGraph(DGData.from_tgb(args.dataset))

    # Register hook
    stats_hook = BasicBatchStatsHook()
    hm = HookManager(keys=['basic_batch_stats'])
    hm.register('basic_batch_stats', stats_hook)
    hm.set_active_hooks('basic_batch_stats')

    loader = DGDataLoader(dg, args.bsize, hook_manager=hm)

    edge_events_list = []
    node_events_list = []
    timestamps_list = []
    unique_nodes_list = []
    avg_degree_list = []
    repeated_events_list = []

    for batch in tqdm(loader, desc='Computing basic batch stats'):
        edge_events_list.append(batch.num_edge_events)
        node_events_list.append(batch.num_node_events)
        timestamps_list.append(batch.num_timestamps)
        unique_nodes_list.append(batch.num_unique_nodes)
        avg_degree_list.append(batch.avg_degree)
        repeated_events_list.append(batch.num_repeated_events)

    logger.info(f'Batches processed: {len(edge_events_list)}')
    if edge_events_list:
        idx = 1880
        logger.info(f'Example num_edge_events: {edge_events_list[idx]}')
        logger.info(f'Example num_node_events: {node_events_list[idx]}')
        logger.info(f'Example num_timestamps: {timestamps_list[idx]}')
        logger.info(f'Example num_unique_nodes: {unique_nodes_list[idx]}')
        logger.info(f'Example avg_degree: {avg_degree_list[idx]}')
        logger.info(f'Example num_repeated_events: {repeated_events_list[idx]}')

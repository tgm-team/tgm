import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager
from tgm.hooks.node_analytics import NodeAnalyticsHook
from tgm.util.logging import enable_logging, log_latency, log_metrics_dict
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Node Analytics Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--num-tracked', type=int, default=1, help='Number of nodes to track'
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)
logger = logging.getLogger('tgm').getChild(Path(__file__).stem)

seed_everything(args.seed)

# Load the dynamic graph
dg = DGraph(DGData.from_tgb(args.dataset))
logger.info(f'Loaded dataset: {args.dataset}')
logger.info(f'Number of nodes: {dg.num_nodes}')
logger.info(f'Number of edges: {dg.num_edges}')


# Select top N most frequent nodes to track
def get_most_frequent_nodes(dg: DGraph, n: int) -> torch.Tensor:
    """Get the n most frequently appearing nodes in the graph."""
    src, dst, _ = dg.edges
    edge_nodes = torch.cat([src, dst], dim=0)
    unique_nodes, counts = torch.unique(edge_nodes, return_counts=True)

    # Sort by frequency
    sorted_indices = torch.argsort(counts, descending=False)
    top_n_nodes = unique_nodes[sorted_indices[:n]]

    return top_n_nodes


tracked_nodes = get_most_frequent_nodes(dg, args.num_tracked)
logger.info(
    f'Tracking {len(tracked_nodes)} most frequent nodes: {tracked_nodes.tolist()}'
)


@log_latency
def run_node_analytics(
    dg: DGraph, bsize: int, tracked_nodes: torch.Tensor
) -> tuple[dict, list]:
    """Run node analytics hook and collect statistics."""
    node_analytics_hook = NodeAnalyticsHook(
        tracked_nodes=tracked_nodes, num_nodes=dg.num_nodes
    )

    hm = HookManager(keys=['node_analytics'])
    hm.register('node_analytics', node_analytics_hook)
    hm.set_active_hooks('node_analytics')

    loader = DGDataLoader(dg, bsize, hook_manager=hm)

    # Collect statistics over time
    batch_stats = []
    node_novelty_history = []
    edge_novelty_history = []
    edge_density_history = []

    for batch_idx, batch in enumerate(tqdm(loader, desc='Processing batches')):
        # Store node stats for this batch
        batch_stats.append(
            {
                'batch_idx': batch_idx,
                'node_stats': dict(batch.node_stats),
                'edge_stats': dict(batch.edge_stats),
            }
        )
        node_novelty_history.append(batch.node_macro_stats['node_novelty'])

        edge_novelty_history.append(batch.edge_stats['edge_novelty'])
        edge_density_history.append(batch.edge_stats['edge_density'])

    # Compute Edge statistics
    edge_stats = {
        'avg_edge_novelty': np.mean(edge_novelty_history),
        'avg_edge_density': np.mean(edge_density_history),
        'total_batches': len(batch_stats),
    }

    node_macro_stats = {
        'avg_node_novelty': np.mean(node_novelty_history),
    }

    return edge_stats, node_macro_stats, batch_stats


# Run the analytics
edge_stats, node_macro_stats, batch_stats = run_node_analytics(
    dg, args.bsize, tracked_nodes
)

# Log aggregate statistics
logger.info(f'\n\nEdge Statistics (bsize={args.bsize}, tracked={args.num_tracked}):')
log_metrics_dict(metrics_dict=edge_stats, logger=logger)
logger.info(
    f'\nNode Macro Statistics (bsize={args.bsize}, tracked={args.num_tracked}):'
)
log_metrics_dict(metrics_dict=node_macro_stats, logger=logger)

# Log detailed statistics for each tracked node (final batch)
logger.info('\n\nFinal Node Statistics:')
final_batch = batch_stats[-1]
for node_id, stats in sorted(final_batch['node_stats'].items()):
    logger.info(f'\nNode {node_id}:')
    log_metrics_dict(metrics_dict=stats, logger=logger)

# Log statistics evolution for the first tracked node
first_tracked_node = int(tracked_nodes[0].item())
logger.info(f'\n\nStatistics Evolution for Node {first_tracked_node}:')
logger.info(
    'Batch | Activity | Degree | New_Neighbors | Lifetime | Time_Since_Last_Seen'
)
logger.info('-' * 90)

for batch_info in batch_stats[:: max(1, len(batch_stats) // 20)]:  # Sample 20 batches
    if first_tracked_node in batch_info['node_stats']:
        stats = batch_info['node_stats'][first_tracked_node]
        logger.info(
            f'{batch_info["batch_idx"]:5d} | '
            f'{stats["activity"]:8.4f} | '
            f'{stats["degree"]:6d} | '
            f'{stats["new_neighbors"]:13d} | '
            f'{stats["lifetime"]:8.2f} | '
            f'{stats["time_since_last_seen"]:20.2f}'
        )

logger.info('\nDone!')

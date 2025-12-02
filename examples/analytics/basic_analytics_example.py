import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager
from tgm.hooks.basic_analytics import BasicBatchAnalyticsHook
from tgm.util.logging import enable_logging, log_latency, log_metrics_dict
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Basic Analytics Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbn-trade', help='Dataset name')
parser.add_argument('--bsize', type=int, nargs='+', default=[200], help='batch sizes')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)
logger = logging.getLogger('tgm').getChild(Path(__file__).stem)

seed_everything(args.seed)

dg = DGraph(DGData.from_tgb(args.dataset))


@log_latency
def run_basic_analytics(dg: DGraph, bsize: int) -> tuple[dict, int, list]:
    analytics_hook = BasicBatchAnalyticsHook()
    hm = HookManager(keys=['basic'])
    hm.register('basic', analytics_hook)
    hm.set_active_hooks('basic')
    loader = DGDataLoader(dg, bsize, hook_manager=hm)

    # Collect batch statistics
    num_edge_events = []
    num_node_events = []
    num_unique_timestamps = []
    num_unique_nodes = []
    avg_degree = []
    num_repeated_edge_events = []
    num_repeated_node_events = []

    for batch in tqdm(loader):
        num_edge_events.append(batch.num_edge_events)
        num_node_events.append(batch.num_node_events)
        num_unique_timestamps.append(batch.num_unique_timestamps)
        num_unique_nodes.append(batch.num_unique_nodes)
        avg_degree.append(batch.avg_degree)
        num_repeated_edge_events.append(batch.num_repeated_edge_events)
        num_repeated_node_events.append(batch.num_repeated_node_events)

    # Prepare mean metrics
    metrics = {
        'num_edge_events': np.mean(num_edge_events),
        'num_node_events': np.mean(num_node_events),
        'num_unique_timestamps': np.mean(num_unique_timestamps),
        'avg_degree': np.mean(avg_degree),
        'num_unique_nodes': np.mean(num_unique_nodes),
        'num_repeated_edge_events': np.mean(num_repeated_edge_events),
        'num_repeated_node_events': np.mean(num_repeated_node_events),
    }

    return metrics, bsize, num_edge_events


for bsize in args.bsize:
    metrics, bsize, num_edge_events = run_basic_analytics(dg, bsize)
    logger.info(f'Mean stats across {len(num_edge_events)} batches (bsize={bsize}):')
    log_metrics_dict(metrics_dict=metrics, logger=logger)

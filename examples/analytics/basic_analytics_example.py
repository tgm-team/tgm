import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager
from tgm.hooks.basic_analytics import BasicBatchAnalyticsHook
from tgm.util.logging import enable_logging
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

for bsize in args.bsize:
    logger.info(f'===== Running BasicBatchAnalyticsHook with batch size {bsize} =====')

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

    for batch in tqdm(loader, desc=f'Computing stats (bsize={bsize})'):
        num_edge_events.append(batch.num_edge_events)
        num_node_events.append(batch.num_node_events)
        num_unique_timestamps.append(batch.num_unique_timestamps)
        num_unique_nodes.append(batch.num_unique_nodes)
        avg_degree.append(batch.avg_degree)
        num_repeated_edge_events.append(batch.num_repeated_edge_events)
        num_repeated_node_events.append(batch.num_repeated_node_events)

    logger.info(f'Batch size {bsize}: processed {len(num_edge_events)} batches')

    batch_idx = 0
    threshold = 10
    for b_idx in range(len(num_edge_events)):
        # take the first batch that have more than 0 num_node_events
        if num_node_events[b_idx] > threshold and num_edge_events[b_idx] > threshold:
            logger.info(f'First batch with > {threshold} node and edge events: {b_idx}')
            batch_idx = b_idx
            break

    # Logging results
    logger.info(f'Example stats for batch {batch_idx}:')
    logger.info(f'  num_edge_events: {num_edge_events[batch_idx]}')
    logger.info(f'  num_node_events: {num_node_events[batch_idx]}')
    logger.info(f'  num_unique_timestamps: {num_unique_timestamps[batch_idx]}')
    logger.info(f'  avg_degree: {avg_degree[batch_idx]}')
    logger.info(f'  unique nodes: {num_unique_nodes[batch_idx]}')
    logger.info(f'  repeated edges: {num_repeated_edge_events[batch_idx]}')
    logger.info(f'  repeated nodes: {num_repeated_node_events[batch_idx]}')

logger.info('Finished computing analytics for all batch sizes.')

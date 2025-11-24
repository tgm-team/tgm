import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, StatelessHook
from tgm.util.logging import enable_logging
from tgm.util.seed import seed_everything

# Parser and logger setup
parser = argparse.ArgumentParser(
    description='Standard Analytics Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)
logger = logging.getLogger('tgm').getChild(Path(__file__).stem)


class StatHook(StatelessHook):
    """Computes a placeholder statistic for demonstration purposes."""

    produces = {'stat_name'}

    def __init__(self, place_holder: str = 'example_value') -> None:
        self._place_holder = place_holder

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.stat_name = self._place_holder  # Placeholder for actual computation
        return batch


# Main execution starts here
seed_everything(args.seed)

dg = DGraph(DGData.from_tgb(args.dataset))

# register hooks
stat_name_hook = StatHook(place_holder='example_value')  # Example placeholder value
hm = HookManager(keys=['stat_name'])
hm.register('stat_name', stat_name_hook)
hm.set_active_hooks('stat_name')

# Apply hook to the DGraph
loader = DGDataLoader(dg, args.bsize, hook_manager=hm)

# Collect batch statistics
stat_name_list = []
for batch in tqdm(loader, desc='Computing stat_name'):
    stat_name_list.append(batch.stat_name)

logger.info(f'Computed {len(stat_name_list)} stat_name estimates')
logger.info(f'Example stat_name estimate: {stat_name_list[0]}')

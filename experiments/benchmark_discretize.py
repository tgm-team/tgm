import time
from pathlib import Path
import json

from tgm import DGData

ROOT_DIR = Path(__file__).resolve().parents[1]
SAVE_PATH = ROOT_DIR / 'experiments' / 'results' / 'benchmark_discretize.json'
NUM_TRAILS = 3
DATASETS = ['tgbl-enron', 'tgbl-lastfm', 'tgbl-subreddit', 'tgbl-uci', 'tgbl-wiki']


def main() -> None:
    results = {}

    for dataset in DATASETS:
        times = []

        for _ in range(NUM_TRAILS):
            data = DGData.from_tgb(dataset)

            start = time.perf_counter()
            data = data.discretize('h')
            end = time.perf_counter()
            times.append(end - start)

        results[dataset] = {'times': times, 'avg_time': sum(times) / NUM_TRAILS}
        print(f'Discretized {dataset} in {results[dataset]["avg_time"]:.2f}s')

    with open(SAVE_PATH, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()

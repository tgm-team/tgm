import time
import numpy as np

from tgm import DGData

DATASETS = [
    'tgbl-enron',
    'tgbl-lastfm',
    'tgbl-subreddit',
    'tgbl-uci',
    'tgbl-wiki',
]

RUNS = 3

for dataset in DATASETS:
    data = DGData.from_tgb(dataset)

    start = time.perf_counter()
    data = data.discretize('h')
    end = time.perf_counter()
    latency = end - start
    print(f'Discretized {dataset} in {latency:.2f}s')

import numpy as np

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGDataLoader
from opendg.nn.memory import EdgeBankPredictor
from opendg.util.perf import Profiling, Usage, compare_usage


def edge_bank(n=100):
    src, dst, t = (
        np.ones(n, dtype=int),
        np.ones(n, dtype=int),
        np.ones(n, dtype=int),
    )
    predictor = EdgeBankPredictor(src, dst, t)
    for i in range(10000):
        predictor.update_memory(i * src, i * dst, i * t)
        predictor.predict_link(i * src, i * dst)


def dgraph_iter(n=100000):
    events = [EdgeEvent(src=i, dst=i, t=i) for i in range(n)]
    dg = DGraph(events)
    loader = DGDataLoader(dg)
    for batch in loader:
        continue


if __name__ == '__main__':
    with Usage('DGraph Iteration'):
        dgraph_iter()

    with Profiling(frac=0.3):
        dgraph_iter()

    f1 = {'func': edge_bank, 'args': [1]}
    f2 = {'func': dgraph_iter, 'args': [500]}
    compare_usage(f1, f2, 'Mock Slow Comparison')
    compare_usage(f1, f1, 'Bar Comparison')
    compare_usage(f2, f1, 'Mock Fast Comparison')

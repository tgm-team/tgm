import pytest

from tgm.graph import DGData, DGraph

from .conftest import DATASETS


@pytest.mark.benchmark(group='graph_discretization')
@pytest.mark.parametrize(
    # We want to measure discretization from seconds, tgbn-trade is already in Years
    'dataset',
    [d for d in DATASETS if d.values[0] != 'tgbn-trade'],
)
@pytest.mark.parametrize('granularity', ['D', 'M', 'Y'])  # daily, monthly, yearly
def test_graph_discretization(benchmark, dataset, granularity):
    data = DGData.from_tgb(dataset)
    dg = DGraph(data)

    benchmark(lambda: data.discretize(granularity))

    num_events = dg.num_events
    throughput = (num_events / benchmark.stats['mean']) / 1e6
    benchmark.extra_info['throughput_events_per_sec'] = throughput
    print(f'{dataset} discretiation ({granularity}): {throughput:.2f} M events/sec')

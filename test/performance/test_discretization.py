import pytest

from .conftest import DATASETS


@pytest.mark.benchmark(group='graph_discretization')
@pytest.mark.parametrize(
    # We want to measure discretization from seconds, tgbn-trade is already in Years
    'dataset',
    [d for d in DATASETS if d.values[0] != 'tgbn-trade'],
)
@pytest.mark.parametrize('granularity', ['D', 'M', 'Y'])  # daily, monthly, yearly
def test_graph_discretization(benchmark, dataset, granularity, preloaded_graphs):
    data = preloaded_graphs[dataset]['data']
    dg = preloaded_graphs[dataset]['dg']

    benchmark(lambda: data.discretize(granularity))

    num_events = dg.num_events
    throughput = (num_events / benchmark.stats['mean']) / 1e6
    benchmark.extra_info['throughput_M_events_per_sec'] = throughput
    print(f'{dataset} discretiation ({granularity}): {throughput:.3f} M events/sec')

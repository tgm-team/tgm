import pytest

from tgm.graph import DGData, DGraph

from .conftest import DATASETS


@pytest.mark.benchmark(group='graph_loading')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_loading(benchmark, dataset):
    data = DGData.from_tgb(dataset)
    out = {}

    def run():
        out['dg'] = DGraph(data)

    benchmark(run)
    num_events = out['dg'].num_events
    throughput = (num_events / benchmark.stats['mean']) / 1e6
    benchmark.extra_info['throughput_events_per_sec'] = throughput
    print(f'{dataset} loading throughput: {throughput:.2f} M events/sec')


@pytest.mark.benchmark(group='graph_split')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_split(benchmark, dataset):
    data = DGData.from_tgb(dataset)
    dg = DGraph(data)

    benchmark(lambda: data.split())

    num_events = dg.num_events
    throughput = (num_events / benchmark.stats['mean']) / 1e6
    benchmark.extra_info['throughput_events_per_sec'] = throughput
    print(f'{dataset} split throughput: {throughput:.2f} M events/sec')

import pytest

from tgm.graph import DGData, DGraph

from .conftest import DATASETS


@pytest.mark.benchmark(group='graph_loading')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_loading(benchmark, dataset):
    out = {}

    def run():
        data = DGData.from_tgb(dataset)
        out['dg'] = DGraph(data)

    benchmark(run)
    num_events = out['dg'].num_events
    throughput = (num_events / benchmark.stats['mean']) / 1e6
    benchmark.extra_info.update(
        {
            'throughput_M_events_per_sec': throughput,
            'num_events': out['dg'].num_events,
        }
    )
    print(f'{dataset} loading throughput: {throughput:.2f} M events/sec')


@pytest.mark.benchmark(group='graph_split')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_split(benchmark, dataset, preloaded_graphs):
    data = preloaded_graphs[dataset]['data']
    dg = preloaded_graphs[dataset]['dg']

    benchmark(lambda: data.split())

    num_events = dg.num_events
    throughput = (num_events / benchmark.stats['mean']) / 1e6
    benchmark.extra_info.update(
        {
            'throughput_M_events_per_sec': throughput,
            'num_events': dg.num_events,
        }
    )
    print(f'{dataset} split throughput: {throughput:.2f} M events/sec')

import pytest

from tgm.graph import DGData, DGraph

from .conftest import DATASETS


@pytest.mark.benchmark(group='graph_loading')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_loading(benchmark, dataset):
    benchmark(lambda: DGraph(DGData.from_tgb(dataset)))


@pytest.mark.benchmark(group='graph_split')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_split(benchmark, dataset):
    data = DGData.from_tgb(dataset)
    benchmark(lambda: data.split())

import pytest

from tgm.graph import DGraph

from .conftest import DATASETS


@pytest.mark.benchmark(group='graph_loading')
@pytest.mark.parametrize('dataset', DATASETS)
def test_graph_loading(benchmark, dataset):
    benchmark(lambda: DGraph(dataset))

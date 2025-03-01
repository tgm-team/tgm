import pytest
import torch

from opendg.events import EdgeEvent
from opendg.graph import DGraph
from opendg.loader import DGDataLoader


def test_init_default_args():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
    ]
    dg = DGraph(events)
    loader = DGDataLoader(dg)
    assert loader._batch_size == 1
    assert loader._batch_unit == 'r'


@pytest.mark.parametrize('batch_unit', ['Y', 'M', 'W', 'D', 'h', 's', 'ms', 'us', 'ns'])
def test_init_ordered_dg_non_ordered_batch(batch_unit):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
    ]
    dg = DGraph(events)
    with pytest.raises(ValueError):
        _ = DGDataLoader(dg, batch_unit=batch_unit)


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_ordered_dg_ordered_batch(batch_size, drop_last):
    pass


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch(batch_size, drop_last):
    pass


@pytest.mark.skip(reason='Not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_non_ordered_batch_with_conversion(
    batch_size, drop_last
):
    pass


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch(batch_size, drop_last):
    pass


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch_batch_split_required(
    batch_size, drop_last
):
    pass


@pytest.mark.skip(reason='not implemented')
@pytest.mark.parametrize('batch_size', [1, 3, 10])
@pytest.mark.parametrize('drop_last', [True, False])
def test_iteration_non_ordered_dg_ordered_batch_multi_batch_required(
    batch_size, drop_last
):
    pass

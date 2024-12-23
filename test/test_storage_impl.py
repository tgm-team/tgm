import pytest

from opendg._storage import DGStorageImplementations


@pytest.fixture(params=DGStorageImplementations)
def DGStorageImpl(request):
    return request.param


def test_init(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    assert storage.to_events() == [(1, 2, 3), (5, 10, 20)]
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 4
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2


def test_init_from_events(DGStorageImpl):
    events = [(1, 2, 3), (5, 10, 20)]
    storage = DGStorageImpl.from_events(events)
    assert storage.to_events() == events
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 4
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2

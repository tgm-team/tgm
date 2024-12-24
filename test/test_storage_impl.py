import pytest

from opendg._storage import (
    DGStorageBackends,
    DGStorageDictBackend,
    get_dg_storage_backend,
    set_dg_storage_backend,
)


@pytest.fixture(params=DGStorageBackends.values())
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


def test_slice_time(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    storage = storage.slice_time(1, 2)
    assert storage.to_events() == [(1, 2, 3)]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 2
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1


def test_slice_time_bad_slice(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    with pytest.raises(ValueError):
        storage.slice_time(2, 1)


def test_slice_nodes(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    storage = storage.slice_nodes([1, 2])
    assert storage.to_events() == [(1, 2, 3)]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 2
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1


def test_slice_nodes_empty_slice(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    storage = storage.slice_nodes([])
    assert storage.to_events() == []
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0


def test_get_nbrs(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    nbrs = storage.get_nbrs([0, 2, 20])
    assert nbrs == {2: [(3, 1)], 20: [(10, 5)]}


def test_get_nbrs_empty_nbrs(DGStorageImpl):
    events_dict = {1: (2, 3), 5: (10, 20)}
    storage = DGStorageImpl(events_dict)
    nbrs = storage.get_nbrs([0])
    assert nbrs == {}


def test_update_single_event(DGStorageImpl):
    events = [(1, 2, 3)]
    storage = DGStorageImpl.from_events(events)
    storage = storage.update((5, 10, 20))
    assert storage.to_events() == [(1, 2, 3), (5, 10, 20)]


def test_update_multiple_events(DGStorageImpl):
    events = []
    storage = DGStorageImpl.from_events(events)
    storage = storage.update([(1, 2, 3), (5, 10, 20)])
    assert storage.to_events() == [(1, 2, 3), (5, 10, 20)]


def test_get_dg_storage_backend():
    backend = get_dg_storage_backend()
    print(backend)


def test_set_dg_storage_backend_with_class():
    set_dg_storage_backend(DGStorageDictBackend)
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_str():
    set_dg_storage_backend('DictionaryBackend')
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_bad_str():
    with pytest.raises(ValueError):
        set_dg_storage_backend('foo')

import pytest
import torch

from opendg._storage import (
    DGStorageBackends,
    get_dg_storage_backend,
    set_dg_storage_backend,
)
from opendg._storage.backends import DGStorageDictBackend
from opendg.events import EdgeEvent, NodeEvent
from opendg.timedelta import TimeDeltaDG


@pytest.fixture(params=DGStorageBackends.values())
def DGStorageImpl(request):
    return request.param


def test_init(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    assert storage.to_events() == events
    assert storage.start_time == 1
    assert storage.end_time == 6
    assert storage.num_nodes == 21
    assert storage.num_edges == 2
    assert storage.num_timestamps == 3
    assert storage.time_delta == td
    assert len(storage) == 3

    expected_node_feats = torch.zeros(6 + 1, 20 + 1, 3, 6)
    expected_node_feats[6, 7] = events[-1].features
    assert torch.equal(storage.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(6 + 1, 20 + 1, 20 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    expected_edge_feats[5, 10, 20] = events[1].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_init_multiple_events_per_timestamp(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(10, 20), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    assert storage.to_events() == events
    assert storage.start_time == 1
    assert storage.end_time == 6
    assert storage.num_nodes == 21
    assert storage.num_edges == 3
    assert storage.num_timestamps == 3
    assert storage.time_delta == td
    assert len(storage) == 3

    expected_node_feats = torch.zeros(6 + 1, 20 + 1, 3, 6)
    expected_node_feats[6, 7] = events[-1].features
    assert torch.equal(storage.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(6 + 1, 20 + 1, 20 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    expected_edge_feats[1, 10, 20] = events[1].features
    expected_edge_feats[5, 10, 20] = events[2].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_init_empty_features(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=5, edge=(10, 20)),
        NodeEvent(time=6, node_id=7),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    assert storage.to_events() == events
    assert storage.start_time == 1
    assert storage.end_time == 6
    assert storage.num_nodes == 21
    assert storage.num_edges == 2
    assert storage.num_timestamps == 3
    assert storage.time_delta == td
    assert len(storage) == 3

    assert storage.node_feats is None
    assert storage.edge_feats is None


def test_init_incompatible_node_feature_dimension(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(3, 6)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]

    td = 'Mock'
    with pytest.raises(ValueError):
        _ = DGStorageImpl(events, td)


def test_init_incompatible_edge_feature_dimension(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        NodeEvent(time=5, node_id=10, features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]

    td = 'Mock'
    with pytest.raises(ValueError):
        _ = DGStorageImpl(events, td)


def test_init_empty(DGStorageImpl):
    events = []
    td = TimeDeltaDG('r')
    storage = DGStorageImpl(events, td)
    assert storage.to_events() == []
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0
    assert storage.time_delta == td
    assert len(storage) == 0

    assert storage.node_feats is None
    assert storage.edge_feats is None


def test_slice_time(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    storage = storage.slice_time(1, 2)
    assert storage.to_events() == [events[0]]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 4
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_delta == td
    assert len(storage) == 1

    assert storage.node_feats is None

    expected_edge_feats = torch.zeros(1 + 1, 3 + 1, 3 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_empty_slice(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    storage = storage.slice_time(2, 3)
    assert storage.to_events() == []
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0
    assert storage.time_delta == td
    assert len(storage) == 0

    assert storage.node_feats is None
    assert storage.edge_feats is None


def test_slice_time_full_slice(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    storage = storage.slice_time(0, 7)
    assert storage.to_events() == events
    assert storage.start_time == 1
    assert storage.end_time == 6
    assert storage.num_nodes == 21
    assert storage.num_edges == 2
    assert storage.num_timestamps == 3
    assert storage.time_delta == td
    assert len(storage) == 3

    expected_node_feats = torch.zeros(6 + 1, 20 + 1, 3, 6)
    expected_node_feats[6, 7] = events[-1].features
    assert torch.equal(storage.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(6 + 1, 20 + 1, 20 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    expected_edge_feats[5, 10, 20] = events[1].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_on_boundary(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    storage = storage.slice_time(1, 5)
    assert storage.to_events() == [events[0]]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 4
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_delta == td
    assert len(storage) == 1

    assert storage.node_feats is None

    expected_edge_feats = torch.zeros(1 + 1, 3 + 1, 3 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_slice_time_bad_slice(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    with pytest.raises(ValueError):
        storage.slice_time(2, 1)


def test_slice_nodes(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    storage = storage.slice_nodes([1, 2])
    assert storage.to_events() == [events[0]]
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 4
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_delta == td
    assert len(storage) == 1

    assert storage.node_feats is None

    expected_edge_feats = torch.zeros(1 + 1, 3 + 1, 3 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_slice_nodes_empty_slice(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    td = TimeDeltaDG('Y', 1)
    storage = DGStorageImpl(events, td)
    storage = storage.slice_nodes([])
    assert storage.to_events() == []
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0
    assert storage.time_delta == td
    assert len(storage) == 0

    assert storage.node_feats is None
    assert storage.edge_feats is None


def test_append_single_event(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
    ]
    td = TimeDeltaDG('r')
    storage = DGStorageImpl(events, td)
    assert storage.start_time == 1
    assert storage.end_time == 1
    assert storage.num_nodes == 4
    assert storage.num_edges == 1
    assert storage.num_timestamps == 1
    assert storage.time_delta == td
    assert len(storage) == 1

    assert storage.node_feats is None

    expected_edge_feats = torch.zeros(1 + 1, 3 + 1, 3 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)

    new_event = EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5))
    storage = storage.append(new_event)
    assert storage.to_events() == events + [new_event]
    assert storage.start_time == 1
    assert storage.end_time == 5
    assert storage.num_nodes == 21
    assert storage.num_edges == 2
    assert storage.num_timestamps == 2
    # assert storage.time_delta == 4 # TODO
    assert len(storage) == 2

    assert storage.node_feats is None

    expected_edge_feats = torch.zeros(5 + 1, 20 + 1, 20 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = events[0].features
    expected_edge_feats[5, 10, 20] = new_event.features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_append_multiple_events(DGStorageImpl):
    events = []
    td = TimeDeltaDG('r')
    storage = DGStorageImpl(events, td)
    assert storage.start_time == None
    assert storage.end_time == None
    assert storage.num_nodes == 0
    assert storage.num_edges == 0
    assert storage.num_timestamps == 0
    assert storage.time_delta == td
    assert len(storage) == 0

    new_events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(2, 5)),
        NodeEvent(time=6, node_id=7, features=torch.rand(3, 6)),
    ]
    storage = storage.append(new_events)
    assert storage.to_events() == events + new_events
    assert storage.start_time == 1
    assert storage.end_time == 6
    assert storage.num_nodes == 21
    assert storage.num_edges == 2
    assert storage.num_timestamps == 3
    assert storage.time_delta == td
    assert len(storage) == 3

    expected_node_feats = torch.zeros(6 + 1, 20 + 1, 3, 6)
    expected_node_feats[6, 7] = new_events[-1].features
    assert torch.equal(storage.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(6 + 1, 20 + 1, 20 + 1, 2, 5)
    expected_edge_feats[1, 2, 3] = new_events[0].features
    expected_edge_feats[5, 10, 20] = new_events[1].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


def test_append_incompatible_node_feature_dimension(DGStorageImpl):
    events = [
        NodeEvent(time=1, node_id=1, features=torch.rand(2, 5)),
    ]
    td = TimeDeltaDG('r')
    storage = DGStorageImpl(events, td)

    new_event = NodeEvent(time=5, node_id=10, features=torch.rand(3, 6))
    with pytest.raises(ValueError):
        _ = storage.append(new_event)


def test_append_incompatible_edge_feature_dimension(DGStorageImpl):
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
    ]
    td = TimeDeltaDG('r')
    storage = DGStorageImpl(events, td)

    new_event = EdgeEvent(time=5, edge=(10, 20), features=torch.rand(3, 6))
    with pytest.raises(ValueError):
        _ = storage.append(new_event)


def test_append_different_feature_dimension_after_slicing_to_empty(DGStorageImpl):
    events = [
        NodeEvent(time=1, node_id=1, features=torch.rand(2, 5)),
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(2, 5)),
    ]
    td = TimeDeltaDG('r')
    storage = DGStorageImpl(events, td)

    storage = storage.slice_nodes([])
    assert len(storage) == 0
    assert storage.node_feats is None
    assert storage.edge_feats is None

    new_events = [
        NodeEvent(time=5, node_id=10, features=torch.rand(3, 6)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(3, 6)),
    ]
    storage = storage.append(new_events)
    assert len(storage) == 1

    expected_node_feats = torch.zeros(5 + 1, 20 + 1, 3, 6)
    expected_node_feats[5, 10] = new_events[0].features

    assert torch.equal(storage.node_feats.to_dense(), expected_node_feats)

    expected_edge_feats = torch.zeros(5 + 1, 20 + 1, 20 + 1, 3, 6)
    expected_edge_feats[5, 10, 20] = new_events[1].features
    assert torch.equal(storage.edge_feats.to_dense(), expected_edge_feats)


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_single_event_graph(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_causes_time_gap(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_bad_time_delta(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_bad_agg_func(DGStorageImpl):
    pass


@pytest.mark.skip(reason='Not implemented')
def test_temporal_coarsening_empty_graph(DGStorageImpl):
    pass


def test_get_dg_storage_backend():
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_class():
    set_dg_storage_backend(DGStorageDictBackend)
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_str():
    set_dg_storage_backend('DictionaryBackend')
    assert get_dg_storage_backend() == DGStorageDictBackend


def test_set_dg_storage_backend_with_bad_str():
    with pytest.raises(ValueError):
        set_dg_storage_backend('foo')

import torch

from opendg.events import EdgeEvent, NodeEvent


def test_node_event_without_node_feat():
    time = 1337
    node_id = 0
    event = NodeEvent(time, node_id)
    assert event.time == time
    assert event.node_id == node_id
    assert event.features is None


def test_node_event_with_node_feat():
    time = 1337
    node_id = 0
    node_features = torch.rand(1, 3, 7)
    event = NodeEvent(time, node_id, node_features)
    assert event.time == time
    assert event.node_id == node_id
    assert torch.equal(event.features, node_features)


def test_node_event_ordering_same_node_id():
    e1 = NodeEvent(time=1337, node_id=0)
    e2 = NodeEvent(time=1338, node_id=0)
    assert e1 < e2
    assert e1 <= e2
    assert e1 != e2
    assert e1 == e1
    assert e2 >= e1
    assert e2 > e1


def test_node_event_ordering_different_node_id():
    e1 = NodeEvent(time=1337, node_id=0)
    e2 = NodeEvent(time=1338, node_id=1)
    assert e1 < e2
    assert e1 <= e2
    assert e1 != e2
    assert e1 == e1
    assert e2 >= e1
    assert e2 > e1


def test_edge_event_without_node_feat():
    time = 1337
    edge = (0, 1)
    event = EdgeEvent(time, edge)
    assert event.time == time
    assert event.edge == edge
    assert event.features is None


def test_edge_event_with_node_feat():
    time = 1337
    edge = (0, 1)
    edge_features = torch.rand(1, 3, 7)
    event = EdgeEvent(time, edge, edge_features)
    assert event.time == time
    assert event.edge == edge
    assert torch.equal(event.features, edge_features)


def test_edge_event_ordering_same_edge():
    e1 = EdgeEvent(time=1337, edge=(0, 1))
    e2 = EdgeEvent(time=1338, edge=(0, 1))
    assert e1 < e2
    assert e1 <= e2
    assert e1 != e2
    assert e1 == e1
    assert e2 >= e1
    assert e2 > e1


def test_edge_event_ordering_different_node_id():
    e1 = EdgeEvent(time=1337, edge=(0, 1))
    e2 = EdgeEvent(time=1338, edge=(1, 2))
    assert e1 < e2
    assert e1 <= e2
    assert e1 != e2
    assert e1 == e1
    assert e2 >= e1
    assert e2 > e1


def test_event_ordering_between_events():
    e1 = NodeEvent(time=1337, node_id=0)
    e2 = EdgeEvent(time=1338, edge=(1, 2))
    assert e1 < e2
    assert e1 <= e2
    assert e1 != e2
    assert e1 == e1
    assert e2 >= e1
    assert e2 > e1

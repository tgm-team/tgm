import torch

from opendg.events import EdgeEvent, NodeEvent


def test_node_event_without_feat():
    time = 1337
    src = 0
    global_idx = 3
    event = NodeEvent(time, src, global_idx)
    assert event.t == time
    assert event.src == src
    assert event.global_idx == global_idx
    assert event.features is None


def test_node_event_with_feat():
    time = 1337
    src = 0
    global_idx = 3
    features = torch.rand(1, 3, 7)
    event = NodeEvent(time, src, global_idx, features=features)
    assert event.t == time
    assert event.src == src
    assert event.global_idx == global_idx
    assert torch.equal(event.features, features)


def test_edge_event_without_feat():
    time = 1337
    src = 0
    dst = 1
    global_idx = 3
    event = EdgeEvent(time, src, dst, global_idx)
    assert event.t == time
    assert event.edge == (src, dst)
    assert event.global_idx == global_idx
    assert event.features is None


def test_edge_event_with_feat():
    time = 1337
    src = 0
    dst = 1
    features = torch.rand(1, 3, 7)
    event = EdgeEvent(time, src, dst, features=features)
    assert event.t == time
    assert event.edge == (src, dst)
    assert torch.equal(event.features, features)

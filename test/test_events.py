import torch

from opendg.events import EdgeEvent, NodeEvent


def test_node_event_without_node_feat():
    time = 1337
    src = 0
    event = NodeEvent(time, src)
    assert event.t == time
    assert event.src == src
    assert event.msg is None


def test_node_event_with_node_feat():
    time = 1337
    src = 0
    msg = torch.rand(1, 3, 7)
    event = NodeEvent(time, src, msg)
    assert event.t == time
    assert event.src == src
    assert torch.equal(event.msg, msg)


def test_edge_event_without_node_feat():
    time = 1337
    src = 0
    dst = 1
    event = EdgeEvent(time, src, dst)
    assert event.t == time
    assert event.edge == (src, dst)
    assert event.msg is None


def test_edge_event_with_node_feat():
    time = 1337
    src = 0
    dst = 1
    msg = torch.rand(1, 3, 7)
    event = EdgeEvent(time, src, dst, msg)
    assert event.t == time
    assert event.edge == (src, dst)
    assert torch.equal(event.msg, msg)

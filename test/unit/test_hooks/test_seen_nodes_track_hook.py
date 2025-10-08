import pytest
import torch

from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import EdgeEventsSeenNodesTrackHook
from tgm.hooks.hook_manager import HookManager


@pytest.fixture
def dg():
    edge_index = torch.IntTensor([[2, 3], [2, 5]])
    edge_timestamps = torch.LongTensor([1, 5])
    node_timestamps = torch.LongTensor([2, 3, 5, 5, 5])
    node_ids = torch.IntTensor([4, 2, 5, 1, 2])

    # Can't actually get node events without dynamic node feats
    dynamic_node_feats = torch.rand(5, 3)
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
        dynamic_node_feats=dynamic_node_feats,
        time_delta='s',
    )

    dg = DGraph(data)
    return dg


def test_seen_nodes_track_hook(dg):
    hm = HookManager(keys=['unit'])
    hm.register('unit', EdgeEventsSeenNodesTrackHook(6))
    hm.set_active_hooks('unit')

    loader = DGDataLoader(dg, batch_unit='s', hook_manager=hm)
    batch_iter = iter(loader)
    batch_1 = next(batch_iter)
    assert len(batch_1.seen_nodes) == 0

    batch_2 = next(batch_iter)
    assert len(batch_2.seen_nodes) == 0

    batch_3 = next(batch_iter)
    assert len(batch_3.seen_nodes) == 1
    assert batch_3.seen_nodes[0].item() == 2

    batch_4 = next(batch_iter)
    assert len(batch_4.seen_nodes) == 2
    assert batch_4.seen_nodes[0].item() == 5
    assert batch_4.seen_nodes[1].item() == 2
    assert torch.equal(batch_4.batch_nodes_mask, torch.Tensor([0, 2]))


def test_seen_nodes_track_hook_bad_init():
    with pytest.raises(ValueError):
        EdgeEventsSeenNodesTrackHook(-1)

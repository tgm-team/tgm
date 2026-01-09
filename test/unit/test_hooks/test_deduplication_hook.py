import pytest
import torch

from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import DeduplicationHook
from tgm.hooks.hook_manager import HookManager


@pytest.fixture
def dg():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.IntTensor([1, 5, 20])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


def test_hook_dependancies():
    assert DeduplicationHook.requires == set()
    assert DeduplicationHook.produces == {'unique_nids', 'global_to_local'}


def test_hook_reset_state():
    assert DeduplicationHook.has_state == False


def test_dedup(dg):
    hook = DeduplicationHook()
    batch = dg.materialize()
    processed_batch = hook(dg, batch)
    torch.testing.assert_close(
        processed_batch.unique_nids, torch.IntTensor([1, 2, 4, 8])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.src), torch.IntTensor([1, 1, 0])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.dst), torch.IntTensor([1, 2, 3])
    )


def test_dedup_with_negatives(dg):
    hook = DeduplicationHook()
    batch = dg.materialize()
    batch.neg = torch.IntTensor([1, 5, 10])  # add some mock negatives

    processed_batch = hook(dg, batch)
    torch.testing.assert_close(
        processed_batch.unique_nids, torch.IntTensor([1, 2, 4, 5, 8, 10])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.src), torch.IntTensor([1, 1, 0])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.dst), torch.IntTensor([1, 2, 4])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.neg), torch.IntTensor([0, 3, 5])
    )


def test_dedup_with_nbrs(dg):
    hook = DeduplicationHook()
    batch = dg.materialize()
    batch.nbr_nids = [  # add some mock neighbours
        torch.IntTensor([1, 5]),  # First hop
        torch.IntTensor([10]),  # Second hop
    ]
    batch.nbr_mask = [torch.IntTensor([1, 1]), torch.IntTensor([1])]

    processed_batch = hook(dg, batch)
    torch.testing.assert_close(
        processed_batch.unique_nids, torch.IntTensor([1, 2, 4, 5, 8, 10])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.src), torch.IntTensor([1, 1, 0])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.dst), torch.IntTensor([1, 2, 4])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.nbr_nids[0]), torch.IntTensor([0, 3])
    )
    torch.testing.assert_close(
        processed_batch.global_to_local(batch.nbr_nids[1]), torch.IntTensor([5])
    )


@pytest.fixture
def node_only_graph():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    edge_timestamps = torch.IntTensor([1, 2, 3, 7, 8])
    edge_x = torch.IntTensor([[1], [2], [3], [0], [0]])
    node_x = torch.rand(5, 5)
    node_timestamps = torch.IntTensor([4, 5, 6, 7, 8])
    node_ids = torch.IntTensor([4, 5, 6, 5, 6])
    data = DGData.from_raw(
        edge_timestamps,
        edge_index,
        edge_x=edge_x,
        node_x=node_x,
        node_timestamps=node_timestamps,
        node_ids=node_ids,
    )
    return DGraph(data)


def test_dedup_node_only_batch(node_only_graph):
    hm = HookManager(keys=['unit'])
    hm.register('unit', DeduplicationHook())
    loader = DGDataLoader(node_only_graph, batch_size=3, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        torch.testing.assert_close(batch_1.unique_nids, torch.IntTensor([1, 2, 3, 4]))
        torch.testing.assert_close(
            batch_1.global_to_local(batch_1.src), torch.IntTensor([0, 1, 2])
        )
        torch.testing.assert_close(
            batch_1.global_to_local(batch_1.dst), torch.IntTensor([1, 2, 3])
        )

        batch_2 = next(batch_iter)
        torch.testing.assert_close(batch_2.unique_nids, torch.IntTensor([4, 5, 6]))
        torch.testing.assert_close(
            batch_2.global_to_local(batch_2.node_ids), torch.IntTensor([0, 1, 2])
        )

        batch_3 = next(batch_iter)
        torch.testing.assert_close(batch_3.unique_nids, torch.IntTensor([4, 5, 6]))
        torch.testing.assert_close(
            batch_3.global_to_local(batch_3.src), torch.IntTensor([0, 1])
        )
        torch.testing.assert_close(
            batch_3.global_to_local(batch_3.dst), torch.IntTensor([1, 2])
        )
        torch.testing.assert_close(
            batch_3.global_to_local(batch_3.node_ids), torch.IntTensor([1])
        )

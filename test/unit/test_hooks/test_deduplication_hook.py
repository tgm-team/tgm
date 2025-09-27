import pytest
import torch

from tgm import DGraph
from tgm.data import DGData
from tgm.hooks import DeduplicationHook


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

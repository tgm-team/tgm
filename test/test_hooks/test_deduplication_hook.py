import pytest
import torch

from tgm.data import DGData
from tgm.graph import DGraph
from tgm.hooks import DeduplicationHook


@pytest.fixture
def dg():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


def test_hook_dependancies():
    assert DeduplicationHook.requires == set()
    assert DeduplicationHook.produces == {
        'unique_nids',
        'nid_to_idx',
        'src_idx',
        'dst_idx',
        'neg_idx',
        'nbr_nids_idx',
    }


def test_dedup(dg):
    hook = DeduplicationHook()
    batch = dg.materialize()
    processed_batch = hook(dg, batch)
    torch.testing.assert_close(
        processed_batch.unique_nids, torch.LongTensor([1, 2, 4, 8])
    )
    torch.testing.assert_close(
        processed_batch.nid_to_idx, torch.LongTensor([-1, 0, 1, -1, 2, -1, -1, -1, 3])
    )
    torch.testing.assert_close(processed_batch.src_idx, torch.LongTensor([1, 1, 0]))
    torch.testing.assert_close(processed_batch.dst_idx, torch.LongTensor([1, 2, 3]))


def test_dedup_with_negatives(dg):
    hook = DeduplicationHook()
    batch = dg.materialize()
    batch.neg = torch.LongTensor([1, 5, 10])  # add some mock negatives

    processed_batch = hook(dg, batch)
    torch.testing.assert_close(
        processed_batch.unique_nids, torch.LongTensor([1, 2, 4, 5, 8, 10])
    )
    torch.testing.assert_close(
        processed_batch.nid_to_idx,
        torch.LongTensor([-1, 0, 1, -1, 2, 3, -1, -1, 4, -1, 5]),
    )
    torch.testing.assert_close(processed_batch.src_idx, torch.LongTensor([1, 1, 0]))
    torch.testing.assert_close(processed_batch.dst_idx, torch.LongTensor([1, 2, 4]))
    torch.testing.assert_close(processed_batch.neg_idx, torch.LongTensor([0, 3, 5]))


def test_dedup_with_nbrs(dg):
    hook = DeduplicationHook()
    batch = dg.materialize()
    batch.nbr_nids = [  # add some mock neighbours
        torch.LongTensor([1, 5]),  # First hop
        torch.LongTensor([10]),  # Second hop
    ]

    processed_batch = hook(dg, batch)
    torch.testing.assert_close(
        processed_batch.unique_nids, torch.LongTensor([1, 2, 4, 5, 8, 10])
    )
    torch.testing.assert_close(
        processed_batch.nid_to_idx,
        torch.LongTensor([-1, 0, 1, -1, 2, 3, -1, -1, 4, -1, 5]),
    )
    torch.testing.assert_close(processed_batch.src_idx, torch.LongTensor([1, 1, 0]))
    torch.testing.assert_close(processed_batch.dst_idx, torch.LongTensor([1, 2, 4]))
    torch.testing.assert_close(
        processed_batch.nbr_nids_idx[0], torch.LongTensor([0, 3])
    )
    torch.testing.assert_close(processed_batch.nbr_nids_idx[1], torch.LongTensor([5]))

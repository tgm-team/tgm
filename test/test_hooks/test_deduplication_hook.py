import pytest
import torch

from tgm.data import DGData
from tgm.hooks import DeduplicationHook


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[2, 2], [2, 4], [1, 8]])
    edge_timestamps = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_timestamps, edge_index)


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


def test_dedup():
    pass


def test_dedup_with_negatives():
    # dg = DGraph(data)
    # hook = NegativeEdgeSamplerHook(low=0, high=10)
    # batch = hook(dg, dg.materialize())
    pass


def test_dedup_with_nbrs():
    pass

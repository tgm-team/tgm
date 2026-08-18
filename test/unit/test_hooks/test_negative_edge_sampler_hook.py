import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.constants import PADDED_NODE_ID
from tgm.data import DGData, DGDataLoader
from tgm.hooks import (
    HistoricalNegativeEdgeSamplerHook,
    HookManager,
    NodeTypeNegativeSamplerHook,
    RandomNegativeEdgeSamplerHook,
)


@pytest.fixture
def data():
    edge_index = torch.IntTensor([[2, 2], [2, 4], [1, 8]])
    edge_time = torch.LongTensor([1, 5, 20])
    return DGData.from_raw(edge_time, edge_index)


def test_hook_dependancies():
    hook = RandomNegativeEdgeSamplerHook(low=0, high=10)
    assert hook.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook.produces == {'neg', 'neg_time'}

    hook_with_id = RandomNegativeEdgeSamplerHook(low=0, high=10, id='foo')
    assert hook_with_id.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook_with_id.produces == {'neg_foo', 'neg_time_foo'}

    hook = HistoricalNegativeEdgeSamplerHook()
    assert hook.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook.produces == {'neg', 'neg_time', 'valid_neg_mask'}

    hook_with_id = HistoricalNegativeEdgeSamplerHook(id='foo')
    assert hook_with_id.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook_with_id.produces == {'neg_foo', 'neg_time_foo', 'valid_neg_mask_foo'}


def test_hook_repre():
    hook_with_id = RandomNegativeEdgeSamplerHook(low=0, high=10, id='foo')
    assert 'foo' in hook_with_id.__repr__()

    hook_with_id = HistoricalNegativeEdgeSamplerHook(id='foo')
    assert 'foo' in hook_with_id.__repr__()


def test_hook_reset_state():
    assert RandomNegativeEdgeSamplerHook.has_state == False
    assert HistoricalNegativeEdgeSamplerHook.has_state == True
    assert NodeTypeNegativeSamplerHook.has_state == True


def test_bad_negative_edge_sampler_init():
    with pytest.raises(ValueError):
        RandomNegativeEdgeSamplerHook(low=0, high=0)
    with pytest.raises(ValueError):
        RandomNegativeEdgeSamplerHook(low=0, high=1, neg_ratio=0)
    with pytest.raises(ValueError):
        RandomNegativeEdgeSamplerHook(low=0, high=1, neg_ratio=2)


def test_negative_edge_sampler(data):
    dg = DGraph(data)
    hook = RandomNegativeEdgeSamplerHook(low=0, high=10)
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg)
    assert torch.is_tensor(batch.neg_time)
    assert batch.neg.shape == batch.edge_dst.shape
    assert batch.neg_time.shape == batch.neg.shape


def test_negative_edge_sampler_with_id(data):
    dg = DGraph(data)
    hook = RandomNegativeEdgeSamplerHook(low=0, high=10, id='foo')
    batch = hook(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert torch.is_tensor(batch.neg_foo)
    assert torch.is_tensor(batch.neg_time_foo)
    assert batch.neg_foo.shape == batch.edge_dst.shape
    assert batch.neg_time_foo.shape == batch.neg_foo.shape


@pytest.fixture
def node_only_data():
    edge_index = torch.IntTensor([[1, 2], [2, 3], [3, 4]])
    edge_time = torch.IntTensor([1, 2, 3])
    node_x = torch.rand(2, 5)
    node_x_time = torch.IntTensor([4, 5])
    node_x_nids = torch.IntTensor([5, 6])
    return DGData.from_raw(
        edge_time,
        edge_index,
        node_x=node_x,
        node_x_time=node_x_time,
        node_x_nids=node_x_nids,
    )


def test_node_only_batch_negative_edge_sampler(node_only_data):
    dg = DGraph(node_only_data)
    hm = HookManager(keys=['unit'])
    hm.register('unit', RandomNegativeEdgeSamplerHook(low=0, high=6))
    loader = DGDataLoader(dg, batch_size=3, hook_manager=hm)
    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        assert isinstance(batch_1, DGBatch)
        assert torch.is_tensor(batch_1.neg)
        assert torch.is_tensor(batch_1.neg_time)
        assert batch_1.neg.shape == batch_1.edge_dst.shape
        assert batch_1.neg_time.shape == batch_1.neg.shape

        batch_2 = next(batch_iter)
        assert isinstance(batch_2, DGBatch)
        assert batch_2.neg.shape == (0,)
        assert batch_2.neg_time.shape == (0,)


@pytest.fixture
def data_test_hst_sampling():
    edge_index = torch.IntTensor(
        [
            # 1st batch
            [1, 5],
            [7, 6],
            [2, 8],
            [7, 8],
            # 2nd batch
            [1, 7],
            [9, 10],
            [3, 10],
            [1, 9],
            # 3rd batch
            [3, 11],
            [2, 10],
            [7, 2],
            [3, 5],
        ]
    )
    edge_time = torch.arange(edge_index.size(0))
    return DGData.from_raw(edge_time, edge_index)


def test_hst_sampling(data_test_hst_sampling):
    dg = DGraph(data_test_hst_sampling)

    hm = HookManager(keys=['unit'])
    sampler = HistoricalNegativeEdgeSamplerHook()

    hm.register('unit', sampler)
    loader = DGDataLoader(dg, batch_size=4, hook_manager=hm)

    with hm.activate('unit'):
        batch_iter = iter(loader)
        batch_1 = next(batch_iter)
        assert batch_1.neg.shape == (4,)
        assert torch.equal(
            batch_1.neg,
            torch.Tensor(
                [PADDED_NODE_ID, PADDED_NODE_ID, PADDED_NODE_ID, PADDED_NODE_ID]
            ),
        )
        assert torch.equal(
            batch_1.valid_neg_mask,
            torch.Tensor([False, False, False, False]),
        )
        assert sampler._memory is not None
        assert sampler._memory.shape == (2, 8)
        assert sampler._count == 4

        batch_2 = next(batch_iter)
        assert batch_2.neg.shape == (4,)
        assert torch.equal(
            batch_2.neg, torch.Tensor([5, PADDED_NODE_ID, PADDED_NODE_ID, 5])
        )
        assert torch.equal(
            batch_2.valid_neg_mask,
            torch.Tensor([True, False, False, True]),
        )
        assert sampler._memory is not None
        assert sampler._memory.shape == (2, 8)
        assert sampler._count == 8

        batch_3 = next(batch_iter)
        assert batch_3.neg.shape == (4,)
        assert torch.equal(batch_3.neg, torch.Tensor([10, 8, 8, 10])) or torch.equal(
            batch_3.neg, torch.Tensor([10, 8, 6, 10])
        )
        assert torch.equal(
            batch_3.valid_neg_mask,
            torch.Tensor([True, True, True, True]),
        )
        assert sampler._memory is not None
        assert sampler._memory.shape == (2, 24)
        assert sampler._count != 0
        assert sampler._count == 12

        sampler.reset_state()
        assert sampler._memory is None
        assert sampler._count == 0


@pytest.fixture
def data_test_node_type_sampling():
    """Fixture for node type sampling tests.

    Graph with 12 nodes (0-11), node types assigned as:
        type 0: nodes [0, 2, 4, 6, 8, 10]  (even)
        type 1: nodes [1, 3, 5, 7, 9, 11]  (odd)

    Edges (src, dst):
        # 1st batch
        [1, 2],  dst=2  type 0
        [3, 4],  dst=4  type 0
        [5, 6],  dst=6  type 0
        [7, 1],  dst=1  type 1
        # 2nd batch
        [2, 3],  dst=3  type 1  -> can sample node 1 (type 1, seen in batch 1)
        [4, 8],  dst=8  type 0  -> can sample nodes 2, 4, 6 (type 0, seen in batch 1)
        [6, 5],  dst=5  type 1  -> can sample node 1 (type 1, seen in batch 1)
        [8, 9],  dst=9  type 1  -> can sample node 1 (type 1, seen in batch 1)
        # 3rd batch
        [1, 10], dst=10 type 0  -> can sample nodes 2, 4, 6, 8 (type 0, seen in batches 1&2)
        [3, 7],  dst=7  type 1  -> can sample nodes 1, 3, 5, 9 (type 1, seen in batches 1&2)
        [5, 11], dst=11 type 1  -> can sample nodes 1, 3, 5, 9 (type 1, seen in batches 1&2)
        [7, 0],  dst=0  type 0  -> can sample nodes 2, 4, 6, 8 (type 0, seen in batches 1&2).
    """
    edge_index = torch.IntTensor(
        [
            # 1st batch
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 1],
            # 2nd batch
            [2, 3],
            [4, 8],
            [6, 5],
            [8, 9],
            # 3rd batch
            [1, 10],
            [3, 7],
            [5, 11],
            [7, 0],
        ]
    )
    edge_time = torch.arange(edge_index.size(0))
    node_type = torch.IntTensor([i % 2 for i in range(12)])  # even=type 0, odd=type 1
    node_type[-1] = (
        2  # The last node (11) is of type 2, which is not present in the graph edges
    )
    return DGData.from_raw(
        edge_time=edge_time, edge_index=edge_index, node_type=node_type
    )


def test_node_type_sampling(data_test_node_type_sampling):
    dg = DGraph(data_test_node_type_sampling)

    hm = HookManager(keys=['unit'])
    sampler = NodeTypeNegativeSamplerHook(num_nodes=dg.num_nodes)

    hm.register('unit', sampler)
    loader = DGDataLoader(dg, batch_size=4, hook_manager=hm)

    with hm.activate('unit'):
        batch_iter = iter(loader)

        # batch 1: no memory yet, all padded
        batch_1 = next(batch_iter)
        assert batch_1.neg.shape == (4,)
        assert torch.equal(
            batch_1.neg,
            torch.full((4,), PADDED_NODE_ID, dtype=batch_1.neg.dtype),
        )
        assert torch.equal(
            batch_1.valid_neg_mask,
            torch.zeros(4, dtype=torch.bool),
        )
        # memory initialized with num_nodes entries
        assert sampler._memory is not None
        # print()
        assert sampler._memory.shape == (dg.num_nodes,)
        # dst nodes 1, 2, 4, 6 recorded in memory
        assert sampler._memory[2] == 0  # type 0
        assert sampler._memory[4] == 0  # type 0
        assert sampler._memory[6] == 0  # type 0
        assert sampler._memory[1] == 1  # type 1

        # batch 2: memory has nodes 1(t1), 2(t0), 4(t0), 6(t0)
        batch_2 = next(batch_iter)
        assert batch_2.neg.shape == (4,)
        assert torch.equal(
            batch_2.valid_neg_mask,
            torch.ones(4, dtype=torch.bool),
        )
        # dst=3(t1) -> neg must be type 1 -> only node 1
        assert batch_2.neg[0].item() == 1
        # dst=8(t0) -> neg must be type 0 -> one of {2, 4, 6}
        assert batch_2.neg[1].item() in {2, 4, 6}
        # dst=5(t1) -> neg must be type 1 -> only node 1
        assert batch_2.neg[2].item() == 1
        # dst=9(t1) -> neg must be type 1 -> only node 1
        assert batch_2.neg[3].item() == 1

        # batch 3: memory now also has nodes 3(t1), 5(t1), 8(t0), 9(t1)
        batch_3 = next(batch_iter)
        assert batch_3.neg.shape == (4,)
        assert torch.equal(
            batch_3.valid_neg_mask,
            torch.Tensor([True, True, False, True]),
        )
        # dst=10(t0) -> neg must be type 0 -> one of {2, 4, 6, 8}
        assert batch_3.neg[0].item() in {2, 4, 6, 8}
        # dst=7(t1)  -> neg must be type 1 -> one of {1, 3, 5, 9}
        assert batch_3.neg[1].item() in {1, 3, 5, 9}
        # dst=11(t1) -> neg must be type 2 -> no nodes of type 2 have been seen yet, so neg is padded
        assert batch_3.neg[2].item() == PADDED_NODE_ID
        # dst=0(t0)  -> neg must be type 0 -> one of {2, 4, 6, 8}
        assert batch_3.neg[3].item() in {2, 4, 6, 8}

        sampler.reset_state()
        assert sampler._memory is None


def test_node_type_sampling_no_node_type():
    """Hook should raise ValueError when dg.node_type is None."""
    edge_index = torch.IntTensor([[1, 2], [3, 4]])
    edge_time = torch.arange(2)
    data = DGData.from_raw(edge_time=edge_time, edge_index=edge_index)
    dg = DGraph(data)

    sampler = NodeTypeNegativeSamplerHook(num_nodes=dg.num_nodes)
    with pytest.raises(ValueError, match='dg.node_type is None'):
        sampler(dg, dg.materialize())


def test_node_type_sampling_with_id(data_test_node_type_sampling):
    """Hook with id should suffix all produced attributes."""
    dg = DGraph(data_test_node_type_sampling)

    sampler = NodeTypeNegativeSamplerHook(num_nodes=dg.num_nodes, id='foo')
    batch = sampler(dg, dg.materialize())

    assert hasattr(batch, 'neg_foo')
    assert hasattr(batch, 'neg_time_foo')
    assert hasattr(batch, 'valid_neg_mask_foo')


def test_node_type_sampling_dependencies():
    """Check requires/produces sets."""
    hook = NodeTypeNegativeSamplerHook(num_nodes=10)
    assert hook.requires == {'edge_src', 'edge_dst', 'edge_time'}
    assert hook.produces == {'neg', 'neg_time', 'valid_neg_mask'}

    hook_with_id = NodeTypeNegativeSamplerHook(num_nodes=10, id='foo')
    assert hook_with_id.produces == {'neg_foo', 'neg_time_foo', 'valid_neg_mask_foo'}


def test_node_type_sampling_bad_init():
    """Hook should raise ValueError when num_nodes <= 0."""
    with pytest.raises(ValueError, match='num_nodes must be a positive integer'):
        NodeTypeNegativeSamplerHook(num_nodes=0)
    with pytest.raises(ValueError, match='num_nodes must be a positive integer'):
        NodeTypeNegativeSamplerHook(num_nodes=-1)

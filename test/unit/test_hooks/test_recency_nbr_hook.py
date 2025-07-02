import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import RecencyNeighborHook


def test_hook_dependancies():
    assert RecencyNeighborHook.requires == set()
    assert RecencyNeighborHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'nbr_mask',
        'times',
    }


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[0], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[-1], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[], num_nodes=2)


@pytest.mark.skip('TODO: Add recency nbr tests')
def test_neighhor_sampler_hook_self_loop():
    pass


@pytest.mark.skip('TODO: Add recency nbr tests')
def test_neighhor_sampler_hook_cycle():
    pass


@pytest.mark.skip('TODO: Add recency nbr tests')
def test_neighbor_sampler_hook():
    # 1-10      1-20        1-30      1-40   |  Now create a batch with every node to see nbrs
    #           2-20                  2-40   |
    #                       3-30             |
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    edge_feats = torch.rand(4, 5)
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)

    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2, 2], num_nodes=51)

    # TODO: Verify features here as well

    # Batch 1
    src = torch.LongTensor([1])
    dst = torch.LongTensor([10])
    time = torch.LongTensor([0])
    edge_feats = torch.rand(1, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 10])] * 2
    exp_times = [torch.LongTensor([0, 0, 0])] * 2
    exp_nbr_mask = [torch.LongTensor([[0, 0], [0, 0]])] * 2
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.times, exp_times)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (2, 2)
    assert batch.nbr_times[0].shape == (2, 2)
    assert batch.nbr_feats[0].shape == (2, 2, 5)
    assert batch.nbr_nids[1].shape == (2, 2)
    assert batch.nbr_times[1].shape == (2, 2)
    assert batch.nbr_feats[1].shape == (2, 2, 5)

    # Batch 2
    src = torch.LongTensor([1, 2])
    dst = torch.LongTensor([20, 20])
    time = torch.LongTensor([1, 1])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 2, 20, 20])] * 2
    exp_nbr_mask = [torch.LongTensor([[0, 0], [0, 0], [0, 0], [0, 0]])] * 2
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (4, 2)
    assert batch.nbr_times[0].shape == (4, 2)
    assert batch.nbr_feats[0].shape == (4, 2, 5)
    assert batch.nbr_nids[1].shape == (4, 2)
    assert batch.nbr_times[1].shape == (4, 2)
    assert batch.nbr_feats[1].shape == (4, 2, 5)

    # Batch 3
    src = torch.Tensor([1, 3])
    dst = torch.Tensor([30, 30])
    time = torch.Tensor([2, 2])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [
        torch.LongTensor([1, 3, 30, 30]),
        torch.LongTensor([1, 3, 30, 30, 10, 20]),
    ]
    exp_nbr_mask = [
        torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0]]),
        torch.LongTensor(
            [
                [1, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ]
        ),
    ]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (4, 2)
    assert batch.nbr_times[0].shape == (4, 2)
    assert batch.nbr_feats[0].shape == (4, 2, 5)
    assert batch.nbr_nids[1].shape == (6, 2)
    assert batch.nbr_times[1].shape == (6, 2)
    assert batch.nbr_feats[1].shape == (6, 2, 5)

    assert batch.nbr_nids[0][0][0] == 10
    assert batch.nbr_times[0][0][0] == 0

    assert batch.nbr_nids[0][0][1] == 20
    assert batch.nbr_times[0][0][1] == 1

    assert batch.nbr_nids[1].shape == (6, 2)
    assert batch.nbr_times[1].shape == (6, 2)
    assert batch.nbr_feats[1].shape == (6, 2, 5)

    assert batch.nbr_nids[1][0][0] == 10
    assert batch.nbr_times[1][0][0] == 0

    assert batch.nbr_nids[1][0][1] == 20
    assert batch.nbr_times[1][0][1] == 1

    assert batch.nbr_nids[1][5][0] == 1
    assert batch.nbr_times[1][5][0] == 1

    assert batch.nbr_nids[1][5][1] == 2
    assert batch.nbr_times[1][5][1] == 1

    # Batch 4
    src = torch.Tensor([1, 2])
    dst = torch.Tensor([40, 40])
    time = torch.Tensor([3, 3])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [
        torch.LongTensor([1, 2, 40, 40]),
        torch.LongTensor([1, 2, 40, 40, 20, 30]),
    ]
    exp_nbr_mask = [
        torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0]]),
        torch.LongTensor(
            [
                [1, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
            ]
        ),
    ]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (4, 2)
    assert batch.nbr_times[0].shape == (4, 2)
    assert batch.nbr_feats[0].shape == (4, 2, 5)
    assert batch.nbr_nids[1].shape == (6, 2)
    assert batch.nbr_times[1].shape == (6, 2)
    assert batch.nbr_feats[1].shape == (6, 2, 5)

    assert batch.nbr_nids[0][0][0] == 20
    assert batch.nbr_times[0][0][0] == 1

    assert batch.nbr_nids[0][0][1] == 30
    assert batch.nbr_times[0][0][1] == 2

    assert batch.nbr_nids[1][0][0] == 20
    assert batch.nbr_times[1][0][0] == 1

    assert batch.nbr_nids[1][0][1] == 30
    assert batch.nbr_times[1][0][1] == 2

    assert batch.nbr_nids[1][5][0] == 1
    assert batch.nbr_times[1][5][0] == 2

    # Batch 5
    src = torch.Tensor([1, 2, 3, 10, 20, 30, 40])
    dst = torch.Tensor([50] * len(src))
    time = torch.Tensor([4] * len(src))
    edge_feats = torch.rand(len(src), 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [
        torch.LongTensor([1, 2, 3, 10, 20, 30, 40, 50, 50, 50, 50, 50, 50, 50]),
        torch.LongTensor(
            [
                1,
                2,
                3,
                10,
                20,
                30,
                40,
                50,
                50,
                50,
                50,
                50,
                50,
                50,
                30,
                40,
                20,
                40,
                1,
                2,
                1,
                3,
                1,
                2,
            ]
        ),
    ]

    exp_nbr_mask = [
        torch.LongTensor(
            [
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        ),
        torch.LongTensor(
            [
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [0, 0],
                [1, 1],
                [1, 1],
            ]
        ),
    ]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0][0][0] == 30
    assert batch.nbr_times[0][0][0] == 2

    assert batch.nbr_nids[0][0][1] == 40
    assert batch.nbr_times[0][0][1] == 3

    assert batch.nbr_nids[0][1][0] == 20
    assert batch.nbr_times[0][1][0] == 1

    assert batch.nbr_nids[0][1][1] == 40
    assert batch.nbr_times[0][1][1] == 3

    assert batch.nbr_nids[0][4][0] == 1
    assert batch.nbr_times[0][4][0] == 1

    assert batch.nbr_nids[0][4][1] == 2
    assert batch.nbr_times[0][4][1] == 1

    assert batch.nbr_nids[0][5][0] == 1
    assert batch.nbr_times[0][5][0] == 2

    assert batch.nbr_nids[0][5][1] == 3
    assert batch.nbr_times[0][5][1] == 2

    assert batch.nbr_nids[0][6][0] == 1
    assert batch.nbr_times[0][6][0] == 3

    assert batch.nbr_nids[0][6][1] == 2
    assert batch.nbr_times[0][6][1] == 3

    assert batch.nbr_nids[1].shape == (24, 2)
    assert batch.nbr_times[1].shape == (24, 2)
    assert batch.nbr_feats[1].shape == (24, 2, 5)

    assert batch.nbr_nids[1][0][0] == 30
    assert batch.nbr_times[1][0][0] == 2

    assert batch.nbr_nids[1][0][1] == 40
    assert batch.nbr_times[1][0][1] == 3

    assert batch.nbr_nids[1][1][0] == 20
    assert batch.nbr_times[1][1][0] == 1

    assert batch.nbr_nids[1][1][1] == 40
    assert batch.nbr_times[1][1][1] == 3

    assert batch.nbr_nids[1][4][0] == 1
    assert batch.nbr_times[1][4][0] == 1

    assert batch.nbr_nids[1][4][1] == 2
    assert batch.nbr_times[1][4][1] == 1

    assert batch.nbr_nids[1][5][0] == 1
    assert batch.nbr_times[1][5][0] == 2

    assert batch.nbr_nids[1][5][1] == 3
    assert batch.nbr_times[1][5][1] == 2

    assert batch.nbr_nids[1][6][0] == 1
    assert batch.nbr_times[1][6][0] == 3

    assert batch.nbr_nids[1][6][1] == 2
    assert batch.nbr_times[1][6][1] == 3

    assert batch.nbr_nids[1][14][0] == 1
    assert batch.nbr_times[1][14][0] == 2

    assert batch.nbr_nids[1][14][1] == 3
    assert batch.nbr_times[1][14][1] == 2

    assert batch.nbr_nids[1][15][0] == 1
    assert batch.nbr_times[1][15][0] == 3

    assert batch.nbr_nids[1][15][1] == 2
    assert batch.nbr_times[1][15][1] == 3

    assert batch.nbr_nids[1][16][0] == 1
    assert batch.nbr_times[1][16][0] == 1

    assert batch.nbr_nids[1][16][1] == 2
    assert batch.nbr_times[1][16][1] == 1

    assert batch.nbr_nids[1][17][0] == 1
    assert batch.nbr_times[1][17][0] == 3

    assert batch.nbr_nids[1][17][1] == 2
    assert batch.nbr_times[1][17][1] == 3

    assert batch.nbr_nids[1][18][0] == 30
    assert batch.nbr_times[1][18][0] == 2

    assert batch.nbr_nids[1][18][1] == 40
    assert batch.nbr_times[1][18][1] == 3

    assert batch.nbr_nids[1][19][0] == 20
    assert batch.nbr_times[1][19][0] == 1

    assert batch.nbr_nids[1][19][1] == 40
    assert batch.nbr_times[1][19][1] == 3

    assert batch.nbr_nids[1][20][0] == 30
    assert batch.nbr_times[1][20][0] == 2

    assert batch.nbr_nids[1][20][1] == 40
    assert batch.nbr_times[1][20][1] == 3

    assert batch.nbr_nids[1][22][0] == 30
    assert batch.nbr_times[1][22][0] == 2

    assert batch.nbr_nids[1][22][1] == 40
    assert batch.nbr_times[1][22][1] == 3

    assert batch.nbr_nids[1][23][0] == 20
    assert batch.nbr_times[1][23][0] == 1

    assert batch.nbr_nids[1][23][1] == 40
    assert batch.nbr_times[1][23][1] == 3


@pytest.mark.skip('TODO: Add recency nbr tests')
def test_neighbor_sampler_hook_neg_edges():
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    edge_feats = torch.rand(4, 5)
    data = DGData.from_raw(edge_timestamps, edge_index, edge_feats)
    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=51)

    # Batch 1
    src = torch.LongTensor([1])
    dst = torch.LongTensor([10])
    time = torch.Tensor([0])
    edge_feats = torch.rand(1, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    # Add negative edges
    batch.neg = torch.LongTensor([7, 8])
    batch = hook(dg, batch)

    # And ensure they are part of the seed nodes on the first hop
    exp_nids = [torch.LongTensor([1, 10, 7, 8])]
    torch.testing.assert_close(batch.nids, exp_nids)
    assert batch.nbr_nids[0].shape == (4, 2)

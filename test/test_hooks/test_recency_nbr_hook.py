import pytest
import torch

from opendg.data import DGData
from opendg.graph import DGBatch, DGraph
from opendg.hooks import RecencyNeighborHook


@pytest.fixture
def data():
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    edge_feats = torch.rand(4, 5)
    return DGData.from_raw(edge_timestamps, edge_index, edge_feats)


def test_hook_dependancies():
    assert RecencyNeighborHook.requires == set()
    assert RecencyNeighborHook.produces == {
        'nids',
        'nbr_nids',
        'nbr_times',
        'nbr_feats',
        'nbr_mask',
    }


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[0], num_nodes=2)
    with pytest.raises(ValueError):
        RecencyNeighborHook(num_nbrs=[-1], num_nodes=2)


def test_neighbor_sampler_hook_init(data):
    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=dg.num_nodes)
    assert hook.num_nbrs == [2]


def test_neighbor_sampler_hook_single_hop(data):
    # 1-10      1-20        1-30      1-40      1-50    | Now create a batch with every single node and node 600, to see the nbrs
    #           2-20                  2-40              |
    #                       3-30                        |
    #                                                   |
    # 10-100    20-200      30-300    40-400    50-500  |

    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=601)

    # TODO: Verify features here as well

    # Batch 1
    src = torch.Tensor([1, 10])
    dst = torch.Tensor([10, 100])
    time = torch.Tensor([0, 0])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)
    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 10, 10, 100])]
    exp_nbr_mask = [torch.LongTensor([[0, 0], [0, 0], [0, 0], [0, 0]])]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (4, 2)
    assert batch.nbr_times[0].shape == (4, 2)
    assert batch.nbr_feats[0].shape == (4, 2, 5)

    # Batch 2
    src = torch.Tensor([1, 2, 20])
    dst = torch.Tensor([20, 20, 200])
    time = torch.Tensor([1, 1, 1])
    edge_feats = torch.rand(3, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 2, 20, 20, 20, 200])]
    exp_nbr_mask = [torch.LongTensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (6, 2)
    assert batch.nbr_times[0].shape == (6, 2)
    assert batch.nbr_feats[0].shape == (6, 2, 5)

    # Batch 3
    src = torch.Tensor([1, 3, 30])
    dst = torch.Tensor([30, 30, 300])
    time = torch.Tensor([2, 2, 2])
    edge_feats = torch.rand(3, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 3, 30, 30, 30, 300])]
    exp_nbr_mask = [torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (6, 2)
    assert batch.nbr_times[0].shape == (6, 2)
    assert batch.nbr_feats[0].shape == (6, 2, 5)

    assert batch.nbr_nids[0][0][0] == 10  # 1-10 edge
    assert batch.nbr_times[0][0][0] == 0  # 1-10 edge

    assert batch.nbr_nids[0][0][1] == 20  # 1-20 edge
    assert batch.nbr_times[0][0][1] == 1  # 1-20 edge

    # Batch 4
    src = torch.Tensor([1, 2, 40])
    dst = torch.Tensor([40, 40, 400])
    time = torch.Tensor([3, 3, 3])
    edge_feats = torch.rand(3, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 2, 40, 40, 40, 400])]
    exp_nbr_mask = [torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (6, 2)
    assert batch.nbr_times[0].shape == (6, 2)
    assert batch.nbr_feats[0].shape == (6, 2, 5)

    assert batch.nbr_nids[0][0][0] == 20  # 1-20 edge
    assert batch.nbr_times[0][0][0] == 1  # 1-20 edge

    assert batch.nbr_nids[0][0][1] == 30  # 1-30 edge
    assert batch.nbr_times[0][0][1] == 2  # 1-30 edge

    # Batch 5
    src = torch.Tensor([1, 50])
    dst = torch.Tensor([50, 500])
    time = torch.Tensor([4, 4])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [torch.LongTensor([1, 50, 50, 500])]
    exp_nbr_mask = [torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0]])]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (4, 2)
    assert batch.nbr_times[0].shape == (4, 2)
    assert batch.nbr_feats[0].shape == (4, 2, 5)

    assert batch.nbr_nids[0][0][0] == 30  # 1-30 edge
    assert batch.nbr_times[0][0][0] == 2  # 1-30 edge

    assert batch.nbr_nids[0][0][1] == 40  # 1-40 edge
    assert batch.nbr_times[0][0][1] == 3  # 1-40 edge

    # Batch 6
    src = torch.Tensor([1, 2, 3, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500])
    dst = torch.Tensor([600] * len(src))
    time = torch.Tensor([5] * len(src))
    edge_feats = torch.rand(len(src), 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)

    exp_nids = [
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
                100,
                200,
                300,
                400,
                500,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
                600,
            ]
        )
    ]
    exp_nbr_mask = [
        torch.LongTensor(
            [
                [1, 1],
                [1, 1],
                [0, 0],
                [1, 1],
                [1, 1],
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
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
    ]
    torch.testing.assert_close(batch.nids, exp_nids)
    torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    assert batch.nbr_nids[0].shape == (26, 2)
    assert batch.nbr_times[0].shape == (26, 2)
    assert batch.nbr_feats[0].shape == (26, 2, 5)

    assert batch.nbr_nids[0][0][0] == 40  # 1-40 edge
    assert batch.nbr_times[0][0][0] == 3

    assert batch.nbr_nids[0][0][1] == 50  # 1-50 edge
    assert batch.nbr_times[0][0][1] == 4

    assert batch.nbr_nids[0][1][0] == 20  # 2-20 edge
    assert batch.nbr_times[0][1][0] == 1

    assert batch.nbr_nids[0][1][1] == 40  # 2-40 edge
    assert batch.nbr_times[0][1][1] == 3

    assert batch.nbr_nids[0][2][0] == 30  # 3-30 edge
    assert batch.nbr_times[0][2][0] == 2

    assert batch.nbr_nids[0][3][0] == 1  # 10-1 edge
    assert batch.nbr_times[0][3][0] == 0

    assert batch.nbr_nids[0][3][1] == 100  # 10-100 edge
    assert batch.nbr_times[0][3][1] == 0

    assert batch.nbr_nids[0][4][0] == 2  # 20-2 edge
    assert batch.nbr_times[0][4][0] == 1

    assert batch.nbr_nids[0][4][1] == 200  # 20-200 edge
    assert batch.nbr_times[0][4][1] == 1

    assert batch.nbr_nids[0][5][0] == 3  # 30-3 edge
    assert batch.nbr_times[0][5][0] == 2

    assert batch.nbr_nids[0][5][1] == 300  # 30-300 edge
    assert batch.nbr_times[0][5][1] == 2

    assert batch.nbr_nids[0][6][0] == 2  # 40-2 edge
    assert batch.nbr_times[0][6][0] == 3

    assert batch.nbr_nids[0][6][1] == 400  # 40-400 edge
    assert batch.nbr_times[0][6][1] == 3

    assert batch.nbr_nids[0][7][0] == 1  # 50-1 edge
    assert batch.nbr_times[0][7][0] == 4

    assert batch.nbr_nids[0][7][1] == 500  # 50-500 edge
    assert batch.nbr_times[0][7][1] == 4


def test_neighbour_sampler_hook_multi_hop(data):
    # 1-10      1-20        1-30      1-40      1-50    | Now create a batch with every single node and node 600, to see the nbrs
    #           2-20                  2-40              |
    #                       3-30                        |
    #                                                   |
    # 10-100    20-200      30-300    40-400    50-500  |

    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2, 2], num_nodes=601)

    # TODO: Verify features here as well

    # Batch 1
    src = torch.Tensor([1, 10])
    dst = torch.Tensor([10, 100])
    time = torch.Tensor([0, 0])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)
    batch = hook(dg, batch)

    [torch.LongTensor([1, 10, 10, 100])]
    [torch.LongTensor([[0, 0], [0, 0], [0, 0], [0, 0]])]
    # torch.testing.assert_close(batch.nids, exp_nids)
    # torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    # assert batch.nbr_nids[0].shape == (4, 2)
    # assert batch.nbr_times[0].shape == (4, 2)
    # assert batch.nbr_feats[0].shape == (4, 2, 5)

    # Batch 2
    src = torch.Tensor([1, 2, 20])
    dst = torch.Tensor([20, 20, 200])
    time = torch.Tensor([1, 1, 1])
    edge_feats = torch.rand(3, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)
    print()
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500]:
        print(k, hook._nbrs[k][0])
    print(f'Nids: {batch.nids}')
    print(f'Nbr nids: {batch.nbr_nids}')
    print(f'Nbr times: {batch.nbr_times}')
    print(f'Nbr mask: {batch.nbr_mask}')

    # exp_nids = [torch.LongTensor([1, 2, 20, 20, 20, 200])]
    # exp_nbr_mask = [torch.LongTensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])]
    # torch.testing.assert_close(batch.nids, exp_nids)
    # torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    # assert batch.nbr_nids[0].shape == (6, 2)
    # assert batch.nbr_times[0].shape == (6, 2)
    # assert batch.nbr_feats[0].shape == (6, 2, 5)

    # Batch 3
    src = torch.Tensor([1, 3, 30])
    dst = torch.Tensor([30, 30, 300])
    time = torch.Tensor([2, 2, 2])
    edge_feats = torch.rand(3, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)
    print()
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500]:
        print(k, hook._nbrs[k][0])
    print(f'Nids: {batch.nids}')
    print(f'Nbr nids: {batch.nbr_nids}')
    print(f'Nbr times: {batch.nbr_times}')
    print(f'Nbr mask: {batch.nbr_mask}')

    # exp_nids = [torch.LongTensor([1, 3, 30, 30, 30, 300])]
    # exp_nbr_mask = [torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])]
    # torch.testing.assert_close(batch.nids, exp_nids)
    # torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    # assert batch.nbr_nids[0].shape == (6, 2)
    # assert batch.nbr_times[0].shape == (6, 2)
    # assert batch.nbr_feats[0].shape == (6, 2, 5)

    # assert batch.nbr_nids[0][0][0] == 10  # 1-10 edge
    # assert batch.nbr_times[0][0][0] == 0  # 1-10 edge

    # assert batch.nbr_nids[0][0][1] == 20  # 1-20 edge
    # assert batch.nbr_times[0][0][1] == 1  # 1-20 edge

    # Batch 4
    src = torch.Tensor([1, 2, 40])
    dst = torch.Tensor([40, 40, 400])
    time = torch.Tensor([3, 3, 3])
    edge_feats = torch.rand(3, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)
    print()
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500]:
        print(k, hook._nbrs[k][0])
    print(f'Nids: {batch.nids}')
    print(f'Nbr nids: {batch.nbr_nids}')
    print(f'Nbr times: {batch.nbr_times}')
    print(f'Nbr mask: {batch.nbr_mask}')

    # exp_nids = [torch.LongTensor([1, 2, 40, 40, 40, 400])]
    # exp_nbr_mask = [torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])]
    # torch.testing.assert_close(batch.nids, exp_nids)
    # torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    # assert batch.nbr_nids[0].shape == (6, 2)
    # assert batch.nbr_times[0].shape == (6, 2)
    # assert batch.nbr_feats[0].shape == (6, 2, 5)

    # assert batch.nbr_nids[0][0][0] == 20  # 1-20 edge
    # assert batch.nbr_times[0][0][0] == 1  # 1-20 edge

    # assert batch.nbr_nids[0][0][1] == 30  # 1-30 edge
    # assert batch.nbr_times[0][0][1] == 2  # 1-30 edge

    # Batch 5
    src = torch.Tensor([1, 50])
    dst = torch.Tensor([50, 500])
    time = torch.Tensor([4, 4])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)
    print()
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500]:
        print(k, hook._nbrs[k][0])
    print(f'Nids: {batch.nids}')
    print(f'Nbr nids: {batch.nbr_nids}')
    print(f'Nbr times: {batch.nbr_times}')
    print(f'Nbr mask: {batch.nbr_mask}')

    # exp_nids = [torch.LongTensor([1, 50, 50, 500])]
    # exp_nbr_mask = [torch.LongTensor([[1, 1], [0, 0], [0, 0], [0, 0]])]
    # torch.testing.assert_close(batch.nids, exp_nids)
    # torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    # assert batch.nbr_nids[0].shape == (4, 2)
    # assert batch.nbr_times[0].shape == (4, 2)
    # assert batch.nbr_feats[0].shape == (4, 2, 5)

    # assert batch.nbr_nids[0][0][0] == 30  # 1-30 edge
    # assert batch.nbr_times[0][0][0] == 2  # 1-30 edge

    # assert batch.nbr_nids[0][0][1] == 40  # 1-40 edge
    # assert batch.nbr_times[0][0][1] == 3  # 1-40 edge

    # Batch 6
    src = torch.Tensor([1, 2, 3, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500])
    dst = torch.Tensor([600] * len(src))
    time = torch.Tensor([5] * len(src))
    edge_feats = torch.rand(len(src), 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    batch = hook(dg, batch)
    print()
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300, 400, 500]:
        print(k, hook._nbrs[k][0])
    print(f'Nids: {batch.nids}')
    print(f'Nbr nids: {batch.nbr_nids}')
    print(f'Nbr times: {batch.nbr_times}')
    print(f'Nbr mask: {batch.nbr_mask}')

    # exp_nids = [
    #    torch.LongTensor(
    #        [
    #            1,
    #            2,
    #            3,
    #            10,
    #            20,
    #            30,
    #            40,
    #            50,
    #            100,
    #            200,
    #            300,
    #            400,
    #            500,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #            600,
    #        ]
    #    )
    # ]
    # exp_nbr_mask = [
    #    torch.LongTensor(
    #        [
    #            [1, 1],
    #            [1, 1],
    #            [0, 0],
    #            [1, 1],
    #            [1, 1],
    #            [1, 1],
    #            [1, 1],
    #            [1, 1],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #            [0, 0],
    #        ]
    #    )
    # ]
    # torch.testing.assert_close(batch.nids, exp_nids)
    # torch.testing.assert_close(batch.nbr_mask, exp_nbr_mask)
    # assert batch.nbr_nids[0].shape == (26, 2)
    # assert batch.nbr_times[0].shape == (26, 2)
    # assert batch.nbr_feats[0].shape == (26, 2, 5)

    # assert batch.nbr_nids[0][0][0] == 40  # 1-40 edge
    # assert batch.nbr_times[0][0][0] == 3

    # assert batch.nbr_nids[0][0][1] == 50  # 1-50 edge
    # assert batch.nbr_times[0][0][1] == 4

    # assert batch.nbr_nids[0][1][0] == 20  # 2-20 edge
    # assert batch.nbr_times[0][1][0] == 1

    # assert batch.nbr_nids[0][1][1] == 40  # 2-40 edge
    # assert batch.nbr_times[0][1][1] == 3

    # assert batch.nbr_nids[0][2][0] == 30  # 3-30 edge
    # assert batch.nbr_times[0][2][0] == 2

    # assert batch.nbr_nids[0][3][0] == 1  # 10-1 edge
    # assert batch.nbr_times[0][3][0] == 0

    # assert batch.nbr_nids[0][3][1] == 100  # 10-100 edge
    # assert batch.nbr_times[0][3][1] == 0

    # assert batch.nbr_nids[0][4][0] == 2  # 20-2 edge
    # assert batch.nbr_times[0][4][0] == 1

    # assert batch.nbr_nids[0][4][1] == 200  # 20-200 edge
    # assert batch.nbr_times[0][4][1] == 1

    # assert batch.nbr_nids[0][5][0] == 3  # 30-3 edge
    # assert batch.nbr_times[0][5][0] == 2

    # assert batch.nbr_nids[0][5][1] == 300  # 30-300 edge
    # assert batch.nbr_times[0][5][1] == 2

    # assert batch.nbr_nids[0][6][0] == 2  # 40-2 edge
    # assert batch.nbr_times[0][6][0] == 3

    # assert batch.nbr_nids[0][6][1] == 400  # 40-400 edge
    # assert batch.nbr_times[0][6][1] == 3

    # assert batch.nbr_nids[0][7][0] == 1  # 50-1 edge
    # assert batch.nbr_times[0][7][0] == 4

    # assert batch.nbr_nids[0][7][1] == 500  # 50-500 edge
    # assert batch.nbr_times[0][7][1] == 4


def test_neighbor_sampler_hook_neg_edges(data):
    dg = DGraph(data)
    hook = RecencyNeighborHook(num_nbrs=[2], num_nodes=601)

    src = torch.Tensor([1, 10])
    dst = torch.Tensor([10, 100])
    time = torch.Tensor([0, 0])
    edge_feats = torch.rand(2, 5)
    batch = DGBatch(src, dst, time, edge_feats=edge_feats)

    # Add negative edges
    batch.neg = torch.Tensor([7, 8])
    batch = hook(dg, batch)

    # And ensure they are part of the seed nodes on the first hop
    exp_nids = [torch.LongTensor([1, 10, 10, 100, 7, 8])]
    torch.testing.assert_close(batch.nids, exp_nids)
    assert batch.nbr_nids[0].shape == (6, 2)

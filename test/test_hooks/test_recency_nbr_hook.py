import pytest

from opendg.events import EdgeEvent
from opendg.graph import DGBatch, DGraph
from opendg.hooks import RecencyNeighborSamplerHook


@pytest.fixture
def events():
    return [
        EdgeEvent(t=1, src=1, dst=10),
        EdgeEvent(t=1, src=1, dst=11),
        EdgeEvent(t=2, src=1, dst=12),
        EdgeEvent(t=2, src=1, dst=13),
    ]


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[0])
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[-1])
    with pytest.raises(ValueError):
        RecencyNeighborSamplerHook(num_nbrs=[1, 2])


def test_neighbor_sampler_hook(events):
    dg = DGraph(events)
    hook = RecencyNeighborSamplerHook(num_nbrs=[2])
    batch = hook(dg)
    assert isinstance(batch, DGBatch)
    print(batch.nbrs)
    # TODO: Add logic for testing


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        LastNeighborHook(num_nodes=0, size=10)
    with pytest.raises(ValueError):
        LastNeighborHook(num_nodes=10, size=0)


def test_neighbor_sampler_hook():
    import time
    from queue import Queue

    import torch

    num_nodes = [1000, 10_000, 100_000, 1_000_000]
    ks = [16, 32, 64]
    bs = [256, 512, 1024]
    print('\n')

    for num_node in num_nodes:
        for k in ks:
            for b in bs:
                print(f'Nodes: {num_node}', end='')
                print(f'\tK: {k}', end='')
                print(f'\tBatchSize: {b}', end='')

                _nbr_dict = {}
                src = torch.randint(low=0, high=num_node - 1, size=(b,))
                dst = torch.randint(low=0, high=num_node - 1, size=(b,))
                ts = torch.randint(low=0, high=num_node - 1, size=(b,))

                st = time.perf_counter()
                for _ in range(10):
                    nids = torch.unique(torch.cat((src, dst)))
                    out_nbrs = {}

                    # retrieve recent neighbors for each node
                    for node in nids.tolist():
                        if node not in _nbr_dict:
                            _nbr_dict[node] = [
                                Queue(maxsize=k),
                                Queue(maxsize=k),
                            ]

                        out_nbrs[node] = []  # (dst,time, edge_feats)
                        out_nbrs[node].append(
                            torch.tensor(list(_nbr_dict[node][0].queue))
                        )  # dst
                        out_nbrs[node].append(
                            torch.tensor(list(_nbr_dict[node][1].queue))
                        )  # time

                    for i in range(src.size(0)):
                        src_nbr = int(src[i].item())
                        if _nbr_dict[src_nbr][0].full():
                            # pop the oldest neighbor
                            for kk in range(len(_nbr_dict[src_nbr])):
                                _nbr_dict[src_nbr][kk].get()

                        _nbr_dict[src_nbr][0].put(dst[i].item())
                        _nbr_dict[src_nbr][1].put(ts[i].item())

                et = time.perf_counter()
                throughput = b * 10 / (et - st)
                print(f'\t {throughput:.2f} updates/s')

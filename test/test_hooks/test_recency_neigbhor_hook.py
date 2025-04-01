import pytest

from opendg.events import EdgeEvent, NodeEvent
from opendg.hooks import LastNeighborHook


@pytest.fixture
def events():
    return [
        NodeEvent(t=1, src=2),
        EdgeEvent(t=1, src=2, dst=2),
        NodeEvent(t=5, src=4),
        EdgeEvent(t=5, src=2, dst=4),
        NodeEvent(t=10, src=6),
        EdgeEvent(t=20, src=1, dst=8),
    ]


def test_bad_neighbor_sampler_init():
    with pytest.raises(ValueError):
        LastNeighborHook(num_nodes=0, size=10)
    with pytest.raises(ValueError):
        LastNeighborHook(num_nodes=10, size=0)


def test_neighbor_sampler_hook(events):
    import time

    import torch

    def _update(src, dst):
        new_nbrs.fill_(-1)
        for i in range(len(src)):
            new_nbrs[i, src[i]] = dst[i]
            new_nbrs[i, dst[i]] = src[i]
        # print(new_nbrs)
        # print((new_nbrs >= 0).sum(dim=1))
        torch.cat([nbrs, new_nbrs])
        # all_nbr = torch.gather(all_nbr, 1, all_nbr.ne(-1).argsort(dim=1, stable=True))
        return nbrs
        # return all_nbr[:, -k:]

    from collections import deque

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

                qs = [deque(maxlen=k)] * num_node
                nbrs = torch.full((num_node, k), -1, dtype=torch.long)
                src = torch.randint(low=0, high=num_node - 1, size=(b,))
                dst = torch.randint(low=0, high=num_node - 1, size=(b,))

                st = time.perf_counter()
                for _ in range(10):
                    unique = torch.cat([src, dst]).unique()
                    for i in range(len(src)):
                        qs[src[i]].append(dst[i])
                        qs[dst[i]].append(src[i])
                    for u in unique:
                        nbrs[u] = torch.tensor(qs[u])
                et = time.perf_counter()
                throughput = b * 10 / (et - st)
                print(f'\t {throughput:.2f} updates/s')

from collections import deque

import numpy as np

_CAYLEY_BOUNDS = [
    (6, 2),
    (24, 3),
    (120, 5),
    (336, 7),
    (1320, 11),
    (2184, 13),
    (4896, 17),
    (6840, 19),
    (12144, 23),
    (24360, 29),
    (29760, 31),
    (50616, 37),
    (68880, 41),
    (79464, 43),
    (103776, 47),
    (148824, 53),
    (205320, 59),
    (226920, 61),
    (300696, 67),
    (357840, 71),
    (388944, 73),
    (492960, 79),
    (571704, 83),
    (704880, 89),
    (912576, 97),
    (1030200, 101),
    (1092624, 103),
    (1224936, 107),
    (1294920, 109),
    (1442784, 113),
    (2048256, 127),
    (2247960, 131),
]


def build_cayley_bank():
    ret_edges = []

    for _, p in _CAYLEY_BOUNDS:
        generators = np.array(
            [
                [[1, 1], [0, 1]],
                [[1, p - 1], [0, 1]],
                [[1, 0], [1, 1]],
                [[1, 0], [p - 1, 1]],
            ]
        )
        ind = 1

        queue = deque([np.array([[1, 0], [0, 1]])])
        nodes = {(1, 0, 0, 1): 0}

        senders = []
        receivers = []

        while queue:
            x = queue.pop()
            x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
            assert x_flat in nodes
            ind_x = nodes[x_flat]
            for i in range(4):
                tx = np.matmul(x, generators[i])
                tx = np.mod(tx, p)
                tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
                if tx_flat not in nodes:
                    nodes[tx_flat] = ind
                    ind += 1
                    queue.append(tx)
                ind_tx = nodes[tx_flat]

                senders.append(ind_x)
                receivers.append(ind_tx)

        ret_edges.append((p, [senders, receivers]))

    return ret_edges


def batched_augment_cayley(num_nodes, cayley_bank):
    p = 2
    chosen_i = -1

    senders = []
    receivers = []

    for i in range(len(_CAYLEY_BOUNDS)):
        sz, p = _CAYLEY_BOUNDS[i]
        if sz >= num_nodes:
            chosen_i = i
            break
    assert chosen_i >= 0

    _p, edge_pack = cayley_bank[chosen_i]
    assert p == _p

    for v, w in zip(*edge_pack):
        if v < num_nodes and w < num_nodes:
            senders.append(v)
            receivers.append(w)

    edge_attr = [[0] * 272 for _ in range(len(senders))]

    return [senders, receivers], edge_attr

import pytest
import torch

from tgm.nn import EdgeBankPredictor

# Node IDs used across tests never exceed this value, so N=25 is large
# enough for the vectorized (dense N x N) backend in every parametrized case.
MAX_N = 25


def _backend_kwargs(backend):
    """Return the extra kwargs required to construct a given backend."""
    return {'N': MAX_N} if backend == 'vectorized' else {}


@pytest.mark.parametrize('backend', ['vectorized', 'lookup'])
@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_unlimited_memory(backend, pos_prob):
    src = torch.Tensor([2, 10])
    dst = torch.Tensor([3, 20])
    ts = torch.Tensor([1, 5])

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='unlimited',
        pos_prob=pos_prob,
        backend=backend,
        **_backend_kwargs(backend),
    )
    assert bank(torch.Tensor([1]), torch.Tensor([1])) == torch.Tensor([0])

    bank.update(torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([7]))
    assert bank(torch.Tensor([1]), torch.Tensor([1])) == torch.Tensor([pos_prob])


@pytest.mark.parametrize('backend', ['vectorized', 'lookup'])
@pytest.mark.parametrize('pos_prob', [0.7, 1.0])
def test_fixed_time_window(backend, pos_prob):
    src = torch.Tensor([1, 2, 3, 4, 5, 6])
    dst = torch.Tensor([2, 3, 4, 5, 6, 7])
    ts = torch.Tensor([1, 2, 3, 4, 5, 6])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
        pos_prob=pos_prob,
        backend=backend,
        **_backend_kwargs(backend),
    )
    assert bank.window_start == 3.5
    assert bank.window_end == 6
    assert bank.window_ratio == WINDOW_RATIO

    assert bank(torch.Tensor([4]), torch.Tensor([5])) == torch.Tensor([pos_prob])
    assert bank(torch.Tensor([3]), torch.Tensor([4])) == torch.Tensor([0])

    # update but time window doesn't move forward
    bank.update(torch.Tensor([3]), torch.Tensor([4]), torch.Tensor([5]))
    assert bank(torch.Tensor([3]), torch.Tensor([4])) == torch.Tensor([pos_prob])

    # update and time window moves forward
    bank.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([7]))
    assert bank(torch.Tensor([7]), torch.Tensor([8])) == torch.Tensor([pos_prob])
    assert bank(torch.Tensor([4]), torch.Tensor([5])) == torch.Tensor([0])

    if backend == 'lookup':
        assert (
            not (torch.Tensor([4]), torch.Tensor([5])) in bank.backend.memory
        )  # The edge should be removed from the memory


@pytest.mark.parametrize('backend', ['vectorized', 'lookup'])
def test_complete_eviction_fixed_time_window(backend):
    src = torch.Tensor([1, 2, 3, 4, 5, 6])
    dst = torch.Tensor([2, 3, 4, 5, 6, 7])
    ts = torch.Tensor([1, 2, 3, 4, 5, 6])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
        backend=backend,
        **_backend_kwargs(backend),
    )
    assert bank.window_start == 3.5
    assert bank.window_end == 6
    assert bank.window_ratio == WINDOW_RATIO

    # Update with edge in the very far future. Evict all existing interactions
    bank.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([100000000]))

    for s, d in zip(src, dst):
        assert bank(s.unsqueeze(0), d.unsqueeze(0)) == torch.Tensor([0])

    assert bank(torch.Tensor([7]), torch.Tensor([8])) == torch.Tensor([bank.pos_prob])

    if backend == 'lookup':
        for s, d in zip(src, dst):
            assert not (s.item(), d.item()) in bank.backend.memory
        assert (7, 8) in bank.backend.memory


def test_out_of_order_construct_fixed():
    src = torch.Tensor([1, 2, 3, 4])
    dst = torch.Tensor([2, 3, 4, 5])
    ts = torch.Tensor([1, 4, 2, 3])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
        backend='lookup',
    )
    assert bank.window_start == 2.5
    assert bank.window_end == 4
    assert bank.window_ratio == WINDOW_RATIO

    expected_edge_order = [(4.0, 5.0), (2.0, 3.0)]
    expected_ts = range(3, 5)
    assert bank.backend._head is not None and bank.backend._tail is not None
    curr = bank.backend._head
    count = 0
    while curr is not None:
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        count += 1
        curr = curr.right

    assert count == 2


def test_out_of_order_update_fixed():
    src = torch.Tensor([1, 2, 3, 4])
    dst = torch.Tensor([2, 3, 4, 5])
    ts = torch.Tensor([1, 4, 2, 3])

    WINDOW_RATIO = 0.5

    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='fixed',
        window_ratio=WINDOW_RATIO,
        backend='lookup',
    )
    bank.update(torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([3]))
    assert bank.window_start == 2.5
    assert bank.window_end == 4
    assert bank.window_ratio == WINDOW_RATIO

    expected_edge_order = [(4.0, 5.0), (1.0, 1.0), (2.0, 3.0)]
    expected_ts = [3, 3, 4]
    assert bank.backend._head is not None and bank.backend._tail is not None
    curr = bank.backend._head
    count = 0
    while curr is not None:
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        count += 1
        curr = curr.right
    assert count == 3

    curr = bank.backend._tail
    count = 3
    while curr is not None:
        count -= 1
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        curr = curr.left
    assert count == 0


def test_out_of_order_construct_unlimited():
    src = torch.Tensor([1, 2, 3])
    dst = torch.Tensor([2, 3, 4])
    ts = torch.Tensor([3, 2, 1])
    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='unlimited',
        backend='lookup',
    )
    assert bank.window_start == 1
    assert bank.window_end == 3

    expected_edge_order = [
        (3.0, 4.0),
        (2.0, 3.0),
        (1.0, 2.0),
    ]
    expected_ts = range(1, 4)
    assert bank.backend._head is not None and bank.backend._tail is not None
    curr = bank.backend._head
    count = 0
    while curr is not None:
        assert curr.edge == expected_edge_order[count]
        assert curr.ts == expected_ts[count]
        count += 1
        curr = curr.right

    assert count == 3


def _make_pair(src, dst, ts, window_ratio=0.5, memory_mode='fixed', N=MAX_N):
    """Construct matching vectorized/lookup predictors from the same data."""
    vectorized = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode=memory_mode,
        window_ratio=window_ratio,
        N=N,
        backend='vectorized',
    )
    lookup = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode=memory_mode,
        window_ratio=window_ratio,
        backend='lookup',
    )
    return vectorized, lookup


def test_backends_agree_unlimited_memory():
    """Vectorized and lookup backends should give identical predictions in
    unlimited memory mode, including after edges are added via update().
    """
    src = torch.Tensor([1, 2, 3, 4, 5])
    dst = torch.Tensor([2, 3, 4, 5, 6])
    ts = torch.Tensor([1, 2, 3, 4, 5])

    vectorized, lookup = _make_pair(src, dst, ts, memory_mode='unlimited')

    # Mix of seen edges, unseen edges, and edges with swapped src/dst.
    query_src = torch.Tensor([1, 2, 3, 6, 2])
    query_dst = torch.Tensor([2, 3, 9, 7, 1])
    assert torch.equal(vectorized(query_src, query_dst), lookup(query_src, query_dst))

    # New edges should update both backends identically, and old edges
    # should never be evicted in unlimited mode.
    vectorized.update(torch.Tensor([9]), torch.Tensor([10]), torch.Tensor([6]))
    lookup.update(torch.Tensor([9]), torch.Tensor([10]), torch.Tensor([6]))

    all_src = torch.cat([query_src, torch.Tensor([9])])
    all_dst = torch.cat([query_dst, torch.Tensor([10])])
    assert torch.equal(vectorized(all_src, all_dst), lookup(all_src, all_dst))


def test_backends_agree_fixed_memory_with_eviction():
    """Vectorized and lookup backends should give identical predictions in
    fixed memory mode, including after the window slides forward and old
    edges fall out of scope.
    """
    src = torch.Tensor([1, 2, 3, 4, 5, 6])
    dst = torch.Tensor([2, 3, 4, 5, 6, 7])
    ts = torch.Tensor([1, 2, 3, 4, 5, 6])

    vectorized, lookup = _make_pair(src, dst, ts, window_ratio=0.5, memory_mode='fixed')

    # Include an unseen edge (still within the N x N bound the vectorized
    # backend allocates for) alongside the known edges.
    query_src = torch.cat([src, torch.Tensor([20])])
    query_dst = torch.cat([dst, torch.Tensor([21])])
    assert torch.equal(vectorized(query_src, query_dst), lookup(query_src, query_dst))

    # Push the window far into the future so the original edges are evicted
    # and only the newly added edge remains valid.
    vectorized.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([100]))
    lookup.update(torch.Tensor([7]), torch.Tensor([8]), torch.Tensor([100]))

    query_src = torch.cat([src, torch.Tensor([7])])
    query_dst = torch.cat([dst, torch.Tensor([8])])
    result_vectorized = vectorized(query_src, query_dst)
    result_lookup = lookup(query_src, query_dst)
    assert torch.equal(result_vectorized, result_lookup)
    # Sanity check: only the new edge should still be a hit.
    assert torch.equal(result_lookup, torch.Tensor([0, 0, 0, 0, 0, 0, lookup.pos_prob]))


@pytest.mark.parametrize('memory_mode', ['unlimited', 'fixed'])
def test_backends_agree_random_stress(memory_mode):
    """Randomized multi-batch stress test: both backends should stay in
    lockstep across several rounds of updates and queries.
    """
    torch.manual_seed(0)
    n_nodes = MAX_N
    n_init_edges = 20

    src = torch.randint(0, n_nodes, (n_init_edges,)).float()
    dst = torch.randint(0, n_nodes, (n_init_edges,)).float()
    ts = torch.arange(1, n_init_edges + 1).float()

    vectorized, lookup = _make_pair(
        src, dst, ts, window_ratio=0.3, memory_mode=memory_mode, N=n_nodes
    )

    next_ts = n_init_edges + 1
    for _ in range(5):
        query_src = torch.randint(0, n_nodes, (30,)).float()
        query_dst = torch.randint(0, n_nodes, (30,)).float()
        assert torch.equal(
            vectorized(query_src, query_dst), lookup(query_src, query_dst)
        )

        batch_size = 5
        upd_src = torch.randint(0, n_nodes, (batch_size,)).float()
        upd_dst = torch.randint(0, n_nodes, (batch_size,)).float()
        upd_ts = torch.arange(next_ts, next_ts + batch_size).float()
        next_ts += batch_size

        vectorized.update(upd_src, upd_dst, upd_ts)
        lookup.update(upd_src, upd_dst, upd_ts)


def test_bad_init_args():
    with pytest.raises(ValueError):
        EdgeBankPredictor(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

    with pytest.raises(TypeError):
        EdgeBankPredictor(1, 2, 3, backend='lookup')

    src = torch.Tensor([2, 10])
    dst = torch.Tensor([3, 20])
    ts = torch.Tensor([1, 5])
    with pytest.raises(ValueError):
        EdgeBankPredictor(src, dst, ts, memory_mode='foo')

    with pytest.raises(ValueError):
        EdgeBankPredictor(src, dst, ts, window_ratio=0)

    # invalid backend
    with pytest.raises(ValueError):
        EdgeBankPredictor(src, dst, ts, backend='foo')

    # vectorized backend requires N
    with pytest.raises(ValueError):
        EdgeBankPredictor(src, dst, ts, backend='vectorized')

    # vectorized backend works fine when N is provided
    EdgeBankPredictor(src, dst, ts, backend='vectorized', N=21)


@pytest.mark.parametrize('backend', ['vectorized', 'lookup'])
def test_bad_update_args(backend):
    src = torch.Tensor([2, 10])
    dst = torch.Tensor([3, 20])
    ts = torch.Tensor([1, 5])
    bank = EdgeBankPredictor(
        src,
        dst,
        ts,
        memory_mode='unlimited',
        backend=backend,
        **_backend_kwargs(backend),
    )

    with pytest.raises(ValueError):
        bank.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([1]))

    with pytest.raises(ValueError):
        bank.update(torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

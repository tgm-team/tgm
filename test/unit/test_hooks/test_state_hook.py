import pytest
import torch

from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, RecencyNeighborHook
from tgm.hooks.base import StatefulHook


@pytest.fixture
def dg():
    edge_index = torch.IntTensor(
        [
            [1, 2],
            [1, 2],
            [2, 3],
        ]
    )
    edge_time = torch.LongTensor([1, 1, 2])
    edge_x = torch.rand(3, 4)

    node_x_time = torch.LongTensor([5, 5, 6])
    node_x_nids = torch.IntTensor([2, 2, 3])
    node_x = torch.rand(3, 3)

    data = DGData.from_raw(
        edge_time=edge_time,
        edge_index=edge_index,
        edge_x=edge_x,
        node_x_time=node_x_time,
        node_x_nids=node_x_nids,
        node_x=node_x,
    )
    return DGraph(data)


@pytest.fixture
def recency_hook(dg):
    return RecencyNeighborHook(
        num_nbrs=[2],
        num_nodes=dg.num_nodes,
        seed_nodes_keys=['edge_src', 'edge_dst'],
        seed_times_keys=['edge_time', 'edge_time'],
    )


class _CounterHook(StatefulHook):
    produces = set()
    requires = set()

    def __init__(self):
        self.counter = 0

    def __call__(self, dg, batch):
        self.counter += 1
        return batch

    def reset_state(self):
        self.counter = 0

    def state_dict(self):
        return {'counter': self.counter}

    def load_state_dict(self, state):
        self.counter = state['counter']


def test_stateful_hook_state_dict_raises_if_not_implemented():
    class ForgottenHook(StatefulHook):
        pass

    with pytest.raises(NotImplementedError):
        ForgottenHook().state_dict()


def test_stateful_hook_load_state_dict_raises_if_not_implemented():
    class ForgottenHook(StatefulHook):
        pass

    with pytest.raises(NotImplementedError):
        ForgottenHook().load_state_dict({})


def test_recency_hook_state_dict_contains_required_keys(dg, recency_hook):
    hm = HookManager(keys=['train'])
    hm.register_shared(recency_hook)
    loader = DGDataLoader(dg, batch_size=1, hook_manager=hm)

    with hm.activate('train'):
        for i, _ in enumerate(loader):
            if i == 1:
                break

    state = recency_hook.state_dict()

    assert '_nbr_ids' in state
    assert '_nbr_times' in state
    assert '_nbr_feats' in state
    assert '_write_pos' in state
    assert '_edge_x_dim' in state
    assert '_need_to_initialize_nbr_feats' in state


def test_recency_hook_state_dict_load_state_dict_roundtrip(dg, recency_hook):
    hm = HookManager(keys=['train'])
    hm.register_shared(recency_hook)
    loader = DGDataLoader(dg, batch_size=1, hook_manager=hm)

    with hm.activate('train'):
        for i, _ in enumerate(loader):
            if i == 1:
                break

    write_pos_before = recency_hook._write_pos.clone()
    nbr_ids_before = recency_hook._nbr_ids.clone()

    state = recency_hook.state_dict()

    recency_hook.reset_state()
    assert not torch.equal(recency_hook._write_pos, write_pos_before)

    recency_hook.load_state_dict(state)

    assert torch.equal(recency_hook._write_pos, write_pos_before)
    assert torch.equal(recency_hook._nbr_ids, nbr_ids_before)


def test_hook_manager_state_dict_saves_stateful_hook():
    hm = HookManager(keys=['train'])
    hook = _CounterHook()
    hook.counter = 42
    hm.register_shared(hook)

    states = hm.state_dict('train')

    assert len(states) >= 1
    saved_counter = list(states.values())[0]['counter']
    assert saved_counter == 42


def test_hook_manager_state_dict_no_duplicate_saves():
    hm = HookManager(keys=['train'])
    hook = _CounterHook()
    hm.register_shared(hook)

    states = hm.state_dict('train')

    assert len(states) == 1


def test_hook_manager_load_state_dict_restores_hook():
    hm = HookManager(keys=['train'])
    hook = _CounterHook()
    hm.register_shared(hook)

    hook.counter = 99
    states = hm.state_dict('train')

    hook.counter = 0
    hm.load_state_dict(states, 'train')

    assert hook.counter == 99


def test_skip_batches_reduces_yielded_count(dg):
    full_batches = list(DGDataLoader(dg, batch_size=1))
    skip_batches = list(DGDataLoader(dg, batch_size=1, skip_batches=1))

    assert len(skip_batches) == len(full_batches) - 1


def test_skip_batches_hook_not_executed(dg):
    hm = HookManager(keys=['train'])
    hook = _CounterHook()
    hm.register('train', hook)

    total = len(list(DGDataLoader(dg, batch_size=1)))

    skip = 1
    loader = DGDataLoader(dg, batch_size=1, hook_manager=hm, skip_batches=skip)

    with hm.activate('train'):
        list(loader)

    assert hook.counter == total - skip

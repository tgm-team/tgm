import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.exceptions import (
    BadHookProtocolError,
    UnresolvableHookDependenciesError,
)
from tgm.hooks import HookManager, StatefulHook, StatelessHook


class MockHook(StatelessHook):
    produces = {'foo'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.time *= 2
        return batch


class MockHookRequires(StatelessHook):
    requires = {'foo'}

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        return batch


class MockHookWithState(StatefulHook):
    has_state: bool = True

    def __init__(self) -> None:
        self.x = 0

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.time *= 2
        return batch

    def reset_state(self) -> None:
        self.x = 1


@pytest.fixture
def dg():
    edge_index = torch.IntTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


def test_str():
    hm = HookManager(keys=['train', 'val'])
    assert isinstance(str(hm), str)

    hm.register_shared(MockHook())
    hm.register('train', MockHook())
    hm.register('train', MockHookWithState())
    hm.register('val', MockHook())
    assert isinstance(str(hm), str)


def test_bad_init_empty_keys():
    with pytest.raises(ValueError):
        _ = HookManager(keys=[])


def test_register():
    hm = HookManager(keys=['foo'])
    hook = MockHook()
    hm.register('foo', hook)

    assert len(hm.keys) == 1
    assert hook in hm._key_to_hooks['foo']
    assert len(hm._key_to_hooks['foo']) == 1


def test_register_multiple():
    hm = HookManager(keys=['train', 'val'])
    hm.register_shared(MockHook())
    hm.register('train', MockHook())
    hm.register('train', MockHookWithState())
    hm.register('val', MockHook())

    assert len(hm.keys) == 2
    assert len(hm._key_to_hooks['train']) == 2
    assert len(hm._key_to_hooks['val']) == 1


def test_register_shared():
    hm = HookManager(keys=['foo'])
    hook = MockHook()
    hm.register_shared(hook)
    assert hook in hm._shared_hooks
    assert len(hm._shared_hooks) == 1


def test_attempt_register_bad_key():
    hm = HookManager(keys=['train'])
    with pytest.raises(KeyError):
        hm.register('foo', MockHook())


def test_attempt_register_bad_hook():
    hm = HookManager(keys=['foo'])
    with pytest.raises(BadHookProtocolError):
        hm.register('foo', object())


def test_attempt_register_shared_bad_hook():
    hm = HookManager(keys=['foo'])
    with pytest.raises(BadHookProtocolError):
        hm.register_shared(object())


def test_attempt_regiser_while_active():
    hm = HookManager(keys=['train'])
    hook = MockHook()
    with hm.activate('train'):
        with pytest.raises(RuntimeError):
            hm.register('train', hook)


def test_attempt_register_shared_while_active():
    hm = HookManager(keys=['train'])
    hook = MockHook()
    with hm.activate('train'):
        with pytest.raises(RuntimeError):
            hm.register_shared(hook)


def test_resolve_hooks_all_hooks():
    h1 = MockHook()
    h2 = MockHookRequires()

    hm = HookManager(keys=['train', 'val'])
    hm.register('train', h2)
    hm.register('train', h1)
    hm.register('val', h2)
    hm.register('val', h1)

    hm.resolve_hooks()
    assert len(hm._key_to_hooks['train']) == 2
    assert len(hm._key_to_hooks['val']) == 2
    assert hm._key_to_hooks['train'].index(h1) < hm._key_to_hooks['train'].index(h2)
    assert hm._key_to_hooks['val'].index(h1) < hm._key_to_hooks['val'].index(h2)


def test_resolve_hooks_by_key():
    h1 = MockHook()
    h2 = MockHookRequires()

    hm = HookManager(keys=['train'])
    hm.register('train', h2)
    hm.register('train', h1)

    hm.resolve_hooks('train')
    assert len(hm._key_to_hooks['train']) == 2
    assert hm._key_to_hooks['train'].index(h1) < hm._key_to_hooks['train'].index(h2)


def test_resolve_hooks_no_solution_no_dag():
    h1 = MockHook()
    h2 = MockHook()
    h1.requires, h1.produces = {'x'}, {'y'}
    h2.requires, h2.produces = {'y'}, {'x'}

    # Cycle-like missing dependency
    hm = HookManager(keys=['train'])
    hm.register('train', h1)
    hm.register('train', h2)

    with pytest.raises(UnresolvableHookDependenciesError):
        hm.resolve_hooks('train')


def test_resolve_hooks_by_key_bad_key():
    hm = HookManager(keys=['train'])
    with pytest.raises(KeyError):
        hm.resolve_hooks('val')


def test_topo_sort_lazy_required(dg):
    h1 = MockHook()
    h2 = MockHookRequires()

    hm = HookManager(keys=['train'])
    hm.register('train', h2)
    hm.register('train', h1)

    # Topo sort should run lazily
    hm.set_active_hooks('train')
    hm.execute_active_hooks(dg, dg.materialize())


def test_topo_sort_lazy_no_solution_missing_requires(dg):
    h = MockHookRequires()

    hm = HookManager(keys=['train'])
    hm.register('train', h)
    hm.set_active_hooks('train')
    with pytest.raises(UnresolvableHookDependenciesError):
        hm.execute_active_hooks(dg, dg.materialize())


def test_topo_sort_cached(dg, monkeypatch):
    hm = HookManager(keys=['train'])

    h1, h2 = MockHook(), MockHook()
    h1.requires, h1.produces = set(), {'x'}
    h2.requires, h2.produces = {'x'}, {'y'}

    hm.register('train', h1)
    hm.register('train', h2)
    call_count = {'n': 0}

    def fake_topo_sort(hooks_list):
        call_count['n'] += 1
        return hooks_list

    monkeypatch.setattr(hm, '_topological_sort_hooks', fake_topo_sort)

    # First call should trigger topo sort
    hm.set_active_hooks('train')
    hm.execute_active_hooks(dg, dg.materialize())
    assert call_count['n'] == 1

    # Second call should reuse cached topo order
    hm.execute_active_hooks(dg, dg.materialize())
    assert call_count['n'] == 1


def test_topo_sort_cached_invalidated(dg, monkeypatch):
    hm = HookManager(keys=['train'])

    h1 = MockHook()
    h1.requires, h1.produces = set(), {'x'}

    hm.register('train', h1)
    call_count = {'n': 0}

    def fake_topo_sort(hooks_list):
        call_count['n'] += 1
        return hooks_list

    monkeypatch.setattr(hm, '_topological_sort_hooks', fake_topo_sort)

    # First execution: topo sort should run
    with hm.activate('train'):
        hm.execute_active_hooks(dg, dg.materialize())
    assert call_count['n'] == 1

    # Register a new hook to invalidate the dirty bit
    h2 = MockHookRequires()
    hm.register('train', h2)

    # Topo sort should run again
    with hm.activate('train'):
        hm.execute_active_hooks(dg, dg.materialize())
    assert call_count['n'] == 2


def test_topo_sort_no_solution_no_dag(dg):
    h1 = MockHook()
    h2 = MockHook()
    h1.requires, h1.produces = {'x'}, {'y'}
    h2.requires, h2.produces = {'y'}, {'x'}

    # Cycle-like missing dependency
    hm = HookManager(keys=['train'])
    hm.register('train', h1)
    hm.register('train', h2)

    hm.set_active_hooks('train')
    with pytest.raises(UnresolvableHookDependenciesError):
        hm.execute_active_hooks(dg, dg.materialize())


def test_set_active_key_bad_key():
    hm = HookManager(keys=['train'])
    with pytest.raises(KeyError):
        hm.set_active_hooks('val')


def execute_active_hooks_empty(dg):
    hm = HookManager(keys=['train'])
    batch = hm.execute_active_hooks(dg, dg.materialize())
    assert isinstance(batch, DGBatch)
    assert batch.src.device == torch.device('cpu')


def execute_active_hooks_keyed(dg):
    hm = HookManager(keys=['train'])
    hook = MockHook()
    hm.register('train', hook)
    hm.set_active_hooks('train')
    exp_batch = dg.materialize()
    exp_batch.time *= 2
    batch = hm.execute_active_hooks(dg, dg.materialize())
    assert batch == exp_batch


def execute_active_hooks_keyed_and_shared(dg):
    hm = HookManager(keys=['train'])
    hook_shared = MockHook()
    hook_keyed = MockHook()
    hm.register_shared(hook_shared)
    hm.register('train', hook_keyed)
    hm.set_active_hooks('train')
    assert len(hm._key_to_hooks['train']) == 2
    exp_batch = dg.materialize()
    exp_batch.time *= 4
    batch = hm.execute_active_hooks(dg, dg.materialize())
    assert batch == exp_batch


def test_reset_state():
    hm = HookManager(keys=['foo'])
    h1 = MockHookWithState()
    assert h1.x == 0
    hm.register_shared(h1)
    hm.reset_state()
    assert h1.x == 1


def test_reset_state_by_key():
    hm = HookManager(keys=['train'])
    h1 = MockHookWithState()
    hm.register('train', h1)
    hm.reset_state('train')
    assert h1.x == 1


def test_attempt_reset_state_bad_key():
    hm = HookManager(keys=['train'])
    with pytest.raises(KeyError):
        hm.reset_state('val')


def test_activate_ctx():
    hm = HookManager(keys=['train', 'val'])
    hm.register('train', MockHook())
    hm.register('val', MockHook())
    with hm.activate('train'):
        assert hm._active_key == 'train'
        with hm.activate('val'):
            assert hm._active_key == 'val'

        assert hm._active_key == 'train'

    assert hm._active_key is None


def test_topo_sort_neg_before_nbr():
    mock_neg_hook, mock_nbr_hook = MockHook(), MockHook()
    mock_neg_hook.requires, mock_neg_hook.produces = set(), {'neg'}
    mock_nbr_hook.requires, mock_nbr_hook.produces = set(), {'nbr_nids'}

    # Register neg first in foo, nbr first in bar
    hm = HookManager(keys=['foo', 'bar'])
    hm.register('foo', mock_neg_hook)
    hm.register('foo', mock_nbr_hook)
    hm.register('bar', mock_nbr_hook)
    hm.register('bar', mock_neg_hook)

    hm.resolve_hooks()
    foo_hooks = hm._key_to_hooks['foo']
    bar_hooks = hm._key_to_hooks['bar']

    # Ensure negatives precede nbrs in both cases
    assert foo_hooks.index(mock_neg_hook) < foo_hooks.index(mock_nbr_hook)
    assert bar_hooks.index(mock_neg_hook) < bar_hooks.index(mock_nbr_hook)

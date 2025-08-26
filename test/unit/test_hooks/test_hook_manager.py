import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import (
    DeduplicationHook,
    HookManager,
    StatefulHook,
    StatelessHook,
)
from tgm.hooks.hooks import DeviceTransferHook


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
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


def test_str():
    hm = HookManager()
    assert isinstance(str(hm), str)

    hm.register_shared(MockHook())
    hm.register('train', MockHook())
    hm.register('train', MockHookWithState())
    hm.register('val', MockHook())
    assert isinstance(str(hm), str)


def test_init_cpu():
    hm = HookManager()
    assert isinstance(hm._device_hook, DeviceTransferHook)
    assert any(isinstance(h, DeduplicationHook) for h in hm._shared_hooks)


@pytest.mark.gpu
def test_init_gpu():
    hm = HookManager(device='cuda')
    assert isinstance(hm._device_hook, DeviceTransferHook)


def test_register():
    hm = HookManager()
    hook = MockHook()
    hm.register('foo', hook)
    assert hook in hm._key_to_hooks['foo']
    assert len(hm._key_to_hooks['foo']) == 2


def test_register_multiple():
    hm = HookManager()
    hm.register_shared(MockHook())
    hm.register('train', MockHook())
    hm.register('train', MockHookWithState())
    hm.register('val', MockHook())

    hm.set_active_hooks('train')
    assert len(hm.get_active_hooks()) == 4  # dedup, shared, requires, state

    hm.set_active_hooks('val')
    assert len(hm.get_active_hooks()) == 3  # dedup, shared, requires


def test_register_shared():
    hm = HookManager()
    hook = MockHook()
    hm.register_shared(hook)
    assert hook in hm._shared_hooks
    assert len(hm._shared_hooks) == 2


def test_attempt_register_bad_hook():
    hm = HookManager()
    with pytest.raises(TypeError):
        hm.register('foo', object())


def test_attempt_register_shared_bad_hook():
    hm = HookManager()
    with pytest.raises(TypeError):
        hm.register_shared(object())


def test_attempt_regiser_while_active():
    hm = HookManager()
    hook = MockHook()
    with hm.activate('train'):
        with pytest.raises(RuntimeError):
            hm.register('train', hook)


def test_attempt_register_shared_while_active():
    hm = HookManager()
    hook = MockHook()
    with hm.activate('train'):
        with pytest.raises(RuntimeError):
            hm.register_shared(hook)


def test_topo_sort_required():
    h1 = MockHook()
    h2 = MockHookRequires()

    hm = HookManager()
    hm.register('train', h1)
    hm.register('train', h2)
    hooks_ordered = hm._key_to_hooks['train']
    assert hooks_ordered.index(h1) < hooks_ordered.index(h2)
    assert len(hm._key_to_hooks['train']) == 3


def test_topo_sort_no_solution_missing_requires():
    h = MockHookRequires()

    hm = HookManager()
    with pytest.raises(ValueError):
        hm.register('train', h)


@pytest.mark.skip(
    'TODO: This test only makes sense if we enable registering multiple hooks at once. '
    'Otherwise, the "missing produced" will trigger an exception before the secon hook can '
    'be registered. Skiping for now, and should reconsider enabling multiple hook registry.'
)
def test_topo_sort_no_solution_no_dag():
    h1 = MockHook()
    h2 = MockHook()
    h1.requires, h1.produces = {'x'}, {'y'}
    h2.requires, h2.produces = {'y'}, {'x'}

    # Cycle-like missing dependency
    hm = HookManager()
    hm.register('train', h1)
    hm.register('train', h2)

    with pytest.raises(ValueError):
        hm._topological_sort_hooks([h1, h2] + hm._shared_hooks)


def test_get_active_hooks():
    hm = HookManager()
    hook = MockHook()
    hm.register('train', hook)
    hm.set_active_hooks('train')
    assert len(hm.get_active_hooks()) == 2  # dedup, and MockHook
    assert any(isinstance(h, DeduplicationHook) for h in hm.get_active_hooks())
    assert any(isinstance(h, MockHook) for h in hm.get_active_hooks())


def test_get_active_hooks_no_active_keys():
    hm = HookManager()
    hook = MockHook()
    hm.register('train', hook)
    with pytest.raises(RuntimeError):
        hm.get_active_hooks()


def execute_active_hooks_empty(dg):
    hm = HookManager()
    batch = hm.execute_active_hooks(dg)
    assert isinstance(batch, DGBatch)
    assert batch.src.device == torch.device('cpu')


@pytest.mark.gpu
def execute_active_hooks_empty_gpu(dg):
    hm = HookManager(device='cuda')
    batch = hm.execute_active_hooks(dg)
    assert isinstance(batch, DGBatch)
    assert batch.src.device == torch.device('cuda')


def execute_active_hooks_keyed(dg):
    hm = HookManager()
    hook = MockHook()
    hm.register('train', hook)
    hm.set_active_hooks('train')
    exp_batch = dg.materialize()
    exp_batch.time *= 2
    batch = hm.execute_active_hooks(dg)
    assert batch == exp_batch


def execute_active_hooks_keyed_and_shared(dg):
    hm = HookManager()
    hook_shared = MockHook()
    hook_keyed = MockHook()
    hm.register_shared(hook_shared)
    hm.register('train', hook_keyed)
    hm.set_active_hooks('train')
    assert len(hm.get_active_hooks()) == 3
    exp_batch = dg.materialize()
    exp_batch.time *= 4
    batch = hm.execute_active_hooks(dg)
    assert batch == exp_batch


def test_reset_state():
    hm = HookManager()
    h1 = MockHookWithState()
    assert h1.x == 0
    hm.register_shared(h1)
    hm.reset_state()
    assert h1.x == 1
    assert hm._device_hook is not None


def test_reset_state_by_key():
    hm = HookManager()
    h1 = MockHookWithState()
    hm.register('train', h1)
    hm.reset_state('train')
    assert h1.x == 1


def test_activate_ctx():
    hm = HookManager()
    hm.register('train', MockHook())
    hm.register('val', MockHook())
    with hm.activate('train'):
        assert hm._active_key == 'train'
        with hm.activate('val'):
            assert hm._active_key == 'val'

        assert hm._active_key == 'train'

    assert hm._active_key is None

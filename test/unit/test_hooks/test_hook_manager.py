from typing import Set

import pytest
import torch

from tgm import DGBatch, DGraph
from tgm.data import DGData
from tgm.hooks import DeduplicationHook, HookManager


class MockHook:
    requires: Set[str] = set()
    produces: Set[str] = set()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.time *= 2
        return batch


class MockHookRequires:
    requires = {'foo'}
    produces: Set[str] = set()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        return batch


@pytest.fixture
def dg():
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data, discretize_time_delta='r')


def test_hook_manager_init_cpu_empty(dg):
    hook = HookManager(dg, hooks=[])
    exp_batch = dg.materialize()
    assert hook(dg) == exp_batch


def test_hook_manager_init_cpu_non_empty(dg):
    hook = HookManager(dg, hooks=[MockHook()])
    exp_batch = dg.materialize()
    exp_batch.time *= 2
    assert hook(dg) == exp_batch


@pytest.mark.gpu
def test_hook_manager_init_gpu_empty(dg):
    dg = dg.to('cuda')

    hook = HookManager(dg, hooks=[])
    assert len(hook.hooks) == 3

    exp_batch = dg.materialize()
    batch = hook(dg)
    torch.testing.assert_close(exp_batch.src, batch.src)
    torch.testing.assert_close(exp_batch.dst, batch.dst)
    torch.testing.assert_close(exp_batch.time, batch.time)


@pytest.mark.gpu
def test_hook_manager_init_gpu_non_empty(dg):
    dg = dg.to('cuda')

    hook = HookManager(dg, hooks=[MockHook()])
    assert len(hook.hooks) == 4

    exp_batch = dg.materialize()
    exp_batch.time *= 2
    batch = hook(dg)
    torch.testing.assert_close(exp_batch.src, batch.src)
    torch.testing.assert_close(exp_batch.dst, batch.dst)
    torch.testing.assert_close(exp_batch.time, batch.time)


def test_hook_manager_bad_hooks(dg):
    with pytest.raises(TypeError):
        _ = HookManager(dg, hooks='foo')
    with pytest.raises(TypeError):
        _ = HookManager(dg, hooks=['foo'])


def test_hook_manager_bad_hook_dependancies(dg):
    with pytest.raises(ValueError):
        _ = HookManager(dg, hooks=[MockHook(), MockHookRequires()])


def test_hook_manager_from_any_none(dg):
    hook = HookManager.from_any(dg, None)
    assert len(hook.hooks) == 1
    assert isinstance(hook.hooks[0], DeduplicationHook)


def test_hook_manager_from_manager(dg):
    hook = HookManager(dg, hooks=[])
    hook2 = HookManager.from_any(dg, hook)
    assert hook is hook2


def test_hook_manager_from_single_hook(dg):
    hook = HookManager.from_any(dg, MockHook())
    assert len(hook.hooks) == 2


def test_hook_manager_from_hook_list(dg):
    hook = HookManager.from_any(dg, [MockHook()])
    assert len(hook.hooks) == 2


def test_hook_manager_from_bad_hook_type(dg):
    with pytest.raises(TypeError):
        _ = HookManager.from_any(dg, 'foo')

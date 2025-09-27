import pytest
import torch

from tgm import DGraph
from tgm.data import DGData
from tgm.hooks import PinMemoryHook


@pytest.fixture
def dg():
    edge_index = torch.IntTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.IntTensor([1, 1, 2, 2])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


def test_hook_dependancies():
    assert PinMemoryHook.requires == set()
    assert PinMemoryHook.produces == set()


def test_hook_reset_state():
    assert PinMemoryHook.has_state == False


@pytest.mark.gpu
def test_pin_memory_hook_cpu(dg):
    # Note: The gpu is not actually used, but torch complains when calling .pin_memory()
    # and no accelerator backend is available in the torch install.
    hook = PinMemoryHook()
    batch = dg.materialize()

    # Add a custom field and ensure it's also pinned
    batch.foo = torch.rand(1, 2)

    processed_batch = hook(dg, batch)
    assert batch == processed_batch
    assert processed_batch.src.is_pinned()
    assert processed_batch.dst.is_pinned()
    assert processed_batch.time.is_pinned()
    assert processed_batch.foo.is_pinned()


@pytest.mark.gpu
def test_pin_memory_hook_gpu(dg):
    hook = PinMemoryHook()
    batch = dg.materialize()
    processed_batch = hook(dg, batch)
    assert batch is processed_batch

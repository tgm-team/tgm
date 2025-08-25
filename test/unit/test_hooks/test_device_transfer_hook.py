import pytest
import torch

from tgm import DGraph
from tgm.data import DGData
from tgm.hooks import DeviceTransferHook


@pytest.fixture
def dg():
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


def test_hook_dependancies():
    assert DeviceTransferHook.requires == set()
    assert DeviceTransferHook.produces == set()


def test_hook_reset_state():
    assert DeviceTransferHook.has_state == False


def test_device_transfer_hook_cpu_cpu(dg):
    hook = DeviceTransferHook('cpu')
    batch = dg.materialize()

    processed_batch = hook(dg, batch)
    assert batch == processed_batch
    assert processed_batch.src.device.type == 'cpu'
    assert processed_batch.dst.device.type == 'cpu'
    assert processed_batch.time.device.type == 'cpu'


@pytest.mark.gpu
def test_device_transfer_hook_cpu_gpu(dg):
    hook = DeviceTransferHook('cuda')
    batch = dg.materialize()

    # Add a custom field and ensure it's also moved
    batch.foo = torch.rand(1, 2)

    processed_batch = hook(dg, batch)
    assert batch == processed_batch
    assert processed_batch.src.device.type == 'cuda'
    assert processed_batch.dst.device.type == 'cuda'
    assert processed_batch.time.device.type == 'cuda'
    assert processed_batch.foo.device.type == 'cuda'


@pytest.mark.gpu
def test_device_transfer_hook_gpu_gpu(dg):
    hook = DeviceTransferHook('cuda')
    batch = dg.materialize()
    batch.src = batch.src.to('cuda')
    batch.dst = batch.dst.to('cuda')
    batch.time = batch.time.to('cuda')

    # Add a custom field and ensure it's also moved
    batch.foo = torch.rand(1, 2, device='cuda')

    processed_batch = hook(dg, batch)
    assert batch == processed_batch
    assert processed_batch.src.device.type == 'cuda'
    assert processed_batch.dst.device.type == 'cuda'
    assert processed_batch.time.device.type == 'cuda'
    assert processed_batch.foo.device.type == 'cuda'

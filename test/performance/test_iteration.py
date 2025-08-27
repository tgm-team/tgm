from functools import partial

import pytest
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from tgm.graph import DGData, DGraph
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.loader import DGDataLoader

from .conftest import DATASETS


def iterate_loader(loader, max_batches=50):
    total_events = 0
    for i, batch in enumerate(loader):
        total_events += len(batch.src)
        if max_batches and i + 1 >= max_batches:
            break
    return total_events


def setup_no_hooks(dg, dataset):
    return None


def setup_dedup_hook(dg, dataset):
    return create_hook_manager(hooks=[])


def setup_random_negs(dg, dataset):
    _, dst, _ = dg.edges
    hook = NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max()))
    return create_hook_manager(hooks=[hook])


def setup_tgb_negs(dg, dataset, sampler_type=None, num_nbrs=None):
    tgb_dataset = PyGLinkPropPredDataset(name=dataset, root='datasets')
    neg_sampler = tgb_dataset.negative_sampler
    tgb_dataset.load_val_ns()
    tgb_dataset.load_test_ns()

    hooks = [TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val')]

    if sampler_type is None:
        return create_hook_manager(hooks)

    if sampler_type == 'recency':
        hooks.append(
            RecencyNeighborHook(
                num_nodes=dg.num_nodes,
                num_nbrs=num_nbrs,
                edge_feats_dim=dg.edge_feats_dim,
            )
        )
    elif sampler_type == 'uniform':
        hooks.append(NeighborSamplerHook(num_nbrs=num_nbrs))
    else:
        raise ValueError(f'Unknown sampler type: {sampler_type}')

    return create_hook_manager(hooks)


HOOK_CONFIGS = {
    'No Hooks': setup_no_hooks,  # Plain iteration
    'Dedup': setup_dedup_hook,  # Basic hook manager,
    'RandomNegatives': setup_random_negs,  # Random negative edges
    'TGBNegatives': setup_tgb_negs,  # TGB negative edges
    'TGBNegatives + UniformNeighborSampler[20]': partial(
        setup_tgb_negs, sampler_type='uniform', num_nbrs=[20]
    ),  # 1 hop, 20nbrs uniform
    'TGBNegatives + RecencyNeighborSampler[20]': partial(
        setup_tgb_negs, sampler_type='recency', num_nbrs=[20]
    ),  # 1 hop, 20nbrs recency
    'TGBNegatives + RecencyNeighborSampler[20, 20]': partial(
        setup_tgb_negs, sampler_type='recency', num_nbrs=[20, 20]
    ),  # 2 hop, 20nbrs recency
}


def create_hook_manager(hooks):
    hm = HookManager(keys=[''])
    for hook in hooks:
        hm.register('', hook)
    hm.set_active_hooks('')
    return hm


def run_epoch(loader):
    total_events = 0
    for batch in loader:
        total_events += len(batch.src)
    return total_events


@pytest.mark.benchmark(group='data_loader_cpu_hooks')
@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('batch_size', [200, 'D'])
@pytest.mark.parametrize('hook_key', list(HOOK_CONFIGS.keys()))
def test_data_loader_cpu_hooks(benchmark, dataset, batch_size, hook_key):
    _, data, _ = DGData.from_tgb(dataset).split()
    dg = DGraph(data, device='cpu')
    hook_manager = HOOK_CONFIGS[hook_key](dg, dataset)

    if isinstance(batch_size, int):
        loader = DGDataLoader(dg, batch_size=batch_size, hook_manager=hook_manager)
    else:
        loader = DGDataLoader(dg, batch_unit=batch_size, hook_maanger=hook_manager)

    result = benchmark(lambda: run_epoch(loader))
    throughput = (dg.num_events / result.mean) / 1e6
    benchmark.extra_info['throughput_events_per_sec'] = throughput

    print(
        f'{dataset.values[0]} | CPU | batch={batch_size} | hooks={hook_key} -> '
        f'{throughput:.2f} M events/sec'
    )


@pytest.mark.gpu
@pytest.mark.benchmark(group='data_loader_gpu_hooks')
@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('batch_size', [200, 'D'])
@pytest.mark.parametrize('hook_key', list(HOOK_CONFIGS.keys()))
def test_data_loader_gpu_hooks(benchmark, dataset, batch_size, hook_key):
    _, data, _ = DGData.from_tgb(dataset).split()
    dg = DGraph(data, device='cuda')
    hook_manager = HOOK_CONFIGS[hook_key](dg, dataset)

    if isinstance(batch_size, int):
        loader = DGDataLoader(dg, batch_size=batch_size, hook_manager=hook_manager)
    else:
        loader = DGDataLoader(dg, batch_unit=batch_size, hook_manager=hook_manager)

    result = benchmark(lambda: run_epoch(loader))
    throughput = (dg.num_events / result.mean) / 1e6
    benchmark.extra_info['throughput_events_per_sec'] = throughput

    print(
        f'{dataset.values[0]} | GPU | batch={batch_size} | hooks={hook_key} -> '
        f'{throughput:.2f} M events/sec'
    )

from functools import partial

import pytest

from tgm import DGraph
from tgm.data import DGDataLoader
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    NeighborSamplerHook,
    RecencyNeighborHook,
    TGBNegativeEdgeSamplerHook,
)

from .conftest import DATASETS


def setup_no_hooks(dg, data, dataset):
    return None


def setup_random_negs(dg, data, dataset):
    dst = dg.edge_dst
    hook = NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max()))
    return create_hook_manager(hooks=[hook])


def setup_tgb_negs(dg, data, dataset, sampler_type=None, num_nbrs=None):
    if dataset.startswith('tgbl'):
        hooks = [TGBNegativeEdgeSamplerHook(dataset_name=dataset, split_mode='val')]
        seed_nodes_keys = ['edge_src', 'edge_dst', 'neg']
        seed_times_keys = ['edge_time', 'edge_time', 'neg_time']
    else:
        hooks = []
        seed_nodes_keys = ['edge_src', 'edge_dst']
        seed_times_keys = ['edge_time', 'edge_time']

    if sampler_type is None:
        return create_hook_manager(hooks)

    if sampler_type == 'recency':
        hooks.append(
            RecencyNeighborHook(
                num_nodes=data.num_nodes,
                num_nbrs=num_nbrs,
                seed_nodes_keys=seed_nodes_keys,
                seed_times_keys=seed_times_keys,
            )
        )
    elif sampler_type == 'uniform':
        hooks.append(
            NeighborSamplerHook(
                num_nbrs=num_nbrs,
                seed_nodes_keys=seed_nodes_keys,
                seed_times_keys=seed_times_keys,
            )
        )
    else:
        raise ValueError(f'Unknown sampler type: {sampler_type}')

    return create_hook_manager(hooks)


HOOK_CONFIGS = {
    'No Hooks': setup_no_hooks,  # Plain iteration
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


@pytest.mark.benchmark(group='data_loader_cpu_hooks')
@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('batch_size', [200, 'D'])
@pytest.mark.parametrize('hook_key', list(HOOK_CONFIGS.keys()))
def test_data_loader_cpu_hooks(
    benchmark, dataset, batch_size, hook_key, preloaded_graphs
):
    if dataset not in preloaded_graphs:
        pytest.skip()

    # Skip temporal batch unit with tgbn-trade which has coarse ("Y") granularity
    if dataset == 'tgbn-trade' and isinstance(batch_size, str):
        pytest.skip()

    full_data = preloaded_graphs[dataset]
    _, data, _ = full_data.split()  # just testing on validation set
    dg = DGraph(data)
    hook_manager = HOOK_CONFIGS[hook_key](dg, full_data, dataset)

    if isinstance(batch_size, int):
        loader = DGDataLoader(dg, batch_size=batch_size, hook_manager=hook_manager)
    else:
        loader = DGDataLoader(dg, batch_unit=batch_size, hook_manager=hook_manager)

    def run_full_loader():
        for _ in loader:
            continue

    benchmark(run_full_loader)

    throughput = (len(dg) / benchmark.stats['mean']) / 1e6
    benchmark.extra_info.update(
        {
            'throughput_M_events_per_sec': throughput,
            'num_events': len(dg),
        }
    )

    print(
        f'{dataset} | CPU | batch={batch_size} | hooks={hook_key} -> '
        f'{throughput:.3f} M events/sec'
    )


@pytest.mark.gpu
@pytest.mark.benchmark(group='data_loader_gpu_hooks')
@pytest.mark.parametrize('dataset', DATASETS)
@pytest.mark.parametrize('batch_size', [200, 'D'])
@pytest.mark.parametrize('hook_key', list(HOOK_CONFIGS.keys()))
def test_data_loader_gpu_hooks(
    benchmark, dataset, batch_size, hook_key, preloaded_graphs
):
    if dataset not in preloaded_graphs:
        pytest.skip()

    # Skip temporal batch unit with tgbn-trade which has coarse ("Y") granularity
    if dataset == 'tgbn-trade' and isinstance(batch_size, str):
        pytest.skip()

    full_data = preloaded_graphs[dataset]
    _, data, _ = full_data.split()  # just testing on validation set
    dg = DGraph(data).to('cuda')

    hook_manager = HOOK_CONFIGS[hook_key](dg, full_data, dataset)

    if isinstance(batch_size, int):
        loader = DGDataLoader(dg, batch_size=batch_size, hook_manager=hook_manager)
    else:
        loader = DGDataLoader(dg, batch_unit=batch_size, hook_manager=hook_manager)

    def run_full_loader():
        for _ in loader:
            continue

    benchmark(run_full_loader)

    throughput = (len(dg) / benchmark.stats['mean']) / 1e6
    benchmark.extra_info.update(
        {
            'throughput_M_events_per_sec': throughput,
            'num_events': len(dg),
        }
    )

    print(
        f'{dataset} | GPU | batch={batch_size} | hooks={hook_key} -> '
        f'{throughput:.3f} M events/sec'
    )

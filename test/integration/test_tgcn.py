import os

import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-trade'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:15:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/tgcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 5"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset_csv', ['test-token.csv'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:15:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_graphprop_pred_mint(slurm_job_runner, dataset_csv):
    data_root = os.environ.get(
        'GRAPH_PROP_PRED_DATA_ROOT', '$ROOT_DIR/examples/graphproppred'
    )
    dataset_path = f'{data_root}/tokens_data/{dataset_csv}'
    cmd = f"""
python "$ROOT_DIR/examples/graphproppred/tgcn.py" \
    --dataset {dataset_path} \
    --device cuda \
    --epochs 5"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.parametrize('batch_time_gran', ['D'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:05:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_graphprop_pred_tgb(slurm_job_runner, dataset, batch_time_gran):
    cmd = f"""
python "$ROOT_DIR/examples/graphproppred/tgcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 5 \
    --batch-time-gran={batch_time_gran}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:15:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_linkproppred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tgcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 5"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

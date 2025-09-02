import os

import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-genre'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/tgcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset_csv', ['test_token.csv'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_graphprop_pred(slurm_job_runner, dataset_csv):
    data_root = os.environ.get(
        'GRAPH_PROP_PRED_DATA_ROOT', '$ROOT_DIR/examples/graphproppred/tokens_data'
    )
    dataset_path = f'{data_root}/{dataset_csv}'
    cmd = f"""
python "$ROOT_DIR/examples/graphproppred/tgcn.py" \
    --path-dataset {dataset_path} \
    --device cuda \
    --epochs 10
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

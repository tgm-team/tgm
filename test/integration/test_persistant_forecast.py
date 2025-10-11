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
    ]
)
def test_persistant_forecast_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/persistant_forecast.py" \
    --dataset {dataset}"""
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
def test_persistant_forecast_graphprop_pred(slurm_job_runner, dataset_csv):
    data_root = os.environ.get(
        'GRAPH_PROP_PRED_DATA_ROOT', '$ROOT_DIR/examples/graphproppred'
    )
    dataset_path = f'{data_root}/tokens_data/{dataset_csv}'
    cmd = f"""
python "$ROOT_DIR/examples/graphproppred/persistant_forecast.py" \
    --path-dataset {dataset_path}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

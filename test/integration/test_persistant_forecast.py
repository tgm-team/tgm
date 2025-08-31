import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-genre'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
    ]
)
def test_persistant_forecast_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/persistant_forecast.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize(
    'path-dataset', ['examples/graphproppred/tokens_data/test_token.csv']
)
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_persistant_forecast_graphprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/graphproppred/persistant_forecast.py" \
    --path-dataset {dataset} \
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

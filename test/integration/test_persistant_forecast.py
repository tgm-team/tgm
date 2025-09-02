import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-trade'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:03:00',
    ]
)
def test_persistant_forecast_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/persistant_forecast.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

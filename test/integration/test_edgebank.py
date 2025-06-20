import pytest


@pytest.mark.slow
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
    ]
)
def test_edgebank(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
    --dataset {dataset}"""
    state, output = slurm_job_runner(
        cmd,
    )
    assert 'Success' in output and state == 'COMPLETED'

    # TODO: Get perf and latency as artifact to upload in CI


@pytest.mark.slow
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
    ]
)
def test_edgebank_tgb(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/TGB/edgebank.py" \
    --dataset {dataset}"""
    state, output = slurm_job_runner(
        cmd,
    )
    assert 'Success' in output and state == 'COMPLETED'

    # TODO: Get perf and latency as artifact to upload in CI

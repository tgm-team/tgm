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
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
    --dataset {dataset}"""
    state, _ = slurm_job_runner(cmd)

    # TODO: Get perf and latency as artifact to upload in CI
    assert state == 'COMPLETED'

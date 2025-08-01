import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
    ]
)
def test_edgebank_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
    ]
)
def test_edgebank_tgb_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/linkproppred/TGB/edgebank.py" \
    --dataset {dataset}"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

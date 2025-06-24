import pytest


@pytest.mark.slow
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_gclstm_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.slow
@pytest.mark.parametrize('dataset', ['tgbl-genre'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_gclstm_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.nodeproppred.dataset import NodePropPredDataset; NodePropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/nodeproppred/gclstm.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

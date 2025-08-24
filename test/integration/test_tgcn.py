import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-trade'])
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
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.nodeproppred.dataset import NodePropPredDataset; NodePropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/nodeproppred/tgcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('path-dataset', ['examples/graphproppred/test_token.csv'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgcn_graphprop_pred(slurm_job_runner, dataset):
    cmd = f"""
echo "Downloading dataset: {dataset}"
#@TODO: Need a script to download the testing dataset
python "$ROOT_DIR/examples/graphproppred/tgcn.py" \
    --path-dataset {dataset} \
    --device cuda \
    --epochs 1
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

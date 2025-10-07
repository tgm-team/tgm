import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:05:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_gcn_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/gcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 5
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-trade'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:05:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_gcn_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/gcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 5
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

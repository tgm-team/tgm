import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_gclstm_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/gclstm.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 15
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-trade'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_gclstm_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/gclstm.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 50
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

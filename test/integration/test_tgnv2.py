import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=1:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgnv2_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tgnv2.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-trade'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=1:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgnv2_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/tgnv2.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-genre'])
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
python "$ROOT_DIR/examples/nodeproppred/tgcn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=3:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tncn_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
TGM_CI_MAX_EVAL_BATCHES_PER_EPOCH=5 \
python "$ROOT_DIR/examples/linkproppred/tncn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

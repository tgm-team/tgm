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
def test_tgn_recency_sampler_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tgn.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1 \
    --sampling recency \
    --n-nbrs 20
    """
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

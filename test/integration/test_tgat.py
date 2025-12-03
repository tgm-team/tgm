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
def test_tgat_linkprop_pred_recency_sampler(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tgat.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1 \
    --sampling recency \
    --n-nbrs 20 20"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


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
def test_tgat_linkprop_pred_uniform_sampler(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tgat.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1 \
    --sampling uniform \
    --n-nbrs 5 5"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tgbn-trade'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=1:00:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgat_nodeprop_pred_recency_sampler(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/tgat.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1 \
    --sampling recency \
    --n-nbrs 20 20"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

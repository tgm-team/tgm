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
def test_tpnet_linkprop_pred(slurm_job_runner, dataset):
    cmd = f"""
TGM_CI_MAX_EVAL_BATCHES_PER_EPOCH=5 \
python "$ROOT_DIR/examples/linkproppred/tpnet.py" \
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
def test_tpnet_nodeprop_pred(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/nodeproppred/tpnet.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

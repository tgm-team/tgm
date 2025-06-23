import pytest


@pytest.mark.slow
@pytest.mark.parametrize('dataset', ['tgbl-wiki'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:10:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgat_recency_sampler(slurm_job_runner, dataset):
    cmd = f"""
echo "Downloading dataset: {dataset}"
echo "y" | python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('{dataset}')"

python "$ROOT_DIR/examples/linkproppred/tgat.py" \
    --dataset {dataset} \
    --device cuda \
    --epochs 1 \
    --sampling recency"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'

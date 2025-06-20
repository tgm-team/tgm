import pytest


@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=8G',
        '--time=0:10:00',
        '--gres=gpu:a100l:1',
    ]
)
def test_tgat_recency_sampler(slurm_job_runner):
    cmd = """
        python "$ROOT_DIR/examples/linkproppred/tgat.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --device cuda \
            --epochs 10 \
            --lr 0.0001 \
            --dropout 0.1 \
            --n-heads 2 \
            --n-nbrs 20 20 \
            --time-dim 100 \
            --embed-dim 100 \
            --sampling recency
        ;;
    """
    state, output = slurm_job_runner(
        cmd,
    )
    assert 'Success' in output and state == 'COMPLETED'

    # TODO: Get perf and latency as artifact to upload in CI

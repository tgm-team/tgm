import pytest


@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=2',
        '--mem=4G',
        '--time=0:10:00',
    ]
)
def test_edgebank(slurm_job_runner):
    cmd = """
        python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --window_ratio 0.15 \
            --pos_prob 1.0 \
            --memory_mode unlimited
    """
    state, output = slurm_job_runner(
        cmd,
    )
    assert 'Success' in output and state == 'COMPLETED'

    # TODO: Get perf and latency as artifact to upload in CI


def test_edgebank_tgb(slurm_job_runner):
    cmd = """
        python "$ROOT_DIR/examples/linkproppred/TGB/edgebank.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --window_ratio 0.15 \
            --pos_prob 1.0 \
            --memory_mode unlimited
    """
    state, output = slurm_job_runner(
        cmd,
    )
    assert 'Success' in output and state == 'COMPLETED'

    # TODO: Get perf and latency as artifact to upload in CI

import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'slurm(resources): Run test via sbatch with given Slurm resources.'
    )


@pytest.fixture
def slurm_job_runner(tmp_path: Path):
    def run(cmd, slurm_args=None):
        slurm_args = slurm_args or []
        job_id = str(uuid.uuid4())[:8]
        slurm_out = tmp_path / f'slurm-{job_id}.out'
        slurm_err = tmp_path / f'slurm-{job_id}.err'

        job_script = f"""#!/bin/bash

        #SBATCH --job-name=pytest-{job_id}
        #SBATCH --output={slurm_out}
        #SBATCH --error={slurm_err}
        {chr(10).join(slurm_args)}
        {cmd}
        """

        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh') as f:
            f.write(job_script)
            script_path = f.name

        output = subprocess.check_output(['sbatch', script_path]).decode()
        job_number = output.strip().split()[-1]

        while True:
            result = subprocess.run(
                ['sacct', '-j', job_number, '--format=State', '--noheader'],
                capture_output=True,
                text=True,
            )
            state = result.stdout.strip().split()[0]
            if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(10)

        output_text = slurm_out.read_text() if slurm_out.exists() else ''
        return state, output_text

    return run


"""
#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS=$(cat "$ROOT_DIR/jobs/configs/datasets.txt")
METHODS=$(cat "$ROOT_DIR/jobs/configs/methods.txt")
SEEDS="0 1 2 3 4"

mkdir -p "$ROOT_DIR/jobs/logs"

get_slurm_resources() {
    local method=$1
    case "$method" in
        edgebank)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:10:00"
            ;;
        tgat)
            echo "--partition=main --cpus-per-task=2 --mem=8G --time=3:00:00 --gres=gpu:a100l:1"
            ;;
        *)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=1:00:00"
            ;;
    esac
}

for METHOD in $METHODS; do
    for DATASET in $DATASETS; do
        for SEED in $SEEDS; do
            RESOURCES=$(get_slurm_resources "$METHOD")
            JOB_NAME="${METHOD}_${DATASET}_${SEED}"
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)

            echo "Submitting: $JOB_NAME"
            sbatch  --job-name="${JOB_NAME}" \
                --output="$ROOT_DIR/jobs/logs/${JOB_NAME}_${TIMESTAMP}.out" \
                --error="$ROOT_DIR/jobs/logs/${JOB_NAME}_${TIMESTAMP}.err" \
                $RESOURCES \
                --wrap="bash $ROOT_DIR/jobs/scripts/run_method.sh $METHOD $DATASET $SEED"
        done
    done
done
"""

"""
#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "$ROOT_DIR/.env"

METHOD=$1
DATASET=$2
SEED=$3

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_NODELIST"
echo "Num CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-None}"
echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Start Time: $(date)"
echo "Method: $METHOD"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "===================="

case "$METHOD" in
    edgebank)
        python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --window_ratio 0.15 \
            --pos_prob 1.0 \
            --memory_mode unlimited
        ;;

    tgat)
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

    *)
        echo "Error: Unknown method '$METHOD'"
        exit 1
        ;;
esac
"""

#!/bin/bash
#SBATCH --job-name=tgm_benchmarks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=00:30:00

set -euo pipefail

module load python/3.10
module load cudatoolkit/11.7

cd "$GITHUB_WORKSPACE"

export TGM_CI_PERF_LOG_BASE=${TGM_CI_PERF_LOG_BASE:-/home/$USER/tgm_ci_perf}
mkdir -p "$TGM_CI_PERF_LOG_BASE"

echo "[$(date)] Starting TGM performance tests on $(hostname)"
echo "Logs will be in $TGM_CI_PERF_LOG_BASE"

# Run the performance tests
./scripts/run_perf_tests.sh --gpu --small --medium \
    >"$TGM_CI_PERF_LOG_BASE/benchmarks-$SLURM_JOB_ID.out" \
    2>"$TGM_CI_PERF_LOG_BASE/benchmarks-$SLURM_JOB_ID.err"

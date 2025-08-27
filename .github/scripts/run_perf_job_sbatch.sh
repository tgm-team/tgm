#!/bin/bash
#SBATCH --job-name=tgm_benchmarks
#SBATCH --output=$HOME/tgm_ci_perf/benchmarks.out
#SBATCH --error=$HOME/tgm_ci_perf/benchmarks.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=00:30:00

set -euo pipefail

module load python/3.10
module load cudatoolkit/11.7

cd "$GITHUB_WORKSPACE"

export TGM_CI_PERF_LOG_BASE=${TGM_CI_PERF_LOG_BASE:-$HOME/tgm_ci_perf}
mkdir -p "$TGM_CI_PERF_LOG_BASE"

# Run the performance tests
./scripts/run_perf_tests.sh --gpu --all

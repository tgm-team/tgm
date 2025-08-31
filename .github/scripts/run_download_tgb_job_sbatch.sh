#!/bin/bash
#SBATCH --job-name=download_tgb
#SBATCH --output=$HOME/tgm_ci_perf/download_tgb.out
#SBATCH --error=$HOME/tgm_ci_perf/download_tgb.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

cd "$GITHUB_WORKSPACE"

# Download the tgb datasets
./scripts/download_tgb_datasets.sh

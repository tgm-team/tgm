#!/bin/bash
#SBATCH --job-name=download_tgb
#SBATCH --output=$HOME/tgm_ci/download_tgb.out
#SBATCH --error=$HOME/tgm_ci/download_tgb.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

DATA_ROOT="$1"

cd "$GITHUB_WORKSPACE"

# Download the tgb datasets
./scripts/download_tgb_datasets.sh "$DATA_ROOT"

#!/bin/bash
#SBATCH --job-name=download_tgb
#SBATCH --output=/home/%u/tgm_ci/download_tgb-%j.out
#SBATCH --error=/home/%u/tgm_ci/download_tgb-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

LOG_DIR="/home/$USER/tgm_ci"
mkdir -p "$LOG_DIR"

DATA_ROOT="$1"

cd "$GITHUB_WORKSPACE"

echo "[$(date)] Starting TGB dataset download into $DATA_ROOT on $(hostname)"
echo "Logs: $LOG_DIR"

# Download the tgb datasets
./scripts/download_tgb_datasets.sh "$DATA_ROOT"

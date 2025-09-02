#!/bin/bash
#SBATCH --job-name=download_tgb
#SBATCH --cpus-per-task=1
#SBATCH --partition=long-cpu
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

cd "$GITHUB_WORKSPACE"

# Logs inside real home
LOG_DIR="$REAL_HOME/tgm_ci"
mkdir -p "$LOG_DIR"

DATA_ROOT="$1"

echo "[$(date)] Starting TGB dataset download into $DATA_ROOT on $(hostname)"
echo "Logs: $LOG_DIR"

# Download the tgb datasets
./scripts/download_tgb_datasets.sh "$DATA_ROOT" >"$LOG_DIR/download-$SLURM_JOB_ID.out" 2>"$LOG_DIR/download-$SLURM_JOB_ID.err"

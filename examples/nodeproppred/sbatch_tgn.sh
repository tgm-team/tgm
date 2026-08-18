#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=tgn_ablation_logs/%x_%j.out
#SBATCH --error=tgn_ablation_logs/%x_%j.err

# Usage: sbatch --job-name=tgn_<config>_s<seed> examples/nodeproppred/sbatch_tgn.sh <strict|parity> <seed>
# Run from the tgm repo root. Results (Best Validation / Best Test) are at the
# end of the .out file.

CONFIG=${1:-parity}
SEED=${2:-1}

module load python/3.10
cd "$SLURM_SUBMIT_DIR"
mkdir -p tgn_ablation_logs

EXTRA=""
if [ "$CONFIG" = "parity" ]; then
  EXTRA="--tgb-parity"
fi

.venv/bin/python -u examples/nodeproppred/tgn.py \
  --dataset tgbn-trade --device cuda --epochs 50 --seed "$SEED" $EXTRA

#!/bin/bash
#SBATCH --job-name=
#SBATCH --output=logs/output/**-%A.%a.out
#SBATCH --error=logs/error/**-%A.%a.err
#SBATCH --time=0-48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --partition=
#SBATCH --constraint=80gb               # ensure A100 80GB

# Cd to project root
cd ~/scratch/tgmlib

# Create directories for logs
mkdir -p logs/output logs/error


git fetch
git pull
git checkout bao/ICLR-experimennts
uv sync

USE_BSIZE=False
BUNIT="m"
for seed in 720 721 722; do
  uv run python experiments/linkproppred/tgat.py \
    --use_bsize="$USE_BSIZE" \
    --experiment_name=type_dataload_effect_bunit \
    --dataset=tgbl-wiki \
    --seed="$seed" \
    --bunit="$BUNIT"
done



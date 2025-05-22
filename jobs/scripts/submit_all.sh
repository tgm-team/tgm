#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS=$(cat "$ROOT_DIR/jobs/configs/datasets.txt")
METHODS=$(cat "$ROOT_DIR/jobs/configs/methods.txt")
SEEDS="0 1 2"

mkdir -p "$ROOT_DIR/jobs/logs"

for METHOD in $METHODS; do
    for DATASET in $DATASETS; do
        for SEED in $SEEDS; do
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            echo "Submitting: $METHOD | $DATASET | $SEED"
            sbatch  --job-name=${METHOD}_${DATASET}_${SEED} \
                    --output="$ROOT_DIR/jobs/logs/${METHOD}_${DATASET}_${SEED}_${TIMESTAMP}.out" \
                    --error="$ROOT_DIR/jobs/logs/${METHOD}_${DATASET}_${SEED}_${TIMESTAMP}.err" \
                    --partition=main \
                    --mem=4G \
                    --cpus-per-task=2 \
                    --wrap="bash $ROOT_DIR/jobs/scripts/run_method.sh $METHOD $DATASET $SEED"
        done
    done
done

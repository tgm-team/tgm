#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS=$(cat "$ROOT_DIR/jobs/configs/datasets.txt")
METHODS=$(cat "$ROOT_DIR/jobs/configs/methods.txt")
SEEDS="0 1 2"

mkdir -p "$ROOT_DIR/jobs/logs"

get_slurm_resources() {
    local method=$1
    case "$method" in
        edgebank)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:10:00"
            ;;
        tgat)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:30:00 --gres=gpu:a100l:1"
            ;;
        tgn)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:30:00 --gres=gpu:a100l:1"
            ;;
        gcn)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:30:00 --gres=gpu:a100l:1"
            ;;
        gclstm)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:30:00 --gres=gpu:a100l:1"
            ;;
        *)
            echo "--partition=main --cpus-per-task=2 --mem=4G --time=0:30:00"
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

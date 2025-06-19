#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "$ROOT_DIR/.env"

METHOD=$1
DATASET=$2
SEED=$3

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_NODELIST"
echo "Num CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-None}"
echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Start Time: $(date)"
echo "Method: $METHOD"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "===================="

case "$METHOD" in
    edgebank)
        python "$ROOT_DIR/examples/linkproppred/edgebank.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --window_ratio 0.15 \
            --pos_prob 1.0 \
            --memory_mode unlimited
        ;;

    tgat)
        python "$ROOT_DIR/examples/linkproppred/tgat.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --device cuda \
            --epochs 10 \
            --lr 0.0001 \
            --dropout 0.1 \
            --n-heads 2 \
            --n-nbrs 20 20 \
            --time-dim 100 \
            --embed-dim 100 \
            --sampling recency
        ;;

    tgn)
        python "$ROOT_DIR/examples/linkproppred/tgn.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --device cuda \
            --epochs 10 \
            --lr 0.0001 \
            --dropout 0.1 \
            --n-heads 2 \
            --n-nbrs 20 \
            --time-dim 100 \
            --embed-dim 100 \
            --sampling recency
        ;;

    gcn)
        python "$ROOT_DIR/examples/linkproppred/gcn.py" \
            --seed $SEED \
            --dataset $DATASET \
            --device cuda \
            --embed-dim 128 \
            --epochs 10 \
            --lr 0.0001 \
            --dropout 0.1 \
            --n-layers 2 \
            --time-gran h
        ;;

    gclstm)
        python "$ROOT_DIR/examples/linkproppred/gclstm.py" \
            --seed $SEED \
            --dataset $DATASET \
            --device cuda \
            --embed-dim 128 \
            --epochs 10 \
            --lr 0.0001 \
            --n-layers 2 \
            --time-gran h
        ;;

    *)
        echo "Error: Unknown method '$METHOD'"
        exit 1
        ;;
esac

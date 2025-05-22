#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "$ROOT_DIR/.env"

METHOD=$1
DATASET=$2
SEED=$3

echo "Running $METHOD on $DATASET with seed $SEED"

case "$METHOD" in
    edgebank)
        python "$ROOT_DIR/examples/edgebank.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --window_ratio 0.15 \
            --pos_prob 1.0 \
            --memory_mode unlimited
        ;;

    tgat)
        python "$ROOT_DIR/examples/tgat.py" \
            --seed $SEED \
            --dataset $DATASET \
            --bsize 200 \
            --epochs 10 \
            --lr 0.0001 \
            --dropout 0.1 \
            --n-heads 2 \
            --n-nbrs [20] \
            --time-dim 100 \
            --embed-dim 100 \
            --sampling recency
        ;;

    *)
        echo "Error: Unknown method '$METHOD'"
        exit 1
        ;;
esac

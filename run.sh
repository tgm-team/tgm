#!/bin/bash

module load cudatoolkit/11.7
module load python/3.10
source .venv/bin/activate

python examples/linkproppred/TGB/tgat.py \
    --dataset tgbl-wiki \
    --bsize 1 \
    --device cpu \
    --epochs 1 \
    --lr 0.0001 \
    --dropout 0 \
    --n-heads 2 \
    --n-nbrs 1 \
    --time-dim 1 \
    --embed-dim 1 \
    --sampling recency

#!/bin/bash

set -euo pipefail
source .env

datasets=(
    "tgbl-wiki"
    "tgbl-lastfm"
    "tgbl-subreddit"
    "tgbl-enron"
    "tgbl-uci"
)

for dataset in "${datasets[@]}"; do
    echo "Downloading dataset: $dataset"
    python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('$dataset')"
    echo "Ok."
done

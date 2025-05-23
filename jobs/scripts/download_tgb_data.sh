#!/bin/bash
set -euo pipefail

# The following assumes we are two directories deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS=$(cat "$ROOT_DIR/jobs/configs/datasets.txt")

source "$ROOT_DIR/.env"

for dataset in $DATASETS; do
    echo "Downloading dataset: $dataset"
    echo "y" | python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset('$dataset')"
    echo "Ok."
done

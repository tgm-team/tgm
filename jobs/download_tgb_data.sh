#!/bin/bash

set -euo pipefail
source .env

datasets=(
    "tgbl-wiki"
    "tgbl-lastfm"
    "tgbl-subreddit"
)

for dataset in "${datasets[@]}"; do
    echo "Downloading dataset: $dataset"
    echo "Ok."
done

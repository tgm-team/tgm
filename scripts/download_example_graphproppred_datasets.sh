#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

declare -A DATASETS_URLS
DATASETS_URLS[tokens_data]="https://raw.githubusercontent.com/Jacob-Chmura/mint-datasets/master/fetch.sh"

print_usage() {
    echo "Usage: $0 DATA_ROOT"
    echo
    echo "Pre-download graph property prediction sample datasets and copy them into the example/graphproppred"
    echo
    echo "Arguments:"
    echo "  DATA_ROOT   Required path to store datasets."
    echo
}

parse_args() {
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        print_usage
        exit 0
    fi

    if [[ -z "${1:-}" ]]; then
        echo "Error: DATA_ROOT argument is required" >&2
        print_usage
        exit 1
    fi

    DATA_ROOT="$1"
    echo "Storing datasets at: $DATA_ROOT"
}

download_dataset() {
    local dataset="$1"
    local url="${DATASETS_URLS[$dataset]}"

    echo "Downloading dataset: $dataset from $url"
    curl -LsSf $url | bash
}

move_dataset_to_data_root() {
    local dataset="$1"
    local dest_dir="$DATA_ROOT/$dataset"
    local src_dir="$dataset"

    if [ -d "$dest_dir" ]; then
        echo "Dataset $dataset already exists at $dest_dir, skipping move."
        rm -rf $src_dir
    else
        echo "Moving $dataset -> $dest_dir"
        mv -v "$src_dir" "$dest_dir"
    fi
}

main() {
    parse_args "$@"

    for dataset in "${!DATASETS_URLS[@]}"; do
        if [ -d "$DATA_ROOT/$dataset" ]; then
            echo "Dataset $dataset already present in $DATA_ROOT, skipping download."
        else
            download_dataset "$dataset"
            move_dataset_to_data_root "$dataset"
        fi
    done

    echo "All sample datasets are available in example/graphproppred"
}

main "$@"

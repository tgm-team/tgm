#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

VENV_TGB_DIR=".venv/lib/python3.10/site-packages/tgb/datasets"

DATASETS=(
    "tgbl_wiki"
    "tgbn_trade"
    #"tgbn_genre"
    #"tgbl_coin"
    #"tgbl_flight" TODO: Start working with the large graphs
    #"tgbn_reddit"
)

print_usage() {
    echo "Usage: $0 DATA_ROOT"
    echo
    echo "Pre-download TGB datasets and copy them into the venv."
    echo
    echo "Arguments:"
    echo "  DATA_ROOT   Required path to store datasets."
    echo
    echo "Environment:"
    echo "  If .venv is missing, it will be created automatically with:"
    echo "    uv venv .venv && uv sync --group dev"
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
    mkdir -p "$DATA_ROOT"
}

check_uv_install() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

setup_venv_if_missing() {
    if [ ! -d ".venv" ]; then
        echo ".venv not found. Setting up environment. This might take a while..."
        uv venv .venv
    fi
    uv sync --group dev
}

download_dataset() {
    local dataset="$1"
    local dataset_name="${dataset//_/-}" # 'tgbl_wiki' -> 'tgbl-wiki'
    echo "Downloading dataset: $dataset_name"

    if [[ "$dataset" == tgbl_* ]]; then
        .venv/bin/python -c "from tgb.linkproppred.dataset import LinkPropPredDataset as DS; DS(name='$dataset_name')"
    elif [[ "$dataset" == tgbn_* ]]; then
        .venv/bin/python -c "from tgb.nodeproppred.dataset import NodePropPredDataset as DS; DS(name='$dataset_name')"
    else
        echo "Unknown TGB dataset: $dataset" >&2
        exit 1
    fi
}

move_dataset_to_data_root() {
    local dataset="$1"
    local src_dir="$VENV_TGB_DIR/$dataset"
    local dest_dir="$DATA_ROOT/$dataset"

    if [ -d "$dest_dir" ]; then
        echo "Dataset $dataset already exists at $dest_dir, skipping move."
    else
        echo "Moving $dataset -> $dest_dir"
        mv -v "$src_dir" "$dest_dir"
    fi
}

copy_dataset_to_venv() {
    local dataset="$1"
    local src_dir="$DATA_ROOT/$dataset"
    local dest_dir="$VENV_TGB_DIR/$dataset"

    mkdir -p "$VENV_TGB_DIR"

    if [ -d "$dest_dir" ]; then
        echo "Dataset $dataset already exists in venv at $dest_dir, skipping copy."
    else
        echo "Copying $dataset -> $dest_dir"
        cp -rv "$src_dir" "$dest_dir/"
    fi
}

main() {
    parse_args "$@"
    check_uv_install
    setup_venv_if_missing

    for dataset in "${DATASETS[@]}"; do
        if [ -d "$DATA_ROOT/$dataset" ]; then
            echo "Dataset $dataset already present in $DATA_ROOT, skipping download."
        else
            download_dataset "$dataset"
            move_dataset_to_data_root "$dataset"
        fi

        copy_dataset_to_venv "$dataset"
        echo
    done

    echo "All TGB datasets prepared and available in venv."
}

main "$@"

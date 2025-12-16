#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"


DATASETS=("tokens_data")

get_dataset_url() {
    case "$1" in
        tokens_data)
            echo "https://raw.githubusercontent.com/Jacob-Chmura/mint-datasets/master/fetch.sh"
            ;;
        *)
            ;;  # return empty for unknown dataset
    esac
}

print_usage() {
    echo "Usage: $0 DATA_ROOT"
    echo
    echo "Pre-download graph property prediction sample datasets and copy them into DATA_ROOT"
    echo
    echo "Arguments:"
    echo "  DATA_ROOT   Required path to store datasets."
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

download_dataset() {
    local dataset="$1"
    local url="$(get_dataset_url "$dataset")"

    if [ -z "$url" ]; then
        echo "Error: unknown dataset: '$dataset'" >&2
        exit 1
    fi

    echo "Downloading dataset: $dataset from $url"
    curl -LsSf $url | bash
}


realpath_compat() {
    local path="$1"
    if command -v realpath >/dev/null 2>&1; then
        # realpath exists, use it if the path exists
        if [ -e "$path" ] || [ -L "$path" ]; then
            realpath "$path"
            return
        fi
    fi
    # fallback: absolute path using pwd
    # https://apple.stackexchange.com/questions/450035/is-the-unix-realpath-command-distributed-with-macos-ventura
    echo "$PWD/$path"
}

move_dataset_to_data_root() {
    local dataset="$1"
    local dest_dir="$DATA_ROOT/$dataset"
    local src_dir="$dataset"

    local abs_src=$(realpath_compat "$src_dir")
    local abs_dest=$(realpath_compat "$dest_dir")
    if [ "$abs_src" = "$abs_dest" ]; then
        echo "Source and destination are the same ($abs_src), skipping move."
        return
    fi

    if [ "$src_dir" = "$dest_dir" ]; then
        echo "Source and destination are the same ($src_dir), skipping move."
        return
    fi

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

    for dataset in "${DATASETS[@]}"; do
        if [ -d "$DATA_ROOT/$dataset" ]; then
            echo "Dataset $dataset already present in $DATA_ROOT, skipping download."
        else
            download_dataset "$dataset"
            move_dataset_to_data_root "$dataset"
        fi
    done

    echo "All sample datasets are available in $DATA_ROOT/tokens_data"
}

main "$@"

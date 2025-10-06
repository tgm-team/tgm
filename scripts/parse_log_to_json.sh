#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

print_usage() {
    echo "Usage: $0 LOG_FILE [JSON_SAVE_PATH]"
    echo
    echo "Parse the structured logs from LOG_FILE and write to JSON_SAVE_PATH."
    echo
    echo "Arguments:"
    echo "  LOG_FILE        Required, the input log file"
    echo "  JSON_SAVE_PATH  Optional, the output file path (default: same directory as log file)"
}

check_uv_install() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

parse_structured_logs() {
    local log_file="$1"
    local json_save_path="$2"

    # Ensure paths are absolute
    log_file="$(realpath "$log_file")"
    json_save_path="$(realpath "$json_save_path")"
    echo "Parsing JSON logs from $log_file → $json_save_path"
    uv run python "$PROJECT_ROOT/tools/log_parser.py" "$log_file" "$json_save_path"
}

main() {
    check_uv_install

    if [[ $# -lt 1 ]]; then
        print_usage
        exit 1
    fi

    local log_file="$1"
    local json_save_path="${2:-${log_file%.log}.json}" # replace .log suffix with .json if present
    parse_structured_logs "$log_file" "$json_save_path"
}

main "$@"

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
    echo "Parsing JSON logs from $log_file → $json_save_path"

    uv run python - <<'EOF' "$log_file" "$json_save_path"
import sys, json, pathlib

log_file, out_file = sys.argv[1], sys.argv[2]
out_path = pathlib.Path(out_file)

with open(log_file, "rb") as f:
    lines = [line.decode() for line in f.read().splitlines()]
    json_lines = [line for line in lines if line.endswith("}")]
    objs = [
        json.loads(line[line.find("{") : line.rfind("}") + 1])
        for line in json_lines
    ]

with out_path.open("w") as f:
    json.dump(objs, f, indent=2)

print(f"✅ Parsed {len(objs)} JSON objects into {out_file}")
EOF

}

main() {
    check_uv_install

    if [[ $# -lt 1 ]]; then
        print_usage
        exit 1
    fi

    local log_file="$1"
    local json_save_path="${2:-$log_file.json}"
    parse_structured_logs "$log_file" "$json_save_path"
}

main "$@"

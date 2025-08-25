#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

print_usage() {
    echo "Usage: $0 [--gpu]"
    echo
    echo "Run performance tests locally."
    echo
    echo "Options:"
    echo "  --gpu    Enable GPU tests (default: CPU-only)"
    echo
    echo "Environment:"
    echo "  If .venv is missing, it will be created automatically with:"
    echo "    uv venv .venv && uv sync --group dev"
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
        uv sync --group dev
    fi
}

run_perf_tests() {
    local marker_arg="-m benchmark"
    if [[ "$GPU" == "1" ]]; then
        marker_arg="-m benchmark and gpu"
    fi

    echo "Running performance tests..."
    .venv/bin/pytest test \
        "$marker_arg" \
        --benchmark-only \
        --benchmark-json="$output_json" \
        -q -vvv

    echo "Performance tests completed successfully."
    echo "Results saved to $output_json"
}

parse_args() {
    GPU=0
    if [[ "${1:-}" == "--gpu" ]]; then
        GPU=1
    elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        print_usage
        exit 0
    fi
}

main() {
    check_uv_install
    parse_args "$@"
    setup_venv_if_missing
    run_perf_tests
}

main "$@"

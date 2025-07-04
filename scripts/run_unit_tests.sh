#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

print_usage() {
    echo "Usage: $0 [--gpu]"
    echo
    echo "Run unit tests with optional GPU markers."
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

run_tests() {
    local marker_arg="-m not gpu"
    if [[ "$GPU" == "1" ]]; then
        marker_arg=""
    fi

    echo "Running unit tests..."
    .venv/bin/pytest test test/unit \
        "$marker_arg" \
        --cov=tgm \
        --cov-report=term-missing:skip-covered \
        -q -vvv
    echo "Tests completed successfully."
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
    run_tests
}

main "$@"

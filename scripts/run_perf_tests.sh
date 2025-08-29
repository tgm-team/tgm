#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

PERF_LOG_DIR="${PROJECT_ROOT}/.benchmarks"
mkdir -p "$PERF_LOG_DIR"

print_usage() {
    echo "Usage: $0 [--gpu] [--all]"
    echo
    echo "Run performance tests locally with pytest-benchmark."
    echo
    echo "Options:"
    echo "  --gpu       Enable GPU benchmarks (default: CPU-only)"
    echo "  --small     Run benchmarks on small datasets (default if no flag given)"
    echo "  --medium    Run benchmarks on medium datasets"
    echo "  --large     Run benchmarks on large datasets"
}

check_uv_install() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

setup_venv_if_missing() {
    if [ ! -d ".venv" ]; then
        echo ".venv not found. Setting up environment..."
        uv venv .venv
    fi
    uv sync --group dev
}

run_perf_tests() {
    local marker_arg="-m benchmark"
    if [[ "$GPU" == "0" ]]; then
        marker_arg="$marker_arg and not gpu"
    fi

    if [[ "${#SIZES[@]}" -eq 0 ]]; then
        # default: small only
        marker_arg="$marker_arg and small"
    else
        marker_arg="$marker_arg and (${SIZES[*]})"
    fi

    local timestamp
    timestamp=$(date '+%Y%m%d-%H%M%S')
    local output_json="$PERF_LOG_DIR/perf_$timestamp.json"

    echo "Running performance tests with marker_arg: $marker_arg"
    .venv/bin/pytest test test/performance \
        "$marker_arg" \
        --benchmark-only \
        --benchmark-json="$output_json" \
        -q -vvv
    echo "Performance tests completed successfully."
    echo "Results saved to $output_json"
}

parse_args() {
    GPU=0
    SIZES=()

    for arg in "$@"; do
        case "$arg" in
            --gpu) GPU=1 ;;
            --small) SIZES+=("small") ;;
            --medium) SIZES+=("medium") ;;
            --large) SIZES+=("large") ;;
            -h|--help) print_usage; exit 0 ;;
            *) echo "Unknown argument: $arg"; print_usage; exit 1 ;;
        esac
    done
}

main() {
    check_uv_install
    parse_args "$@"
    setup_venv_if_missing
    run_perf_tests
}

main "$@"

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

    sizes=()
    [[ "$RUN_SMALL" == "1" ]] && sizes+=("small")
    [[ "$RUN_MEDIUM" == "1" ]] && sizes+=("medium")
    [[ "$RUN_LARGE" == "1" ]] && sizes+=("large")

    # default to small if no sizes specified
    if [ ${#sizes[@]} -eq 0 ]; then
        sizes=("small")
    fi

    # join sizes with "or"
    sizes_expr="${sizes[0]}"
    for s in "${sizes[@]:1}"; do
        sizes_expr="$sizes_expr or $s"
    done

    marker_arg="$marker_arg and ($sizes_expr)"

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
    RUN_SMALL=0
    RUN_MEDIUM=0
    RUN_LARGE=0

    for arg in "$@"; do
        case "$arg" in
            --gpu) GPU=1 ;;
            --small) RUN_SMALL=1 ;;
            --medium) RUN_MEDIUM=1 ;;
            --large) RUN_LARGE=1 ;;
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

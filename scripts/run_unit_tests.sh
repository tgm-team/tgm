#!/usr/bin/env bash
set -eou pipefail

print_usage() {
    echo "Usage: $0 [--gpu]"
    echo
    echo "Run unit tests with coverage."
    echo
    echo "Options:"
    echo "  --gpu     Include GPU tests (default is CPU-only)."
    echo
    echo "Requirements:"
    echo "  uv installed (https://docs.astral.sh/uv/getting-started/installation/)"
}

check_uv_install() {
    echo -n "Checking uv install... "
    if command -v uv >/dev/null 2>&1; then
        echo "Ok."
    else
        echo "not found!" >&2
        echo "Please install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
        exit 1
    fi
}

run_tests() {
    local marker_arg
    if [[ "$INCLUDE_GPU" == "false" ]]; then
        echo "Running unit tests (CPU-only)..."
        marker_arg="-m not gpu"
    else
        echo "Running unit tests (CPU + GPU)..."
        marker_arg=""
    fi

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$SCRIPT_DIR/.."
    cd "$PROJECT_ROOT"

    echo "Resolving environment from $PROJECT_ROOT/pyproject.toml ..."
    uv venv .venv
    uv sync --group dev

    .venv/bin/pytest test test/unit \
        "$marker_arg" \
        --cov=tgm \
        --cov-report=term-missing:skip-covered \
        -q -vvv

    echo "Tests completed successfully."
}

main() {
    INCLUDE_GPU="false"

    for arg in "$@"; do
        case $arg in
            --gpu)
                INCLUDE_GPU="true"
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $arg" >&2
                print_usage
                exit 1
                ;;
        esac
    done

    check_uv_install
    run_tests
}

main "$@"

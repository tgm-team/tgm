#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

print_usage() {
    echo "Usage: $0"
    echo
    echo "Build and serve documentation locally."
    echo "  --build-only    Build docs exit instead of serving"
    echo
    echo "Environment:"
    echo "  If .venv is missing, it will be created automatically with:"
    echo "    uv venv .venv && uv sync --group docs"
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
    uv sync --group docs
}

build_docs() {
    echo "Building documentation..."
    uv run mkdocs build --strict
}

serve_docs() {
    echo "Serving documentation..."
    uv run mkdocs serve --strict

}

main() {
    check_uv_install
    setup_venv_if_missing

    BUILD_ONLY=false
    for arg in "$@"; do
        case $arg in
            --build-only)
                BUILD_ONLY=true
                ;;
        esac
    done

    if [ "$BUILD_ONLY" = true ]; then
        build_docs
    else
        serve_docs
    fi
}

main "$@"

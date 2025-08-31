#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

ARTIFIACTS=(
    "examples/graphproppred/tokens_data"
)

print_usage() {
    echo "Usage: $0"
    echo
    echo "Clean up artifacts used for integration tests"
    echo
    echo "Arguments:"
    echo
}


remove_artifacts() {
    local artifacts_path="$1"
    echo "Remove artifacts at: $artifacts_path"
    rm -rf $artifacts_path
}


main() {
    print_usage

    for artifact_path in "${ARTIFIACTS[@]}"; do
        if [ -d "$artifact_path" ]; then
            remove_artifacts $artifact_path
        else
            echo "Artifacts $artifact_path no longer exists. Skip clean up"
        fi
    done

    echo "All sample datasets are available in example/graphproppred"
}

main "$@"

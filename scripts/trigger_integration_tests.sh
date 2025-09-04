#!/usr/bin/env bash
set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW="integration.yml"

print_usage() {
    echo "Usage: $0 [branch]"
    echo
    echo "Trigger the integration test workflow."
    echo
    echo "Arguments:"
    echo "  branch   Optional branch name (default: current Git branch, or main)"
}

# show usage if -h/--help
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

"$SCRIPT_DIR/trigger_workflow.sh" "$WORKFLOW" "$@"

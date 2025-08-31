#!/usr/bin/env bash
set -eou pipefail

OWNER="tgm-team"
REPO="tgm"
DEFAULT_WORKFLOW="integration.yml"
PERF_WORKFLOW="performance.yml"
ENV_FILE=".env"

print_usage() {
    echo "Usage: $0 [branch]"
    echo
    echo "Trigger GitHub Actions workflow in repo '$OWNER/$REPO'."
    echo
    echo "Arguments:"
    echo "  branch    Optional branch name (default: current working branch)"
    echo "  --perf    Optional flag to trigger performance workflow instead of integration tests"
    echo
    echo "Environment:"
    echo "  Requires TGM_CI_TOKEN in $ENV_FILE"
}

parse_args() {
    BRANCH="main"
    WORKFLOW="$DEFAULT_WORKFLOW"

    # Default branch: current Git branch
    if git rev-parse --git-dir > /dev/null 2>&1; then
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
    else
        BRANCH="main"  # fallback if not in a git repo
    fi

    for arg in "$@"; do
        case "$arg" in
            -h|--help)
                print_usage
                exit 0
                ;;
            --perf)
                WORKFLOW="$PERF_WORKFLOW"
                ;;
            *)
                BRANCH="$arg"
                ;;
        esac
    done

    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Error: $ENV_FILE file not found" >&2
        exit 1
    fi

    source "$ENV_FILE"

    if [[ -z "${TGM_CI_TOKEN:-}" ]]; then
        echo "Error: TGM_CI_TOKEN is not set in $ENV_FILE" >&2
        exit 1
    fi
}

trigger_workflow() {
    echo "Triggering workflow '$WORKFLOW' on branch '$BRANCH'..."
    local response http_code

    response=$(curl -sS -w "%{http_code}" -o /tmp/gh_response.log -X POST \
            -H "Accept: application/vnd.github+json" \
            -H "Authorization: Bearer $TGM_CI_TOKEN" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "https://api.github.com/repos/$OWNER/$REPO/actions/workflows/$WORKFLOW/dispatches" \
            -d "{\"ref\":\"$BRANCH\"}") || {
        echo "Error: Failed to contact GitHub API" >&2
        exit 1
    }

    http_code="${response:(-3)}"

    if [[ "$http_code" == "204" ]]; then
        echo "Workflow triggered successfully on branch '$BRANCH'."
        echo "View runs: https://github.com/$OWNER/$REPO/actions/workflows/$WORKFLOW"
    else
        echo "Failed to trigger workflow (HTTP $http_code)" >&2
        echo "Response body:" >&2
        exit 1
    fi
}

main() {
    parse_args "$@"
    trigger_workflow
}

main "$@"

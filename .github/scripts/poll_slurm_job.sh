#!/usr/bin/env bash
set -eou pipefail

print_usage() {
    echo "Usage: $0 JOB_ID [--timeout SECONDS] [--interval SECONDS]"
    echo
    echo "Poll a SLURM job until it completes."
    echo
    echo "Arguments:"
    echo "  JOB_ID               SLURM job ID to monitor"
    echo
    echo "Options:"
    echo "  --timeout SECONDS    Maximum wait time (default: 3600)"
    echo "  --interval SECONDS   Polling interval (default: 10)"
}

parse_args() {
    JOB_ID=""
    TIMEOUT=3600
    INTERVAL=10

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --timeout) TIMEOUT="$2"; shift 2 ;;
            --interval) INTERVAL="$2"; shift 2 ;;
            -h|--help) print_usage; exit 0 ;;
            *)
                if [[ -z "$JOB_ID" ]]; then
                    JOB_ID="$1"
                else
                    echo "Unknown argument: $1"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$JOB_ID" ]]; then
        echo "Error: JOB_ID is required"
        print_usage
        exit 1
    fi
}

poll_job() {
    local elapsed=0
    echo "Polling SLURM job $JOB_ID every $INTERVAL seconds (timeout: $TIMEOUT s)..."

    while true; do
        local state
        state=$(sacct -j "$JOB_ID" --format=State --noheader | awk '{print $1}' | tail -n1)

        case "$state" in
            COMPLETED)
                echo "Job $JOB_ID completed successfully."
                exit 0
                ;;
            FAILED|CANCELLED|TIMEOUT)
                echo "Job $JOB_ID ended with state: $state"
                exit 2
                ;;
            PENDING|CONFIGURING|RUNNING|COMPLETING)
                # still running
                ;;
            *)
                echo "Job $JOB_ID unknown state: '$state', continuing..."
                ;;
        esac

        sleep "$INTERVAL"
        elapsed=$((elapsed + INTERVAL))
        if (( elapsed >= TIMEOUT )); then
            echo "Timeout reached ($TIMEOUT s) waiting for job $JOB_ID"
            exit 3
        fi
    done
}

main() {
    parse_args "$@"
    poll_job
}

main "$@"

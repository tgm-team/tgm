#!/bin/bash
set -euo pipefail

# The following assumes we are one directlory deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST='0.0.0.0'
PORT=8000

echo "Starting CI listener FastAPI server..."

mkdir -p "$ROOT_DIR/ci/logs"

echo "Syncing CI dependancies"
uv sync --group ci
source "$ROOT_DIR/.env"

echo "Changing directory to project root: $ROOT_DIR"
cd "$ROOT_DIR"

echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Starting FastAPI listener with uvicorn on ${HOST}:${PORT}"
uvicorn scripts.run_ci_listener:app --host ${HOST} --port ${PORT}

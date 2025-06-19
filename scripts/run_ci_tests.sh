#!/bin/bash
set -euo pipefail

# The following assumes we are one directlory deep from the root
# directory, and the root directory contains the .env file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting CI tests..."
uv run pytest "$ROOT_DIR/test/integration"
echo "CI tests finished."

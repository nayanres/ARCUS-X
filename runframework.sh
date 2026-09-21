#!/usr/bin/env bash
# Backward-compatible spelling for the canonical run_framework.sh entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_framework.sh" "$@"

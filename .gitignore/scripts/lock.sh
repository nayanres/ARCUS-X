#!/usr/bin/env bash
# Regenerate both pinned lockfiles from their flexible specs.
#
#   requirements.txt        -> requirements-lock.txt        (runtime / API-only)
#   requirements-local.txt  -> requirements-lock-local.txt  (optional local inference)
#
# Requires pip-tools:  pip install -r requirements-dev.txt
# On Windows use scripts/lock.bat instead (handles the long-path workaround).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m piptools compile requirements.txt -o requirements-lock.txt --strip-extras
python -m piptools compile requirements-local.txt -o requirements-lock-local.txt --strip-extras

echo "Locks regenerated: requirements-lock.txt, requirements-lock-local.txt"

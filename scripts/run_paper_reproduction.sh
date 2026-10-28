#!/usr/bin/env bash
# Compatibility entry point; implementation lives under scripts/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/experiments/run_paper_reproduction.sh" "$@"

#!/usr/bin/env bash
# run_server.sh — Dust Stage 23 production server launcher.
#
# Starts the Python server (server.py) and blocks until interrupted.
#
# Usage:
#   bash deploy/run_server.sh [OPTIONS]
#
# Options (passed through to server.py):
#   --port PORT       Override WebSocket/HTTP listen port (default: 8765)
#   --seed SEED       World seed (only applied on first run / --reset)
#   --reset           Wipe world_state/ and start fresh
#   --state-dir DIR   Path to world_state directory
#   --headless        Run without display (default for server)
#
# Environment:
#   DUST_PORT         Overrides --port when set
#   DUST_SEED         Overrides --seed when set
#
# Examples:
#   # Start with defaults
#   bash deploy/run_server.sh
#
#   # Custom port + fresh world
#   bash deploy/run_server.sh --port 9000 --reset

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# Apply environment overrides
EXTRA_ARGS=()
if [[ -n "${DUST_PORT:-}" ]]; then
  EXTRA_ARGS+=(--port "${DUST_PORT}")
fi
if [[ -n "${DUST_SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "${DUST_SEED}")
fi

echo "Starting Dust server…"
echo "  Repo : ${REPO_ROOT}"
echo "  Args : ${EXTRA_ARGS[*]:-<defaults>} $*"
echo ""

cd "${REPO_ROOT}"
exec "${PYTHON}" server.py --headless "${EXTRA_ARGS[@]}" "$@"

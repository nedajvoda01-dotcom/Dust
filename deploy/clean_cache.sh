#!/usr/bin/env bash
# clean_cache.sh — Dust Stage 23 build & cache cleaner.
#
# Usage:
#   bash deploy/clean_cache.sh [--reset-world]
#
# What it does:
#   1. Removes all hashed asset files from web_client/dist/
#   2. Removes asset-manifest.json
#   3. Removes Python bytecode caches (__pycache__, *.pyc)
#   4. [Optional] Removes world_state/ directory when --reset-world is passed.
#
# After cleaning, re-run build_web.sh and then run_server.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${REPO_ROOT}/web_client/dist"

RESET_WORLD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset-world) RESET_WORLD=1; shift ;;
    *) shift ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Wipe dist/ (keep directory itself)
# ---------------------------------------------------------------------------
if [[ -d "${DIST_DIR}" ]]; then
  echo "Cleaning ${DIST_DIR}…"
  find "${DIST_DIR}" -type f -delete
  echo "  Done."
else
  echo "  dist/ does not exist — nothing to clean."
fi

# ---------------------------------------------------------------------------
# 2. Python bytecode caches
# ---------------------------------------------------------------------------
echo "Removing __pycache__ and *.pyc…"
find "${REPO_ROOT}" \
  -not -path "${REPO_ROOT}/.git/*" \
  -type d -name "__pycache__" \
  -exec rm -rf {} + 2>/dev/null || true
find "${REPO_ROOT}" \
  -not -path "${REPO_ROOT}/.git/*" \
  -type f -name "*.pyc" \
  -delete 2>/dev/null || true
echo "  Done."

# ---------------------------------------------------------------------------
# 3. Optional world state reset
# ---------------------------------------------------------------------------
if [[ "${RESET_WORLD}" -eq 1 ]]; then
  WORLD_DIR="${REPO_ROOT}/world_state"
  if [[ -d "${WORLD_DIR}" ]]; then
    echo "Removing world_state/…"
    rm -rf "${WORLD_DIR}"
    echo "  Done."
  else
    echo "  world_state/ does not exist — skipping."
  fi
fi

echo ""
echo "Clean complete."
echo "Run: bash deploy/build_web.sh && bash deploy/run_server.sh"

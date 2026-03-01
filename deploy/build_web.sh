#!/usr/bin/env bash
# build_web.sh — Dust Stage 23 web build pipeline.
#
# Copies web_client source files into web_client/dist/ with content
# hashes embedded in filenames.  Rewrites references in index.html
# and the service worker so clients use the correct hashed URLs.
#
# Process:
#   1. Compute a stable BUILD_ID from the source files.
#   2. Substitute __BUILD_ID__ in each source file into a temporary copy.
#   3. Hash the substituted copy, rename it, and place it in dist/.
#
# Output layout (web_client/dist/):
#   index.html           — no-cache entry point
#   client.<hash>.js     — main client bundle (hash of post-substitution content)
#   worker.<hash>.js     — network worker
#   sw.js                — service worker (never hashed per SW spec)
#   asset-manifest.json  — maps logical names to hashed filenames
#
# After running this script, start the server with:
#   bash deploy/run_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/web_client"
OUT_DIR="${REPO_ROOT}/web_client/dist"

# Parse optional --out argument
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    *)     shift ;;
  esac
done

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Derive a stable BUILD_ID from source files (before substitution).
# This is the logical build version used in cache-busting and WORLD_SYNC.
# ---------------------------------------------------------------------------
BUILD_ID="$(cat "${SRC_DIR}/client.js" "${SRC_DIR}/worker.js" | sha256sum | cut -c1-8)"

# ---------------------------------------------------------------------------
# Helper: SHA-256 hash (first 8 hex chars) of a file's content
# ---------------------------------------------------------------------------
file_hash() {
  sha256sum "$1" | cut -c1-8
}

# ---------------------------------------------------------------------------
# Helper: substitute BUILD_ID placeholder, hash the result, copy to dist/.
# Echoes the final basename (e.g. "client.a1b2c3d4.js").
# ---------------------------------------------------------------------------
hash_copy() {
  local src="$1"
  local stem="$2"
  local ext="$3"
  local tmp
  tmp="$(mktemp)"
  # Apply placeholder substitution to get the actual deployed content.
  sed "s/__BUILD_ID__/${BUILD_ID}/g" "${src}" > "${tmp}"
  local h
  h="$(file_hash "${tmp}")"
  local dest="${OUT_DIR}/${stem}.${h}.${ext}"
  cp "${tmp}" "${dest}"
  rm -f "${tmp}"
  echo "${stem}.${h}.${ext}"
}

# ---------------------------------------------------------------------------
# Step 2: Bundle JS assets (with BUILD_ID already substituted)
# ---------------------------------------------------------------------------
CLIENT_HASHED="$(hash_copy "${SRC_DIR}/client.js" "client" "js")"
WORKER_HASHED="$(hash_copy "${SRC_DIR}/worker.js" "worker" "js")"

# ---------------------------------------------------------------------------
# Step 3: Service worker — stable path (not hashed per SW spec), with BUILD_ID
# ---------------------------------------------------------------------------
sed "s/__BUILD_ID__/${BUILD_ID}/g" "${SRC_DIR}/sw.js" > "${OUT_DIR}/sw.js"

# ---------------------------------------------------------------------------
# Step 4: index.html — rewrite script src to hashed bundle names
# ---------------------------------------------------------------------------
sed \
  -e "s|client\.js|${CLIENT_HASHED}|g" \
  -e "s|worker\.js|${WORKER_HASHED}|g" \
  -e "s|__BUILD_ID__|${BUILD_ID}|g" \
  "${SRC_DIR}/index.html" > "${OUT_DIR}/index.html"

# ---------------------------------------------------------------------------
# Step 5: asset-manifest.json — consumed by SW and build tests
# ---------------------------------------------------------------------------
cat > "${OUT_DIR}/asset-manifest.json" <<JSON
{
  "buildId": "${BUILD_ID}",
  "files": {
    "client.js": "${CLIENT_HASHED}",
    "worker.js": "${WORKER_HASHED}",
    "sw.js":     "sw.js",
    "index.html":"index.html"
  }
}
JSON

echo ""
echo "Build complete."
echo "  Build ID : ${BUILD_ID}"
echo "  Output   : ${OUT_DIR}"
echo ""
echo "Run: bash deploy/run_server.sh"

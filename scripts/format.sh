#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0

Run Python formatters (black/ruff) if available.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if command -v black >/dev/null 2>&1; then
  echo "Running black..."
  black .
else
  echo "black not found; skipping"
fi

if command -v ruff >/dev/null 2>&1; then
  echo "Running ruff format..."
  ruff format .
else
  echo "ruff not found; skipping"
fi

echo "Formatting complete."
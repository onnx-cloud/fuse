#!/usr/bin/env bash
set -euo pipefail

# Installs a local `fuse` command.
#
# Default: installs into the project venv if present, otherwise into user site.

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=${PYTHON:-.venv/bin/python}
  MODE=${MODE:-venv}
else
  PYTHON=${PYTHON:-python3}
  MODE=${MODE:-user}
fi

usage() {
  cat <<EOF
Usage: $0

Installs the `fuse` CLI locally.

Env vars:
- PYTHON: python executable to use
- MODE: venv | user
  - venv: installs into ./.venv (recommended)
  - user: installs into user site-packages (~/.local) via --user

Examples:
- MODE=venv ./scripts/install_local.sh
- MODE=user ./scripts/install_local.sh
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" == "venv" ]]; then
  echo "Installing editable package into .venv"
  "$PYTHON" -m uv pip install --upgrade pip
  "$PYTHON" -m uv pip install -e .
  echo "Installed. Run: .venv/bin/fuse --help"
  exit 0
fi

if [[ "$MODE" == "user" ]]; then
  echo "Installing editable package into user site (~/.local)"
  "$PYTHON" -m uv pip install --upgrade --user pip
  "$PYTHON" -m uv pip install --user -e .
  echo "Installed. Ensure ~/.local/bin is on PATH, then run: fuse --help"
  exit 0
fi

echo "Unknown MODE: $MODE" >&2
usage
exit 2

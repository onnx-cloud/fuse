#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=${PYTHON:-.venv/bin/python}
else
  PYTHON=${PYTHON:-python3}
fi
NAME=${NAME:-fuse}
DIST_DIR=${DIST_DIR:-dist-exe}

usage() {
  cat <<EOF
Usage: $0

Build a standalone executable using PyInstaller.

Notes:
- Cross-platform: build on the target OS (Windows exe on Windows, macOS app on macOS).
- Uses entrypoint src/__main__.py.

Env vars:
- PYTHON: Python executable to use (default: python3)
- NAME: output binary name (default: fuse)
- DIST_DIR: output directory (default: dist-exe)
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if ! ${PYTHON} -c "import PyInstaller" >/dev/null 2>&1; then
  echo "Missing PyInstaller. Installing into current environment..."
  ${PYTHON} -m uv pip install --upgrade pyinstaller
fi

rm -rf build/ "$DIST_DIR"/ pyinstaller-build/
mkdir -p "$DIST_DIR"

echo "Building one-file executable '$NAME'"
${PYTHON} -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name "$NAME" \
  --distpath "$DIST_DIR" \
  src/__main__.py

echo "Done: $DIST_DIR/$NAME"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-}
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

if ! "$PYTHON_BIN" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "PyInstaller not available in $PYTHON_BIN. Install with: $PYTHON_BIN -m uv pip install pyinstaller"
  exit 1
fi

if ! command -v vsce >/dev/null 2>&1; then
  echo "vsce not found. Install with: npm i -g vsce"
  exit 1
fi

SERVER_DIR="$ROOT_DIR/vscode-extension/server"
mkdir -p "$SERVER_DIR"
rm -f "$SERVER_DIR/fuse-lsp" "$SERVER_DIR/fuse-lsp.exe"

PYINSTALLER_TEMP="$ROOT_DIR/.tmp_pyinstaller"
rm -rf "$PYINSTALLER_TEMP"
"$PYTHON_BIN" -m PyInstaller --noconfirm --onefile --name fuse-lsp --distpath "$SERVER_DIR" --workpath "$PYINSTALLER_TEMP" --specpath "$PYINSTALLER_TEMP" "$ROOT_DIR/src/lsp_server.py"
rm -rf "$PYINSTALLER_TEMP"

cd vscode-extension
npm install
vsce package

echo "Created VSIX package in vscode-extension/"

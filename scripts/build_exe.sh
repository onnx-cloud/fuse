#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=${PYTHON:-.venv/bin/python}
else
  PYTHON=${PYTHON:-python3}
fi
NAME=${NAME:-fuse}
DIST_DIR=${DIST_DIR:-dist}

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
- DIST_DIR: output directory (default: dist)
EOF
}

PATCH_ONLY=false
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

# internal mode useful for tests: apply version/build-time patch but skip
# the heavy PyInstaller invocation. leaves backup file behind so caller can
# restore manually.
for arg in "$@"; do
    if [[ "$arg" == "--patch-only" ]]; then
        PATCH_ONLY=true
    fi
done

if ! ${PYTHON} -c "import PyInstaller" >/dev/null 2>&1; then
  echo "Missing PyInstaller. Installing into current environment..."
  ${PYTHON} -m uv pip install --upgrade pyinstaller
fi

# Build a wheel and install it in this environment so that
# `importlib.metadata.version("fuse")` returns a real value and
# the runtime `__build_time__` is populated from the same timestamp.
# This mirrors the normal packaging workflow used by `make build`.
#
# We intentionally build the wheel rather than using `pip install -e .`
# because _build_time.txt is generated during wheel creation.
./scripts/build_wheel.sh
${PYTHON} -m pip install --upgrade dist/*.whl

rm -rf build/ "$DIST_DIR"/ pyinstaller-build/
mkdir -p "$DIST_DIR"

echo "Building one-file executable '$NAME'"

# PyInstaller imports src/__init__.py directly from the source tree, which
# relies on importlib.metadata to compute the version and build_time. When
# packaging from a checkout the metadata lookup returns nothing, so the
# fallback values (0.0.0 / "unknown") are shipped.  We temporarily rewrite
# the two constant assignments and then restore the original file.
PATCH_FILE=src/__init__.py
BACKUP_FILE=${PATCH_FILE}.bak

if [[ -f "$PATCH_FILE" ]]; then
    echo "Injecting version/build-time into $PATCH_FILE"
    ver=$(${PYTHON} - <<'PY'
from src.util.project_version import get_project_version
print(get_project_version() or "")
PY
)
    bt=$(cat src/_build_time.txt 2>/dev/null || echo unknown)
    cp "$PATCH_FILE" "$BACKUP_FILE"
    ${PYTHON} - <<PY
import pathlib, re
# shell variables injected here
ver = "$ver"
bt = "$bt"
path = pathlib.Path("$PATCH_FILE")
text = path.read_text().splitlines()
out = []
for line in text:
    if line.startswith('__version__ ='):
        out.append(f'__version__ = "{ver}"')
    elif line.startswith('__build_time__ ='):
        out.append(f'__build_time__ = "{bt}"')
    else:
        out.append(line)
path.write_text("\n".join(out) + "\n")
PY
fi

if [[ "$PATCH_ONLY" == "true" ]]; then
    echo "--patch-only requested; $PATCH_FILE has been modified and backup is at $BACKUP_FILE"
    # leave patched file in place; caller can restore manually with the backup
    exit 0
fi

${PYTHON} -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --name "$NAME" \
  --distpath "$DIST_DIR" \
  src/__main__.py

# restore original file if we patched it (skip when patch-only)
if [[ "$PATCH_ONLY" != "true" && -f "$BACKUP_FILE" ]]; then
    mv "$BACKUP_FILE" "$PATCH_FILE"
    echo "Restored original $PATCH_FILE"
fi

echo "Done: $DIST_DIR/$NAME"

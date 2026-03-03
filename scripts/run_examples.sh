#!/usr/bin/env bash
set -euo pipefail

# Prefer project virtualenv if present, otherwise fall back to system python
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=${PYTHON:-.venv/bin/python}
else
  PYTHON=${PYTHON:-python3}
fi
OUT_DIR=${OUT_DIR:-onnx}
VALIDATE=false

usage() {
  cat <<EOF
Usage: $0 [--validate] [-h]

Run the onnx lowering for examples in ./examples/golden and place outputs in ${OUT_DIR}/
--validate: run onnx.checker on generated models
EOF
}

while [[ ${1:-} != "" ]]; do
  case $1 in
    --validate) VALIDATE=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

mkdir -p "$OUT_DIR"

# Pre-flight check: ensure required Python packages are available to avoid
# obscure ModuleNotFoundError during conversion.
$PYTHON - <<PY || exit 1
import sys
missing = []
for pkg, mod in [('lark-parser','lark'), ('onnx','onnx')]:
    try:
        __import__(mod)
    except Exception:
        missing.append(pkg)
if missing:
    print('Missing packages: ' + ', '.join(missing), file=sys.stderr)
    print('Install with: uv pip install ' + ' '.join(missing), file=sys.stderr)
    sys.exit(1)
print('Required Python packages are present.')
PY

echo "Converting examples -> ${OUT_DIR}/"
# Run export via the testable command API to avoid relying on package executable
$PYTHON - <<PY
from src.cli import commands
from src.cli import cli_helpers
files = cli_helpers.find_fuse_files('examples/golden')
res = commands.cmd_compile(files, out_dir='$OUT_DIR')
for p in res:
    print(p)
PY

if [ "$VALIDATE" = true ]; then
  echo "Validating generated ONNX models..."
  for f in "$OUT_DIR"/*.onnx; do
    [ -e "$f" ] || continue
    PYTHON="$PYTHON" scripts/validate_onnx.sh "$f"
  done
fi

echo "Done. Models are in ${OUT_DIR}/"
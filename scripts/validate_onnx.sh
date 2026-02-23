#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

usage() {
  cat <<EOF
Usage: $0 <model.onnx>

Validate an ONNX model using onnx.checker.check_model
EOF
}

if [[ ${1:-} == "" || ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 1
fi

MODEL=$1

$PYTHON - "$MODEL" <<PY
import sys
# Avoid importing the local 'onnx' package from the repo root by removing
# the empty-string cwd entry from sys.path (the installed package will be used)
if sys.path and sys.path[0] == '':
    sys.path.pop(0)
import onnx
try:
    model = onnx.load(sys.argv[1])
    onnx.checker.check_model(model)
    print(f"[OK] {sys.argv[1]}")
except Exception as e:
    print(f"[FAIL] {sys.argv[1]} - {e}", file=sys.stderr)
    raise SystemExit(1)
PY

#!/usr/bin/env bash
set -euo pipefail

# Validate ONNX models in onnx/cookbook/ and optionally run a quick reference
# evaluation with ReferenceEvaluator (requires onnx). Usage: --run to execute

PYTHON=${PYTHON:-.venv/bin/python}
export PYTHON
OUT_DIR=${OUT_DIR:-onnx/cookbook}
DO_RUN=false

usage() {
  cat <<EOF
Usage: $0 [--run] [-h]

--run: attempt to execute each model via onnxruntime (fallback to ReferenceEvaluator)
-h: show this help
EOF
}

while [[ ${1:-} != "" ]]; do
  case $1 in
    --run) DO_RUN=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [ ! -d "$OUT_DIR" ]; then
  echo "No models found at $OUT_DIR" >&2
  exit 1
fi

for f in "$OUT_DIR"/*.onnx; do
  [ -e "$f" ] || continue
  echo "Validating $f..."
  # Use bash to run the validator to avoid invoking the shell script as Python
  bash scripts/validate_onnx.sh "$f"
  if [ "$DO_RUN" = true ]; then
    echo "Running $f (quick runtime check)..."
    $PYTHON - "$f" <<PY || true
import sys
# Avoid importing local 'onnx' package by removing cwd from sys.path
if sys.path and sys.path[0] == '':
    sys.path.pop(0)
import numpy as np
import onnx
m = onnx.load(sys.argv[1])
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    shp = [d if d > 0 else 1 for d in (inp.shape or [])]
    sample = np.ones(shp, dtype=np.float32)
    outs = sess.run(None, {inp.name: sample})
    print('onnxruntime OK - outputs:', [o.shape for o in outs])
except Exception as e:
    try:
        from onnx.reference import ReferenceEvaluator
        sess = ReferenceEvaluator(m)
        inp = m.graph.input[0]
        shp = [d.dim_value if (hasattr(d, 'dim_value') and d.dim_value > 0) else 1 for d in inp.type.tensor_type.shape.dim]
        sample = np.ones(shp, dtype=np.float32)
        outs = sess.run(None, {inp.name: sample})
        print('ReferenceEvaluator OK - outputs:', [o.shape for o in outs])
    except Exception as e2:
        print('Runtime check failed:', e, e2)
PY
  fi
done

echo "Done."
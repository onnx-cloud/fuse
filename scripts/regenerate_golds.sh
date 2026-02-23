#!/usr/bin/env bash
set -euo pipefail

# Regenerate golden ONNX artifacts into the committed 'onnx/' directory.
# Intended for local development. After running, inspect changes and commit if intended.

# Prefer project venv python if present
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=${PYTHON:-.venv/bin/python}
else
  PYTHON=${PYTHON:-python3}
fi

OUT_DIR=${OUT_DIR:-onnx}

echo "Generating ONNX goldens into '${OUT_DIR}/' (validate enabled)"
OUT_DIR="$OUT_DIR" ./scripts/run_examples.sh --validate

echo "Done. Review changes with 'git status' and run tests: 'make test'"

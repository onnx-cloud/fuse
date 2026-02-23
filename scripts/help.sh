#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Available scripts (in scripts/):

  setup_dev.sh     Create .venv and install dev dependencies
  check_env.sh     Verify Python + required packages (lark-parser, onnx, pytest)
  run_examples.sh  Convert examples to ONNX (use --validate to run onnx.checker)
  validate_onnx.sh Validate a single ONNX model
  run_tests.sh     Run pytest
  build_wheel.sh   Build a wheel into ./dist/
  format.sh        Run formatters (black, ruff)
  lint.sh          Run linters (ruff)
  help.sh          Show this help

Example usage:
  ./scripts/check_env.sh
  ./scripts/setup_dev.sh
  ./scripts/run_examples.sh --validate
EOF
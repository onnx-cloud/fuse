#!/usr/bin/env bash
set -euo pipefail

python -m pytest tests/cli/test_cli_lint_json_schema.py -q

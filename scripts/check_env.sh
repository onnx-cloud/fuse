#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

usage() {
  cat <<EOF
Usage: $0 [--help]

Quick environment checker for development.
Checks Python version and presence of common packages (lark-parser, onnx, pytest).
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

echo "Using PYTHON=${PYTHON}"

$PYTHON - <<'PY'
import sys
reqs = {
    'lark-parser': 'lark',
    'onnx': 'onnx',
    'pytest': 'pytest',
    'pygls': 'pygls',
}
ver = sys.version_info
print(f"Python {ver[0]}.{ver[1]}.{ver[2]}")
missing = []
for pkg, mod in reqs.items():
    try:
        __import__(mod)
        print(f"OK: {pkg} (module {mod})")
    except Exception:
        missing.append(pkg)
if missing:
    print('\nMissing packages: ' + ', '.join(missing), file=sys.stderr)
    print('Install with: uv pip install ' + ' '.join(missing), file=sys.stderr)
    raise SystemExit(1)
print('\nAll required packages are present.')
PY
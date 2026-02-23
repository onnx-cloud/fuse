#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

usage() {
  cat <<EOF
Usage: $0

Build a wheel (requires 'build' package). Output is in ./dist/
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if ! $PYTHON -c "import build" >/dev/null 2>&1; then
  echo "Missing 'build' package. Installing..."
  $PYTHON -m uv pip install --upgrade build
fi

echo "Building wheel"
# Record a deterministic UTC build timestamp for the package so runtime
# `fuse --version` can report a reproducible build time that matches the
# produced wheel. This is written into `src/_build_time.txt` which is
# packaged along with the source files.
python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
Path('src/_build_time.txt').write_text(datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z'))
print('Wrote src/_build_time.txt')
PY

$PYTHON -m build --wheel

echo "Done. Files placed in ./dist/"                                                                                                                                                                                                                                                                                                                                        
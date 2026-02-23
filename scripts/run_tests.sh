#!/usr/bin/env bash
set -euo pipefail

# Prefer the repository virtualenv pytest runner when available so
# local/CI runs are deterministic and use the dev deps installed into
# .venv. Export PYTEST to override.
if [[ -x .venv/bin/python && -z "${PYTEST:-}" ]]; then
  PYTEST=".venv/bin/python -m pytest"
else
  PYTEST=${PYTEST:-pytest}
fi

# If PYTEST is a command name that doesn't exist, fall back to the venv
# python -m pytest invocation when possible (helps users who forgot to
# activate the venv but have it present).
if ! command -v ${PYTEST%% *} >/dev/null 2>&1 && [[ -x .venv/bin/python ]]; then
  PYTEST=".venv/bin/python -m pytest"
fi

usage() {
  cat <<EOF
Usage: $0 [pytest-args]

Run test suite using pytest. Pass extra args to pytest.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

echo "Running tests ($PYTEST)"
$PYTEST "$@"

echo "Tests finished."
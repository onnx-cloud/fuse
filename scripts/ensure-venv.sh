#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/ensure-venv.sh [--create]
# Checks that a virtualenv is active (VIRTUAL_ENV set) and refers to the
# project's .venv when applicable. Exits non-zero with a helpful message when
# no venv is active so users immediately notice when they forgot to `source`.

CREATE=0
if [ "${1:-}" = "--create" ] || [ "${1:-}" = "-c" ]; then
    CREATE=1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ "$CREATE" -eq 1 ]; then
        echo "No active virtualenv. Creating .venv..."
        python3 -m venv .venv
        echo "Created .venv. Activate it with: source .venv/bin/activate"
        exit 0
    fi
    echo "ERROR: Virtualenv not activated. Run 'source .venv/bin/activate' or use '.venv/bin/python -m <command>' to run commands inside the project venv." >&2
    exit 2
fi

# If the project venv exists, warn when a different venv is active
if [ -x ".venv/bin/python" ]; then
    # Compare real paths to avoid symlink differences
    VENV_REAL=$(realpath "$VIRTUAL_ENV") || VENV_REAL="$VIRTUAL_ENV"
    PROJECT_REAL=$(realpath ".venv") || PROJECT_REAL=".venv"
    if [ "$VENV_REAL" != "$PROJECT_REAL" ]; then
        echo "WARNING: Active virtualenv ($VIRTUAL_ENV) is different from project .venv (./.venv). If you want to use the project venv, run: source .venv/bin/activate" >&2
    fi
fi

# If we reached here the env appears to be active (even if it's not the project .venv)
echo "Virtualenv active: ${VIRTUAL_ENV}"
exit 0

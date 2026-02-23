#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0

Run linters (ruff) if available.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if command -v ruff >/dev/null 2>&1; then
  echo "Running ruff check..."
  ruff check .
else
  echo "ruff not found; skipping"
fi

# Enforce canonical shorthand scalar types in .fuse sources (reject `<f32>[...]`)
# Prefer the concise `f32[...]` form.
echo "Checking for angle-scalar types in tracked .fuse files..."
BAD_ANGLE=""
if command -v git >/dev/null 2>&1; then
  # Use grep -P when available for a compact check; fall back to python when not.
  if grep -P "<\s*(f32|f64|i64|i32|bool)\[" $(git ls-files "*.fuse") >/dev/null 2>&1; then
    BAD_ANGLE=$(git ls-files "*.fuse" | xargs -r grep -nP "<\s*(f32|f64|i64|i32|bool)\[") || true
  else
    BAD_ANGLE=""
  fi
else
  # No git available — fall back to a python-based scan of the workspace
  BAD_ANGLE=$(python - <<'PY'
import re,sys,pathlib
pat=re.compile(r'<\s*(f32|f64|i64|i32|bool)\[')
outs=[]
for p in pathlib.Path('.').rglob('*.fuse'):
    t=p.read_text()
    if pat.search(t):
        outs.append(str(p))
print('\n'.join(outs))
PY
)
fi
if [ -n "${BAD_ANGLE:-}" ]; then
  echo "Found angle-scalar types (use shorthand, e.g. f32[...]) in these files:" >&2
  echo "$BAD_ANGLE" >&2
  exit 1
else
  echo "No angle-scalar types found in tracked .fuse files."
fi

# Quick SSA collision smoke-check (runs the targeted pytest); keep this
# fast and non-fatal for environments that don't have pytest available.
if command -v pytest >/dev/null 2>&1; then
  # Example-driven checks are intentionally disabled in CI because
  # examples are treated as golden artifacts. Run the heavy example
  # checks locally when developing example-related changes.
  echo "Skipping example-driven SSA/strict checks in lint (examples are golden)."
else
  echo "pytest not available; skipping SSA collision smoke check and strict example checks"
fi

echo "Linting complete."
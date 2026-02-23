#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to download a pinned ORT Web runtime into third_party/ort_web
# Usage: ./scripts/install_ort_web.sh [--dest <dir>] [--js-url <url>] [--wasm-url <url>]

DEST=${DEST:-third_party/ort_web}
LATEST="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest)
      LATEST="true"
      shift
      ;;
    --tag)
      TAG="$2"; shift 2
      ;;
    --dest)
      DEST="$2"; shift 2
      ;;
    *)
      # Positional args: js-url wasm-url
      if [ -z "${JS_URL:-}" ]; then
        JS_URL="$1"; shift
      elif [ -z "${WASM_URL:-}" ]; then
        WASM_URL="$1"; shift
      else
        shift
      fi
      ;;
  esac
done

if [ "$LATEST" = "true" ]; then
  # Fetch latest release and install
  python - <<'PY'
from src.ort_web_install import install_latest_ort_web
import sys
try:
    res = install_latest_ort_web("$DEST")
    print('Installed:', res)
except Exception as e:
    print('Error:', e)
    sys.exit(2)
PY
  exit $?
fi

# If a tag is supplied, install and pin the release by tag
if [ ! -z "${TAG:-}" ]; then
  python - <<'PY'
from src.ort_web_install import install_release_by_tag
import sys
try:
    res = install_release_by_tag("$DEST", "$TAG")
    print('Installed and pinned:', res)
except Exception as e:
    print('Error:', e)
    sys.exit(2)
PY
  exit $?
fi

# fallback to direct URLs
JS_URL=${JS_URL:-"https://ONNX.cloud/ort-wasm.js"}
WASM_URL=${WASM_URL:-"https://ONNX.cloud/ort-wasm.wasm"}

python -m src.ort_web_install --dest "$DEST" --js-url "$JS_URL" --wasm-url "$WASM_URL"

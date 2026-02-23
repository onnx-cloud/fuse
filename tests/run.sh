#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <test-file>"
  exit 2
fi

TEST_FILE="$1"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$SCRIPT_DIR/run.py" "$TEST_FILE" 
echo "Test $TEST_FILE completed successfully."

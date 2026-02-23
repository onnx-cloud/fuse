#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <generated_dir> <expected_dir>" >&2
  exit 2
fi

GENERATED_DIR=$1
EXPECTED_DIR=$2

missing=0
mismatch=0

echo "Comparing generated ONNX files in '$GENERATED_DIR' with expected in '$EXPECTED_DIR'..."

for f in "$EXPECTED_DIR"/*.onnx; do
  [ -e "$f" ] || continue
  name=$(basename "$f")
  g="$GENERATED_DIR/$name"
  if [ ! -e "$g" ]; then
    echo "MISSING: $name not generated (expected at $g)"
    missing=$((missing+1))
    continue
  fi
  # Validate the generated model for expected files so CI detects invalid but
  # regeneratable goldens.
  if ! PYTHON=${PYTHON:-python3} ./scripts/validate_onnx.sh "$g" >/dev/null 2>&1; then
    echo "INVALID: generated model $name failed validation"
    mismatch=$((mismatch+1))
    echo "  generated: $g (sha256: $(sha256sum "$g" | awk '{print $1}'))"
    echo "  expected: $f (sha256: $(sha256sum "$f" | awk '{print $1}'))"
    continue
  fi
  if ! cmp -s "$f" "$g"; then
    echo "MISMATCH: $name differs"
    echo "  expected: $f (sha256: $(sha256sum "$f" | awk '{print $1}'))"
    echo "  generated: $g (sha256: $(sha256sum "$g" | awk '{print $1}'))"
    mismatch=$((mismatch+1))
  fi
done

# Check for extra generated files not present in expected dir
extra=0
for f in "$GENERATED_DIR"/*.onnx; do
  [ -e "$f" ] || continue
  name=$(basename "$f")
  if [ ! -e "$EXPECTED_DIR/$name" ]; then
    echo "EXTRA: generated contains unexpected file $name"
    extra=$((extra+1))
  fi
done

if [ $missing -ne 0 ] || [ $mismatch -ne 0 ] || [ $extra -ne 0 ]; then
  echo "\nGolden consistency check failed: missing=$missing mismatch=$mismatch extra=$extra" >&2
  echo "To update goldens locally: './scripts/regenerate_golds.sh' then review and 'git add' any changed .onnx files." >&2
  exit 1
fi

echo "All golden ONNX files match the committed artifacts. ✅"

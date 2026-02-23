#!/usr/bin/env bash
set -euo pipefail

# Exports each Fuse example in examples/cookbook/ to
# a flat ONNX file under onnx/<name>.onnx using the `fuse` CLI.
# The fuse onnx command may write into a directory; this script
# finds the produced .onnx file and moves it into the flat layout.

mkdir -p onnx
# Track exported files for downstream numeric checks
exports_file="onnx/exports.txt"
rm -f "$exports_file"

FUSE_CMD=${FUSE_CMD:-fuse}
if ! command -v "$FUSE_CMD" >/dev/null 2>&1; then
  echo "Warning: fuse CLI '$FUSE_CMD' not found in PATH. Set FUSE_CMD to override." >&2
fi

output_dir="onnx/cookbook"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for f in examples/cookbook/*.fuse; do
  base=$(basename "$f" .fuse)
  tmpdir="onnx_tmp/${base}.onnx"
  rm -rf "$tmpdir"
  mkdir -p "$tmpdir"

  echo "Exporting $f -> $tmpdir"
  "$FUSE_CMD" onnx -f "$f" -o "$tmpdir"

  found=$(find "$tmpdir" -maxdepth 2 -type f -name '*.onnx' -print -quit || true)
  if [ -z "$found" ]; then
    echo "No .onnx produced for $f" >&2
    exit 1
  fi

  mv -f "$found" "$output_dir/${base}.onnx"
  rm -rf "$tmpdir"
  echo "Wrote $output_dir/${base}.onnx"
  echo "$output_dir/${base}.onnx" >> "$exports_file"
done

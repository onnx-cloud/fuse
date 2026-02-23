#!/usr/bin/env bash
set -euo pipefail

# Execute a small set of example notebooks using papermill and write to artifacts/
# Intended to run inside the container (working dir = /workspace) or locally

ARTIFACT_DIR=${1:-artifacts}
mkdir -p "$ARTIFACT_DIR"

echo "Running Papermill examples (artifacts -> $ARTIFACT_DIR)"

# Run the welcome notebook (small checks)
papermill jupyter/notebooks/welcome.ipynb "$ARTIFACT_DIR/welcome.executed.ipynb" --cwd /workspace

echo "Running quick_fuse with small param"
papermill jupyter/notebooks/quick_fuse.ipynb "$ARTIFACT_DIR/quick_fuse.executed.ipynb" -p input_value 2 --cwd /workspace

ls -l "$ARTIFACT_DIR"
echo "Papermill runs completed successfully."
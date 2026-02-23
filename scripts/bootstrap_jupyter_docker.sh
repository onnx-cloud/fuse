#!/usr/bin/env bash
set -euo pipefail

# Helper: bootstrap an interactive jupyter environment inside a container
# Installs kernel and optional extras for the 'fuse' user.

USER=${1:-fuse}
KERNEL_NAME=${2:-fuse}

python -m ipykernel install --name "${KERNEL_NAME}" --display-name "ONNX Fuse" --user
echo "Done. Kernel '${KERNEL_NAME}' available for user ${USER}."

#!/usr/bin/env bash
set -euo pipefail

# Starts Jupyter lab on host (connects to local workspace) using docker
IMAGE=${1:-fuse-jupyter:local}
PORT=${2:-8888}
WORKDIR=${PWD}

docker run --rm -it \
  -p ${PORT}:8888 \
  -v ${WORKDIR}:/workspace \
  -w /workspace \
  --name fuse-jupyter \
  ${IMAGE}

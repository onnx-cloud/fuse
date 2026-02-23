#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=fuse-jupyter-dev

DOCKER_BUILDKIT=1 docker build -t ${IMAGE_NAME} -f docker/jupyter/Dockerfile .

echo "Running container (port 8888 mapped to host 8888)"
docker run --rm -it -p 8888:8888 -v "$(pwd)":/workspace ${IMAGE_NAME}

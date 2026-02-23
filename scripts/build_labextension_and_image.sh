#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=fuse-jupyter-widget
DOCKERFILE=docker/jupyter/Dockerfile

echo "Building labextension (attempt local build if npm available)..."
if [ -d "jupyter/labextensions/fuse" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "npm found — running local build"
    (cd jupyter/labextensions/fuse && npm ci && npm run build)
  else
    echo "npm not found — skipping local build. Docker multi-stage build will build the extension inside node-builder"
  fi
fi

echo "Building Docker image ${IMAGE_NAME}..."
DOCKER_BUILDKIT=1 docker build -t ${IMAGE_NAME} -f ${DOCKERFILE} .

echo "Running container (background) and forwarding 8888..."
CONTAINER_ID=$(docker run -d -p 8888:8888 -v "$(pwd)":/workspace ${IMAGE_NAME})

echo "Waiting for server to be ready..."
until curl -sSf http://localhost:8888/fuse/api/health >/dev/null 2>&1; do
  sleep 1
done

echo "Server ready. Checking welcome page and health"
curl -s http://localhost:8888/fuse/welcome | head -n 5
curl -s http://localhost:8888/fuse/api/health | jq .

echo "Stopping container ${CONTAINER_ID}"
docker stop ${CONTAINER_ID}

echo "Done."
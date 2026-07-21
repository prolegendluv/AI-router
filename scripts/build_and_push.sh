#!/usr/bin/env bash
# Build a linux/amd64 image and push it to a public registry.
# Usage:  ./scripts/build_and_push.sh  ghcr.io/<user>/amd-router:latest
set -euo pipefail
cd "$(dirname "$0")/.."

if ! ls ./models/*.gguf >/dev/null 2>&1; then
  echo "No *.gguf in ./models — copy your Gemma E4B GGUF there first." >&2
  exit 1
fi

IMAGE_TAG="${1:?Usage: build_and_push.sh <registry/image:tag>}"
docker buildx build --platform linux/amd64 --tag "$IMAGE_TAG" --push .
echo "Pushed $IMAGE_TAG. Confirm it is PUBLIC in your registry settings."

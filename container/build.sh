#!/usr/bin/env bash
#
# Build the intlmatchpredictor image with Apple's `container` CLI.
#
# macOS / Apple Silicon only — see docs/Apple_Container_Setup.md.
# Functionally equivalent to `docker compose build` for the `api` service in
# docker-compose.yml: same Dockerfile, same build context, same tag.
#
# One-time prerequisite (not run automatically by this script, since it
# recreates the builder VM and only needs doing once per machine): the
# default builder VM is sized 2 CPU / 2 GiB, which is tight for this image.
# PyMC/PyTensor, XGBoost, and netCDF4 all pull in native-extension wheels
# during `pip install .` in the builder stage. Bump it before your first
# build:
#
#   container builder stop
#   container builder delete
#   container builder start --cpus 4 --memory 8g
#
# Usage:
#   ./container/build.sh [image_tag]
#
# Arguments:
#   image_tag: Tag to build the image as (default: intlmatchpredictor:latest).
#
set -euo pipefail

readonly IMAGE_TAG="${1:-intlmatchpredictor:latest}"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Ensuring the container runtime is running"
container system start >/dev/null 2>&1 || true

echo "==> Building ${IMAGE_TAG} from ${PROJECT_ROOT}/Dockerfile"
container build \
    --tag "${IMAGE_TAG}" \
    --file "${PROJECT_ROOT}/Dockerfile" \
    "${PROJECT_ROOT}"

echo "==> Built ${IMAGE_TAG}"
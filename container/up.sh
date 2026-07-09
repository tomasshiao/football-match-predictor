#!/usr/bin/env bash
#
# Start the wc26predictor container with Apple's `container` CLI.
#
# macOS / Apple Silicon only — see docs/Apple_Container_Setup.md.
#
# `container` (as of 1.0.0) has no native docker-compose equivalent, so this
# script is the direct, single-service translation of docker-compose.yml's
# `api` service: same image, same port mapping, same three volumes.
#
# Env overrides (mirrors the ${API_PORT:-8000} pattern already used in
# docker-compose.yml):
#   API_PORT        Host port to publish              (default: 8000)
#   CONTAINER_CPUS  vCPUs given to the container VM    (default: 4)
#   CONTAINER_MEM   Memory given to the container VM   (default: 4g)
#   RUN_AS_ROOT     Run as root instead of `predictor` (default: 1, see below)
#
# Deltas from docker-compose.yml (see docs/Apple_Container_Setup.md for
# details):
#   * `container run` has no --restart policy flag (no CLI equivalent exists
#     as of container 1.0). The container simply stays stopped if it exits;
#     `container start wc26predictor` brings it back.
#   * The Dockerfile's HEALTHCHECK instruction is Docker-specific image
#     metadata; `container` doesn't evaluate it. Use `make mac-status` (or
#     curl the /docs endpoint) to check liveness instead.
#   * Runs as root (RUN_AS_ROOT=1 below), unlike the Dockerfile's default
#     non-root `predictor` user that docker-compose.yml uses on Linux. As of
#     container 1.0, both bind mounts and named volumes are presented to the
#     guest VM owned by root regardless of host-side ownership — Apple's own
#     docs demonstrate this (a host file owned by a regular user shows up as
#     root:root once mounted: https://github.com/apple/container/blob/main/docs/how-to.md),
#     and it's tracked upstream at https://github.com/apple/container/issues/1830.
#     A non-root process therefore cannot write to /app/data, /app/models, or
#     the pytensor_cache volume — which is exactly the "Unable to create the
#     compiledir directory" error PyTensor raises. `chown`-ing the host-side
#     ./data or ./models folders from Terminal does not help: the ownership
#     you'd set there isn't what the guest VM sees. Set RUN_AS_ROOT=0 to try
#     non-root again once that upstream issue is resolved.
#
# Usage:
#   ./container/up.sh [image_tag]
#
# Arguments:
#   image_tag: Image to run (default: wc26predictor:latest).
#
set -euo pipefail

readonly IMAGE_TAG="${1:-wc26predictor:latest}"
readonly CONTAINER_NAME="wc26predictor"
readonly VOLUME_NAME="pytensor_cache"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

readonly PORT="${API_PORT:-8000}"
readonly CPUS="${CONTAINER_CPUS:-4}"
readonly MEMORY="${CONTAINER_MEM:-4g}"
readonly RUN_AS_ROOT="${RUN_AS_ROOT:-1}"

readonly DATA_DIR="${PROJECT_ROOT}/data"
readonly MODELS_DIR="${PROJECT_ROOT}/models"

# Bind-mount sources must exist and be absolute, mirroring what Docker does
# implicitly for ./data and ./models in docker-compose.yml.
mkdir -p "${DATA_DIR}" "${MODELS_DIR}"

echo "==> Ensuring the container runtime is running"
container system start >/dev/null 2>&1 || true

echo "==> Ensuring named volume '${VOLUME_NAME}' exists"
container volume create "${VOLUME_NAME}" >/dev/null 2>&1 || true

# Idempotent restart: drop any previous instance of this container so
# re-running this script behaves like `docker compose up -d` on an
# already-running stack instead of erroring on a name collision.
if container list --all --format json 2>/dev/null | grep -q "\"${CONTAINER_NAME}\""; then
    echo "==> Replacing existing '${CONTAINER_NAME}' container"
    container stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

USER_FLAGS=()
if [[ "${RUN_AS_ROOT}" == "1" ]]; then
    USER_FLAGS=(--user root)
fi

echo "==> Starting ${CONTAINER_NAME} on port ${PORT} (${CPUS} vCPU / ${MEMORY} RAM)"
container run \
    --detach \
    --name "${CONTAINER_NAME}" \
    --init \
    --cpus "${CPUS}" \
    --memory "${MEMORY}" \
    ${USER_FLAGS[@]+"${USER_FLAGS[@]}"} \
    --publish "${PORT}:8000" \
    --volume "${DATA_DIR}:/app/data" \
    --volume "${MODELS_DIR}:/app/models" \
    --volume "${VOLUME_NAME}:/app/.pytensor_cache" \
    "${IMAGE_TAG}"

echo "==> ${CONTAINER_NAME} is up: http://localhost:${PORT}/docs"
echo "    First request will be slow: PyTensor JIT-compiles on first NUTS sampling call."
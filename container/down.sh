#!/usr/bin/env bash
#
# Stop and remove the wc26predictor container started by container/up.sh.
#
# macOS / Apple Silicon only — see docs/Apple_Container_Setup.md.
#
# Equivalent to `docker compose down`: the named `pytensor_cache` volume and
# the ./data, ./models bind-mounted directories are left intact unless
# --volume is passed.
#
# Usage:
#   ./container/down.sh            # stop + remove the container only
#   ./container/down.sh --volume   # also delete the pytensor_cache volume
#
set -euo pipefail

readonly CONTAINER_NAME="wc26predictor"
readonly VOLUME_NAME="pytensor_cache"

echo "==> Stopping ${CONTAINER_NAME}"
container stop "${CONTAINER_NAME}" >/dev/null 2>&1 || echo "    (not running)"

echo "==> Removing ${CONTAINER_NAME}"
container rm "${CONTAINER_NAME}" >/dev/null 2>&1 || echo "    (already removed)"

if [[ "${1:-}" == "--volume" ]]; then
    echo "==> Deleting volume ${VOLUME_NAME}"
    container volume rm "${VOLUME_NAME}" >/dev/null 2>&1 || echo "    (already removed)"
fi
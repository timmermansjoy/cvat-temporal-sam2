#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR
readonly COMPOSE=(
    docker compose --project-name cvat
    -f "$ROOT_DIR/docker-compose.yml"
    -f "$ROOT_DIR/components/serverless/docker-compose.serverless.yml"
    -f "$ROOT_DIR/docker-compose.dev.yml"
)

if command -v nuctl >/dev/null; then
    nuctl delete function pth-facebookresearch-sam2 --platform local >/dev/null 2>&1 || true
fi

"${COMPOSE[@]}" down --remove-orphans

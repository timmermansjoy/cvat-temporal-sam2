#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR
readonly NUCLIO_VERSION='1.16.3'
readonly COMPOSE=(
    docker compose --project-name cvat
    -f "$ROOT_DIR/docker-compose.yml"
    -f "$ROOT_DIR/components/serverless/docker-compose.serverless.yml"
)

device='cpu'
if [[ "${1:-}" == '--gpu' ]]; then
    device='gpu'
    shift
elif [[ "${1:-}" == '--cpu' ]]; then
    shift
fi

if (( $# != 1 )); then
    printf 'Usage: %s [--cpu|--gpu] <public-hostname-or-ip>\n' "$0" >&2
    exit 2
fi

export CVAT_HOST="$1"
if [[ -z "$CVAT_HOST" || "$CVAT_HOST" == *://* || "$CVAT_HOST" == */* ]]; then
    printf 'Set CVAT_HOST to the hostname or IP users open in their browser.\n' >&2
    printf 'Example: %s cvat.example.com\n' "$0" >&2
    exit 2
fi

if ! command -v nuctl >/dev/null; then
    printf 'nuctl %s is required to match the Nuclio dashboard.\n' "$NUCLIO_VERSION" >&2
    exit 1
fi

if [[ "$device" == 'gpu' ]] && ! nvidia-smi >/dev/null 2>&1; then
    printf 'An NVIDIA GPU with a working driver is required for the SAM2 GPU deployment.\n' >&2
    exit 1
fi

"${COMPOSE[@]}" up -d

for _ in {1..30}; do
    if nuctl get function --platform local >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if ! nuctl get function --platform local >/dev/null 2>&1; then
    printf 'Nuclio did not become ready. Check the nuclio container logs.\n' >&2
    exit 1
fi

"$ROOT_DIR/serverless/deploy_${device}.sh" "$ROOT_DIR/serverless/pytorch/facebookresearch/sam2/nuclio"

printf 'CVAT is available at http://%s:8080\n' "$CVAT_HOST"
if [[ "$device" == 'cpu' ]]; then
    printf 'SAM2 is running on CPU and is intended only for testing. Use --gpu for production tracking.\n'
fi

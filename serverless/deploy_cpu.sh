#!/bin/bash
# Sample commands to deploy nuclio functions on CPU

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

export DOCKER_BUILDKIT=1

if [[ "$FUNCTIONS_DIR" == "$SCRIPT_DIR" || "$FUNCTIONS_DIR" == "$SCRIPT_DIR/openvino"/* ]]; then
    docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"
fi

nuctl create project cvat --platform local

if [[ -f "$FUNCTIONS_DIR/function.yaml" ]]; then
    function_configs=("$FUNCTIONS_DIR/function.yaml")
else
    function_configs=()
    while IFS= read -r -d '' func_config; do
        function_configs+=("$func_config")
    done < <(find "$FUNCTIONS_DIR" -type f -name function.yaml -print0)
fi

for func_config in "${function_configs[@]}"; do
    func_root="$(dirname "$func_config")"
    func_rel_path="${func_root#"$SCRIPT_DIR"/}"

    if [ -f "$func_root/Dockerfile" ]; then
        docker build -t "cvat.${func_rel_path//\//.}.base" "$func_root"
    fi

    echo "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

nuctl get function --platform local

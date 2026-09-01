#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-vllm_node}"
WAIT_SECONDS="${WARMUP_CONTAINER_WAIT_SECONDS:-1800}"

deadline=$((SECONDS + WAIT_SECONDS))
until docker inspect "$CONTAINER" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for warmup container: $CONTAINER" >&2
        exit 1
    fi
    sleep 2
done

docker exec -i "$CONTAINER" python3 - "$@" \
    < "$SCRIPT_DIR/warmup-deepseek-v4-prefill.py"
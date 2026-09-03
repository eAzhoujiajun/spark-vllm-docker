#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONTAINER="${CONTAINER:-${CONTAINER_NAME:-vllm_node}}"
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

if [[ "${DSPARK_SKIP_BOOT_SHAPE_WARMUP:-0}" != "1" ]]; then
    bash "$SCRIPT_DIR/boot-shape-warmup.sh" || {
        status=$?
        echo "WARN: boot-shape warmup exited with status $status; server remains usable but may JIT cold shapes later." >&2
    }
fi
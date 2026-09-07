#!/bin/bash
# Warm the DeepSeek V4 Flash (Vision-Exp) serving stack after launch.
#
# Chain (combined from both upstream DGX Spark implementations):
#   1. prefill write-side warmup   -> warmup-deepseek-v4-prefill.py:
#        sends live /v1/chat/completions requests whose prompt lengths sweep the
#        prefill kernel's working set (default 512..131072 tok).
#        (tonyd2wild: "the engine runs ~30% slow after boot or ~30 min idle",
#         so warm the Triton/DeepSeek kernels before real traffic.)
#   2. request-shape warmup        -> boot-shape-warmup.sh:
#        burns the spec-decode/prefill/sampler Triton shape buckets (BLOCK
#        ladder, chat arms C=1/2/3/4/6, sampling arms, think-off, and one
#        image_url arm) so no request has to JIT mid-serve
#        (MiaAI-Lab issue #117; 600 s NCCL watchdog hazard on TP=2).
#
# Non-fatal by design: warmup gaps only degrade to a mid-serve JIT (the very
# thing this runs to avoid), never to a failed boot — so a warmup failure is
# WARN, not an error. Set DSPARK_SKIP_BOOT_SHAPE_WARMUP=1 to skip stage 2.
#
# Env (all optional; defaults match the vision-exp native recipe):
#   CONTAINER_NAME    container running the server   (default vllm_node)
#   WARMUP_BASE_URL   server base URL for boot-shape (default http://127.0.0.1:8888)
#   SERVED_MODEL_NAME model name                      (default deepseek-v4-flash-vision-exp)
#   WARMUP_MAX_CONCURRENCY  concurrency for boot-shape (default MAX_NUM_SEQS or 6)
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
BASE_URL="${WARMUP_BASE_URL:-http://127.0.0.1:8888}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
MAX_CONCURRENCY="${WARMUP_MAX_CONCURRENCY:-${MAX_NUM_SEQS:-6}}"
WAIT_SECONDS="${WARMUP_CONTAINER_WAIT_SECONDS:-1800}"
SKIP_BOOT_SHAPE="${DSPARK_SKIP_BOOT_SHAPE_WARMUP:-0}"

echo "warmup: container=$CONTAINER base=$BASE_URL model=$MODEL concurrency=$MAX_CONCURRENCY"

# Stage 0: wait for the server container to exist.
deadline=$((SECONDS + WAIT_SECONDS))
until docker inspect "$CONTAINER" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for warmup container: $CONTAINER" >&2
        exit 1
    fi
    sleep 2
done

# Stage 1: prefill write-side warmup, inside the container (it owns the API's
# loopback + tokenizer/model paths).
docker exec -i "$CONTAINER" python3 - "$@" \
    < "$SCRIPT_DIR/warmup-deepseek-v4-prefill.py"

# Stage 2: request-shape warmup (spec-decode/prefill/sampler Triton buckets).
if [[ "$SKIP_BOOT_SHAPE" != "1" ]]; then
    DSPARK_WARMUP_MAX_CONCURRENCY="$MAX_CONCURRENCY" \
        bash "$SCRIPT_DIR/boot-shape-warmup.sh" "$BASE_URL" "$MODEL" || {
        status=$?
        echo "WARN: boot-shape warmup exited with status $status; server remains usable but may JIT cold shapes later." >&2
    }
else
    echo "warmup: boot-shape warmup skipped (DSPARK_SKIP_BOOT_SHAPE_WARMUP=1)"
fi

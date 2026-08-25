#!/bin/bash
set -euo pipefail

PREFIX="[fix-decode-topk-cache]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
TARGET_IDX="$VLLM_ROOT/v1/attention/backends/mla/indexer.py"
TARGET_SPARSE="$VLLM_ROOT/model_executor/layers/sparse_attn_indexer.py"
PATCHER_IDX="$MOD_DIR/patch_indexer_metadata.py"
PATCHER_SPARSE="$MOD_DIR/patch_sparse_indexer_decode.py"
# 0 = reuse disabled (always rescan, pre-fix behavior); >0 = replay decode
# top-k for this many steps between full rescans (default 32).
REFRESH_INTERVAL="${VLLM_SPARSE_DECODE_REFRESH_INTERVAL:-32}"

echo "=== fix-decode-topk-cache mod ==="

if [[ ! -d "$VLLM_ROOT" ]]; then
    echo "$PREFIX vLLM package not found at $VLLM_ROOT" >&2
    exit 1
fi
for f in "$TARGET_IDX" "$TARGET_SPARSE" "$PATCHER_IDX" "$PATCHER_SPARSE"; do
    if [[ ! -f "$f" ]]; then
        echo "$PREFIX missing file: $f" >&2
        exit 1
    fi
done

python3 "$PATCHER_IDX" --check "$TARGET_IDX"
python3 "$PATCHER_IDX" "$TARGET_IDX"
python3 "$PATCHER_IDX" --check "$TARGET_IDX"

python3 "$PATCHER_SPARSE" --check "$TARGET_SPARSE"
python3 "$PATCHER_SPARSE" "$TARGET_SPARSE"
python3 "$PATCHER_SPARSE" --check "$TARGET_SPARSE"

find "$(dirname "$TARGET_IDX")" "$(dirname "$TARGET_SPARSE")" \
    \( -name "__pycache__" -o -name "*.pyc" \) -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX Enabled with VLLM_SPARSE_DECODE_REFRESH_INTERVAL=$REFRESH_INTERVAL"
echo "=== OK: decode top-k replayed between rescans; quality gate = accept rate ==="

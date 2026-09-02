#!/bin/bash
set -euo pipefail

PREFIX="[fix-deepseek-v4-tool-args-wrapper]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
TARGET="$VLLM_ROOT/parser/deepseek_v4.py"
PATCHER="$MOD_DIR/patch_model.py"

echo "=== DeepSeek-V4 tool-args wrapper mod (unwrap commands) ==="

if [[ ! -d "$VLLM_ROOT" ]]; then
    echo "$PREFIX vLLM package not found at $VLLM_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PATCHER" ]]; then
    echo "$PREFIX patcher not found at $PATCHER" >&2
    exit 1
fi

if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX vLLM parser module not found at $TARGET" >&2
    exit 1
fi

python3 "$PATCHER" --check "$TARGET"
python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

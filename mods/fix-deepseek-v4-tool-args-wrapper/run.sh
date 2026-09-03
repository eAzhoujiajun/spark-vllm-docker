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

# Also patch deepseek_v32 if present
if [[ -f "$VLLM_ROOT/parser/deepseek_v32.py" ]]; then
    python3 "$PATCHER" "$VLLM_ROOT/parser/deepseek_v32.py" || true
fi

# Apply reasoning_effort mapping in deepseek_v4 tokenizer if present
if [[ -f "$VLLM_ROOT/tokenizers/deepseek_v4.py" ]]; then
    python3 -c 'from pathlib import Path; p=Path("'"$VLLM_ROOT"'/tokenizers/deepseek_v4.py"); s=p.read_text(); old="elif reasoning_effort in (\"max\", \"xhigh\"):\n                reasoning_effort = \"max\"\n            else:\n                reasoning_effort = \"high\""; new="elif reasoning_effort in (\"max\", \"xhigh\"):\n                reasoning_effort = \"max\"\n            elif reasoning_effort == \"high\":\n                reasoning_effort = \"high\"\n            else:\n                reasoning_effort = \"low\""; updated=s.replace(old,new); (new in updated or old not in updated) and p.write_text(updated)' || true
fi

# Apply issue 21 encoder dict-arguments hotfix if encoding file is present
if [[ -f "$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py" && -f "$MOD_DIR/hotfix-encoding-dsv4-issue21.py" ]]; then
    python3 "$MOD_DIR/hotfix-encoding-dsv4-issue21.py" "$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py" || true
fi

# Apply suppress stops in reasoning
if [[ -f "$MOD_DIR/hotfix-dsv4-suppress-stops-in-reasoning.py" ]]; then
    python3 "$MOD_DIR/hotfix-dsv4-suppress-stops-in-reasoning.py" || true
fi

# Apply grammar advance across reasoning boundary (#44993)
if [[ -f "$MOD_DIR/hotfix-dsv4-grammar-advance.sh" ]]; then
    VLLM_ROOT="$VLLM_ROOT" bash "$MOD_DIR/hotfix-dsv4-grammar-advance.sh" || true
fi

# Apply tool call truncation safety (issue #55)
if [[ -f "$MOD_DIR/hotfix-dsv4-issue55-tool-truncation.py" ]]; then
    python3 "$MOD_DIR/hotfix-dsv4-issue55-tool-truncation.py" "$VLLM_ROOT" || true
fi

# Apply assistant final continuation (issue #52) if encoding module is installed
if [[ -f "$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py" && -f "$MOD_DIR/hotfix-dsv4-assistant-final-continuation.py" ]]; then
    python3 "$MOD_DIR/hotfix-dsv4-assistant-final-continuation.py" "$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py" || true
fi

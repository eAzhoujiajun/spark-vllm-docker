#!/bin/bash
set -euo pipefail

PREFIX="[deepseek-v4-vision-exp]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
MODEL_PATH="${DEEPSEEK_VISION_MODEL_PATH:-/models/DeepSeek-V4-Flash-Vision-Exp}"
ENCODING_SOURCE="$MODEL_PATH/encoding/encoding_dsv4.py"
ENCODING_TARGET="$PYTHON_ROOT/vllm/tokenizers/deepseek_v4_encoding.py"
PATCH_ROOT="/opt/dspark-patches"

echo "=== DeepSeek V4 Flash Vision-Exp multimodal mod ==="

for file in \
    "$PYTHON_ROOT/vllm/models/deepseek_v4/nvidia/model.py" \
    "$PYTHON_ROOT/vllm/models/deepseek_v4/nvidia/dspark.py" \
    "$ENCODING_SOURCE" \
    "$MOD_DIR/hotfix-dsv4-vision-exp.py" \
    "$MOD_DIR/vision_exp/apply.py" \
    "$MOD_DIR/vision_exp/processor.py" \
    "$MOD_DIR/vision_exp/vision.py"; do
    if [[ ! -f "$file" ]]; then
        echo "$PREFIX missing required file: $file" >&2
        exit 1
    fi
done

mkdir -p "$PATCH_ROOT/vision_exp"
cp "$MOD_DIR/vision_exp/"*.py "$PATCH_ROOT/vision_exp/"
cp "$ENCODING_SOURCE" "$ENCODING_TARGET"

# Reasoning-effort mapping
python3 -c 'from pathlib import Path; p=Path("'"$PYTHON_ROOT"'/vllm/tokenizers/deepseek_v4.py"); s=p.read_text(); old="elif reasoning_effort in (\"max\", \"xhigh\"):\n                reasoning_effort = \"max\"\n            else:\n                reasoning_effort = \"high\""; new="elif reasoning_effort in (\"max\", \"xhigh\"):\n                reasoning_effort = \"max\"\n            elif reasoning_effort == \"high\":\n                reasoning_effort = \"high\"\n            else:\n                reasoning_effort = \"low\""; updated=s.replace(old,new); (new in updated or old not in updated) and p.write_text(updated)' || true

# Issue #21 encoder dict arguments fix
if [[ -f "$MOD_DIR/hotfix-encoding-dsv4-issue21.py" ]]; then
    python3 "$MOD_DIR/hotfix-encoding-dsv4-issue21.py" "$ENCODING_TARGET"
fi

python3 "$MOD_DIR/hotfix-dsv4-vision-exp.py" \
    "$PATCH_ROOT/vision_exp" \
    "$PYTHON_ROOT/vllm/models/deepseek_v4/nvidia/model.py" \
    "$ENCODING_TARGET" \
    "$PYTHON_ROOT/vllm/models/deepseek_v4/nvidia/dspark.py"
python3 "$MOD_DIR/hotfix-dsv4-vision-exp.py" --status

# Suppress stops in reasoning
if [[ -f "$MOD_DIR/hotfix-dsv4-suppress-stops-in-reasoning.py" ]]; then
    python3 "$MOD_DIR/hotfix-dsv4-suppress-stops-in-reasoning.py" || true
fi

# Grammar advance across reasoning boundary
if [[ -f "$MOD_DIR/hotfix-dsv4-grammar-advance.sh" ]]; then
    VLLM_ROOT="$PYTHON_ROOT/vllm" bash "$MOD_DIR/hotfix-dsv4-grammar-advance.sh" || true
fi

# Tool-call truncation safety
if [[ -f "$MOD_DIR/hotfix-dsv4-issue55-tool-truncation.py" ]]; then
    python3 "$MOD_DIR/hotfix-dsv4-issue55-tool-truncation.py" "$PYTHON_ROOT/vllm" || true
fi

# Assistant final continuation
if [[ -f "$MOD_DIR/hotfix-dsv4-assistant-final-continuation.py" ]]; then
    python3 "$MOD_DIR/hotfix-dsv4-assistant-final-continuation.py" "$ENCODING_TARGET" || true
fi

find "$PYTHON_ROOT/vllm" "$PATCH_ROOT/vision_exp" \
    \( -name "__pycache__" -o -name "*.pyc" \) -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: DeepSeek V4 image_url support installed ==="
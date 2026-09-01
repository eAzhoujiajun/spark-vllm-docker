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

python3 "$MOD_DIR/hotfix-dsv4-vision-exp.py" \
    "$PATCH_ROOT/vision_exp" \
    "$PYTHON_ROOT/vllm/models/deepseek_v4/nvidia/model.py" \
    "$ENCODING_TARGET" \
    "$PYTHON_ROOT/vllm/models/deepseek_v4/nvidia/dspark.py"
python3 "$MOD_DIR/hotfix-dsv4-vision-exp.py" --status

find "$PYTHON_ROOT/vllm" "$PATCH_ROOT/vision_exp" \
    \( -name "__pycache__" -o -name "*.pyc" \) -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: DeepSeek V4 image_url support installed ==="
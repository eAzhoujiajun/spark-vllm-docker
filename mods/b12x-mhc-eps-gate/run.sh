#!/bin/bash
# b12x-mhc-eps-gate: relax the fused Gram mHC kernel gate so it accepts
# rms_norm_eps <= 1e-6 (DeepSeek-V4-Flash-Vision-Exp ships rms_norm_eps=1e-20;
# the 0731 model ships 1e-6). The gate was hard == 1.0e-6, so vision-exp could
# never use the fused kernel and fell back to the per-layer tilelang mHC path
# (43 layers x 3 tilelang ops per decode step).
# Numerics verified offline: run_mhc_pre_partial + run_mhc_finalize_gram with
# rms_eps=1e-20 produce finite output identical in magnitude to 1e-6.
set -euo pipefail

PREFIX="[b12x-mhc-eps-gate]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_ROOT="/usr/local/lib/python3.12/dist-packages"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-$DEFAULT_PYTHON_ROOT}}"
TARGET="$PYTHON_ROOT/b12x/norm/mhc/_impl.py"
PATCHER="$MOD_DIR/patch_mhc_gate.py"

echo "=== b12x-mhc-eps-gate mod ==="

if [[ ! -d "$PYTHON_ROOT/b12x" ]]; then
    echo "$PREFIX b12x package not found under $PYTHON_ROOT" >&2
    exit 1
fi
if [[ ! -f "$TARGET" ]]; then
    echo "$PREFIX missing target: $TARGET" >&2
    exit 1
fi

python3 "$PATCHER" "$TARGET"
python3 "$PATCHER" --check "$TARGET"

# Drop bytecode caches so the patched module is recompiled on next import.
find "$PYTHON_ROOT/b12x/norm/mhc" \( -name "__pycache__" -o -name "*.pyc" \) -exec rm -rf {} + 2>/dev/null || true

echo "$PREFIX OK: fused mHC gate accepts rms_norm_eps <= 1e-6 (form A constant widened or form B sites patched)"
echo "=== OK ==="

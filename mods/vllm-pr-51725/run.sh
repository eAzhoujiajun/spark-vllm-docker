#!/bin/bash
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PR_NUMBER="$(tr -d '\r\n' < "$MOD_DIR/pr-number")"
PATCH_FILE="$MOD_DIR/pr.diff"
EXPECTED_SHA256="$(tr -d '\r\n' < "$MOD_DIR/pr.sha256")"
PREFIX="[vllm-pr #${PR_NUMBER}]"

if ! command -v git >/dev/null 2>&1; then
    echo "$PREFIX git is required to apply this runtime PR." >&2
    echo "$PREFIX Apply mods/use-official-vllm before --apply-vllm-pr when using an image without git." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to locate the installed vLLM package." >&2
    exit 1
fi

ACTUAL_SHA256="$(python3 - "$PATCH_FILE" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    echo "$PREFIX Patch checksum mismatch; refusing to apply it." >&2
    exit 1
fi

VLLM_PACKAGE_DIR="${VLLM_PACKAGE_DIR:-$(python3 - <<'PY'
from importlib.util import find_spec

spec = find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Could not locate the installed vLLM package")
print(next(iter(spec.submodule_search_locations)))
PY
)}"

if [[ ! -d "$VLLM_PACKAGE_DIR" || "$(basename "$VLLM_PACKAGE_DIR")" != "vllm" ]]; then
    echo "$PREFIX Invalid installed vLLM package directory: $VLLM_PACKAGE_DIR" >&2
    exit 1
fi

PYTHON_ROOT="$(dirname "$VLLM_PACKAGE_DIR")"
cd "$PYTHON_ROOT"
APPLY_ARGS=(--binary --include='vllm/**')

echo "$PREFIX Applying validated runtime patch $EXPECTED_SHA256 to $VLLM_PACKAGE_DIR"
if git apply --reverse --check "${APPLY_ARGS[@]}" "$PATCH_FILE" >/dev/null 2>&1; then
    echo "$PREFIX Patch is already applied; skipping."
elif git apply --check "${APPLY_ARGS[@]}" "$PATCH_FILE"; then
    git apply "${APPLY_ARGS[@]}" "$PATCH_FILE"
    echo "$PREFIX Applied successfully."
else
    echo "$PREFIX Patch does not apply cleanly to the installed vLLM package." >&2
    echo "$PREFIX Rebuild with build-and-copy.sh --apply-vllm-pr $PR_NUMBER if this PR is not runtime-compatible." >&2
    exit 1
fi

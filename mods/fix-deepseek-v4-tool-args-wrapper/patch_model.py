#!/usr/bin/env python3
"""Unwrap one more model-side tool-arg wrapper for the DeepSeek-V4 DSML parser.

The B12X fork's ``_unwrap_wrapper_args`` (vllm/parser/deepseek_v4.py) already
normalizes ``{"arguments": {...}}`` and ``{"input": {...}}`` wrappers that
DeepSeek-V4 models intermittently emit instead of plain DSML parameters.
The deepseek-v4-flash-vision-exp variant additionally emits
``{"commands": [{"command": ..., "description": ...}]}`` — a single-element
list (or dict) whose keys are a subset of the tool's schema properties — which
is NOT unwrapped today, so the wrapped arguments reach the client as-is and
fail JSON-Schema validation (DSH: "invalid arguments: missing required
property ..."). This patch adds ``commands`` to the wrapper candidates and
accepts a single-element list wrapper as well as a dict.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: fix-deepseek-v4-tool-args-wrapper v2"

# The wrapper-candidate loop in _unwrap_wrapper_args (deepseek_v4.py).
WRAPPER_LOOP_RE = re.compile(
    r'^( +)for wrapper in \("arguments", "input"(?:, "commands")?\):.*$',
    re.MULTILINE,
)
WRAPPER_LOOP_PATCH = (
    '\\1for wrapper in ("arguments", "input", "commands", "parameters", "params", "kwargs", "args", "payload", "body", "tool_input"):'
    '  ' + MARKER
)

# The check for wrapper key and inner extraction/dict check in the same function.
INNER_CHECK_RE = re.compile(
    r'^( +)if set\(args\.keys\(\)\) != \{wrapper\} or wrapper in allowed:\n'
    r'\1    continue\n'
    r'\1inner = args\[wrapper\]\n'
    r'(?:\1if isinstance\(inner, list\) and len\(inner\) == 1 and isinstance\(inner\[0\], dict\):\n'
    r'\1    inner = inner\[0\]\n)?'
    r'\1if isinstance\(inner, str\):\n'
    r'\1    try:\n'
    r'\1        inner = json\.loads\(inner\)\n'
    r'\1    except json\.JSONDecodeError:\n'
    r'\1        return args_json\n'
    r'(?:\1if isinstance\(inner, list\) and len\(inner\) == 1 and isinstance\(inner\[0\], dict\):\n'
    r'\1    inner = inner\[0\]\n)?'
    r'\1if isinstance\(inner, dict\) and set\(inner\.keys\(\)\)\.issubset\(allowed\):\n'
    r'\1    return json\.dumps\(inner, ensure_ascii=False\)',
    re.MULTILINE,
)
INNER_CHECK_PATCH = (
    '\\1if wrapper not in args or wrapper in allowed:\n'
    '\\1    continue\n'
    '\\1if set(args.keys()) != {wrapper} and any(k in allowed for k in args.keys()):\n'
    '\\1    continue\n'
    '\\1inner = args[wrapper]\n'
    '\\1if isinstance(inner, str):\n'
    '\\1    try:\n'
    '\\1        inner = json.loads(inner)\n'
    '\\1    except (json.JSONDecodeError, ValueError):\n'
    '\\1        continue\n'
    '\\1if isinstance(inner, list) and len(inner) == 1 and isinstance(inner[0], dict):\n'
    '\\1    inner = inner[0]\n'
    '\\1if isinstance(inner, dict):\n'
    '\\1    if set(inner.keys()).issubset(allowed) or any(k in allowed for k in inner.keys()):\n'
    '\\1        return json.dumps(inner, ensure_ascii=False)'
)


def patched_text(text: str) -> str:
    if MARKER in text:
        compile(text, "<patched deepseek_v4.py>", "exec")
        return text

    # Strip older v1 marker if present
    text = re.sub(r' *# spark-vllm mod: fix-deepseek-v4-tool-args-wrapper v1', '', text)

    loop_matches = list(WRAPPER_LOOP_RE.finditer(text))
    if len(loop_matches) != 1:
        raise ValueError(
            "expected exactly one supported wrapper loop in deepseek_v4.py; "
            f"found {len(loop_matches)}"
        )
    dict_matches = list(INNER_CHECK_RE.finditer(text))
    if len(dict_matches) != 1:
        raise ValueError(
            "expected exactly one supported inner-dict check in deepseek_v4.py; "
            f"found {len(dict_matches)}"
        )

    text = WRAPPER_LOOP_RE.sub(WRAPPER_LOOP_PATCH, text, count=1)
    text = INNER_CHECK_RE.sub(INNER_CHECK_PATCH, text, count=1)
    compile(text, "<patched deepseek_v4.py>", "exec")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    text = args.target.read_text(encoding="utf-8")
    if args.check:
        if MARKER in text:
            result = patched_text(text)  # verifies patched state, no rewrite
            assert result is text
            print(f"[fix-deepseek-v4-tool-args-wrapper] already patched: {args.target}")
        else:
            # Unpatched and check-only: validate that it is patchable.
            patched_text(text)
            print(f"[fix-deepseek-v4-tool-args-wrapper] unpatched (patchable): {args.target}")
    else:
        new_text = patched_text(text)
        if new_text == text:
            print(f"[fix-deepseek-v4-tool-args-wrapper] no change needed: {args.target}")
        else:
            args.target.write_text(new_text, encoding="utf-8")
            print(f"[fix-deepseek-v4-tool-args-wrapper] patched: {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

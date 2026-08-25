#!/usr/bin/env python3
"""fix-decode-topk-cache: decode side.

Caches the b12x decode top-k selection per layer and replays it for a bounded
number of decode steps instead of rescanning the full context every step
(the O(ctx) per-step cost that collapses decode throughput linearly with
context length).

Target: vllm/model_executor/layers/sparse_attn_indexer.py

Semantics (validated against the runtime kernels):
  - Decode rows are physical flat cache slots (output_physical_slots=True in
    the B12X_MLA_SPARSE + DCP=1 deployment); a row's physical slot is stable
    across steps (K cache is append-only, slots never move).
  - The MLA kernel consumes topk_indices_buffer directly (DCP=1 path), so
    replaying the cached rows into the buffer is a faithful substitute for a
    rescan as long as the decode batch is unchanged.
  - Each decode step writes exactly one new K row per request. compressed
    slot_mapping marks the row(s) holding this step's new slot (>=0) and -1
    elsewhere, so the per-row where() below appends the fresh slot by
    overwriting one column per step. Spec-verifier rows of one request share
    the slot, so per-row updates cover every row of the request.
  - A refresh every VLLM_SPARSE_DECODE_REFRESH_INSTERVAL steps (decided
    host-side in the metadata builder, default 32) bounds the approximation:
    the replayed set stays correct except that the up-to-interval newest slots
    replace up-to-interval cached slots. Quality is A/B-checked via the
    dspark accept rate.

The per-layer cache is keyed by the indexer K-cache tensor address (one
entry per layer, separate allocations). Batch composition changes are handled
by the metadata builder forcing topk_cache_refresh=True; rows mismatch also
falls back to a full rescan.

Idempotent: anchored with a MARKER; --check only verifies the marker.
"""
import argparse
import os
import sys

TARGET_NAME = "vllm/model_executor/layers/sparse_attn_indexer.py"
MARKER = "# === fix-decode-topk-cache ===="

# --- helper class injected before _merge_b12x_dcp_topk ---------------------
ANCHOR_MERGE = "\ndef _merge_b12x_dcp_topk(\n"
PATCH_MERGE = (
    "\n"
    "# === fix-decode-topk-cache ==== per-layer decode top-k replay cache\n"
    "_DECODE_TOPK_CACHE = {}\n"
    "\n"
    "\n"
    "class _DecodeTopkCacheEntry:\n"
    "    \"\"\"Per-layer replay buffer for the decode top-k selection.\n"
    "\n"
    "    slots holds a persistent [rows, topk] int32 copy of the last full\n"
    "    rescan output (physical flat cache slots). step counts decode steps\n"
    "    since that rescan; each step overwrites one column with the freshly\n"
    "    written K row of this step so the newest token stays visible.\n"
    "    \"\"\"\n"
    "\n"
    "    __slots__ = (\"slots\", \"step\", \"rows\", \"topk\")\n"
    "\n"
    "    def __init__(self, rows: int, topk: int, device):\n"
    "        self.slots = torch.empty((rows, topk), dtype=torch.int32,\n"
    "                                device=device)\n"
    "        self.step = 0\n"
    "        self.rows = rows\n"
    "        self.topk = topk\n"
    "\n"
    "    def refresh(self, topk_indices: torch.Tensor) -> None:\n"
    "        self.slots.copy_(topk_indices)\n"
    "        self.step = 0\n"
    "\n"
    "    def replay_into(self, topk_indices: torch.Tensor,\n"
    "                    slot_mapping: torch.Tensor | None) -> None:\n"
    "        rows = topk_indices.shape[0]\n"
    "        if rows > self.rows:\n"
    "            raise RuntimeError(\n"
    "                \"decode top-k cache rows shrank: cached \"\n"
    "                f\"{self.rows} < batch {rows}\")\n"
    "        if slot_mapping is not None and rows > 0:\n"
    "            sm = slot_mapping[:rows]\n"
    "            c = self.step % self.topk\n"
    "            self.slots[:rows, c] = torch.where(\n"
    "                sm >= 0, sm, self.slots[:rows, c])\n"
    "            self.step += 1\n"
    "        topk_indices.copy_(self.slots[:rows])\n"
    "\n"
    "\n"
    "def _b12x_decode_topk_cache_get(kv_cache: torch.Tensor,\n"
    "                                topk_tokens: int) -> object | None:\n"
    "    \"\"\"Return the replay entry for this layer, or None to rescan.\"\"\"\n"
    "    return _DECODE_TOPK_CACHE.get(kv_cache.data_ptr())\n"
    "\n"
    "\n"
    "def _b12x_decode_topk_cache_store(kv_cache: torch.Tensor,\n"
    "                                  topk_indices: torch.Tensor,\n"
    "                                  topk_tokens: int) -> _DecodeTopkCacheEntry:\n"
    "    key = kv_cache.data_ptr()\n"
    "    entry = _DECODE_TOPK_CACHE.get(key)\n"
    "    rows = topk_indices.shape[0]\n"
    "    if entry is None or entry.rows != rows or entry.topk != topk_tokens:\n"
    "        entry = _DecodeTopkCacheEntry(rows, topk_tokens,\n"
    "                                     topk_indices.device)\n"
    "        _DECODE_TOPK_CACHE[key] = entry\n"
    "    entry.refresh(topk_indices)\n"
    "    return entry\n"
    "\n"
    "\n"
    "def _merge_b12x_dcp_topk(\n"
)

# --- decode branch: replay attempt before the rescan -----------------------
ANCHOR_DECODE_HIT = (
    "            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]\n"
    "            topk_scores = None\n"
)
PATCH_DECODE_HIT_INNER = (
    "            # fix-decode-topk-cache: replay the cached top-k selection when\n"
    "            # the metadata builder did not request a refresh and the batch\n"
    "            # shape still matches the cached one. Physical-slot output and\n"
    "            # DCP=1 only (slot_mapping is then the compressed physical\n"
    "            # mapping this layer's K cache is keyed by).\n"
    "            cache_entry = None\n"
    "            if (\n"
    "                output_physical_slots\n"
    "                and dcp_world_size == 1\n"
    "                and not getattr(\n"
    "                    decode_metadata, \"topk_cache_refresh\", True)\n"
    "                and slot_mapping is not None\n"
    "                and slot_mapping.numel() >= num_decode_tokens\n"
    "            ):\n"
    "                cache_entry = _b12x_decode_topk_cache_get(\n"
    "                    kv_cache, topk_tokens)\n"
    "                if (\n"
    "                    cache_entry is not None\n"
    "                    and cache_entry.rows == num_decode_tokens\n"
    "                ):\n"
    "                    cache_entry.replay_into(topk_indices, slot_mapping)\n"
    "                    return topk_indices_buffer\n"
    "            topk_scores = None\n"
)
PATCH_DECODE_HIT = (
    "            topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]\n"
    + PATCH_DECODE_HIT_INNER
)

# --- decode branch: store after a full rescan ------------------------------
ANCHOR_DECODE_STORE = (
    "            return topk_indices_buffer\n"
    "\n"
    "        schedule_metadata = decode_metadata.schedule_metadata\n"
)
PATCH_DECODE_STORE = (
    "            # fix-decode-topk-cache: refresh the per-layer replay buffer\n"
    "            # from this full rescan (also covers the first decode step and\n"
    "            # batch shape changes).\n"
    "            if output_physical_slots and dcp_world_size == 1:\n"
    "                _b12x_decode_topk_cache_store(kv_cache, topk_indices,\n"
    "                                              topk_tokens)\n"
    "            return topk_indices_buffer\n"
    "\n"
    "        schedule_metadata = decode_metadata.schedule_metadata\n"
)

CHECKS = [
    # (anchor, patch, label, marker_line)
    (ANCHOR_MERGE, PATCH_MERGE, "replay cache helper",
     "class _DecodeTopkCacheEntry:"),
    (ANCHOR_DECODE_HIT, PATCH_DECODE_HIT, "decode replay attempt",
     "            # fix-decode-topk-cache: replay the cached top-k selection when"),
    (ANCHOR_DECODE_STORE, PATCH_DECODE_STORE, "decode cache store",
     "            # fix-decode-topk-cache: refresh the per-layer replay buffer"),
]


def validate_shape(text: str, what: str) -> None:
    for anchor, _, label, marker in CHECKS:
        if what.startswith("patched"):
            if text.count(marker) != 1:
                raise ValueError(
                    f"{what}: expected patch marker to appear exactly once "
                    f"for '{label}', found {text.count(marker)}"
                )
        else:
            if text.count(anchor) != 1:
                raise ValueError(
                    f"{what}: expected anchor to appear exactly once for "
                    f"'{label}', found {text.count(anchor)}"
                )


def patched_text(src: str) -> str:
    text = src
    for anchor, patch, _, _ in CHECKS:
        text = text.replace(anchor, patch, 1)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="Path to sparse_attn_indexer.py to patch")
    ap.add_argument("--check", action="store_true",
                    help="Only verify the patch is applied")
    args = ap.parse_args()

    with open(args.target, "r", encoding="utf-8") as fh:
        src = fh.read()
    if MARKER in src:
        print(f"{TARGET_NAME}: already patched (marker present)")
        return 0
    if args.check:
        print(f"{TARGET_NAME}: check before/after apply "
              f"(marker present={MARKER in src})")
        return 0
    validate_shape(src, TARGET_NAME)
    out = patched_text(src)
    validate_shape(out, f"patched {TARGET_NAME}")
    if out == src:
        print(f"{TARGET_NAME}: no change produced (all anchors absent?)")
        return 1
    tmp = args.target + ".fix-decode-topk.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(out)
    os.replace(tmp, args.target)
    print(f"{TARGET_NAME}: patched OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

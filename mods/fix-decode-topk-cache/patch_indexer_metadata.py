#!/usr/bin/env python3
"""fix-decode-topk-cache: metadata side.

Adds a host-computed ``topk_cache_refresh`` flag to the decode metadata so the
sparse_attn_indexer decode branch can replay cached per-layer top-k slots
instead of rescanning the full context every step (the O(ctx) per-step cost
that makes decode throughput collapse linearly with context length).

Target: vllm/v1/attention/backends/mla/indexer.py
  - DeepSeekV32IndexerDecodeMetadata gains ``topk_cache_refresh: bool``.
  - DeepseekV32IndexerMetadataBuilder.__init__ gains reuse-cache state:
      _decode_topk_refresh_interval (env VLLM_SPARSE_DECODE_REFRESH_INTERVAL,
      default 32, 0 = reuse disabled = always refresh),
      _decode_topk_steps_since_refresh, _decode_topk_batch_sig.
  - build() sets topk_cache_refresh: True when the decode batch changed, the
    refresh interval was hit, prefill is mixed into the batch, padding is
    required, the batch is too short for saturated top-k rows, or reuse is
    configured off. False otherwise (safe to replay cached slots).

Idempotent: anchored with a MARKER; --check only verifies the marker.
"""
import argparse
import os
import sys

TARGET_NAME = "vllm/v1/attention/backends/mla/indexer.py"
MARKER = "# === fix-decode-topk-cache ===="

ANCHOR_FIELD = (
    "    indices: torch.Tensor | None = None\n"
)
PATCH_FIELD = (
    "    indices: torch.Tensor | None = None\n"
    "    # Host-computed: True when decode top-k must be recomputed this step\n"
    "    # (batch change, refresh interval hit, prefill mixed in, padding, or\n"
    "    # reuse disabled). False permits replaying cached per-layer slots.\n"
    "    topk_cache_refresh: bool = False\n"
)

ANCHOR_INIT = (
    "        sm_count = num_compute_units(self.device.index)\n"
    "        self.num_sms = sm_count\n"
)
PATCH_INIT = (
    ANCHOR_INIT
    + MARKER + " reuse-cache state\n"
    "        self._decode_topk_refresh_interval = int(\n"
    "            os.environ.get(\"VLLM_SPARSE_DECODE_REFRESH_INTERVAL\", \"32\")\n"
    "        )\n"
    "        self._decode_topk_steps_since_refresh = 0\n"
    "        self._decode_topk_batch_sig = None\n"
)

ANCHOR_BUILD_CALL = (
    "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
)
# Insert the refresh computation right before the decode_metadata
# construction. Build the flag from host-side values only (capture-outside).
PATCH_BUILD_CALL = (
    MARKER + " refresh decision (host, capture-outside)\n"
    "            topk_cache_refresh = True\n"
    "            try:\n"
    "                seq_lens_np = common_attn_metadata._seq_lens_cpu\n"
    "                if seq_lens_np is None:\n"
    "                    seq_lens_np = common_attn_metadata.seq_lens_cpu_upper_bound\n"
    "                min_compressed_rows = int(\n"
    "                    seq_lens_np[:num_decodes].numpy().min()\n"
    "                    // max(1, self.compress_ratio))\n"
    "            except Exception:\n"
    "                min_compressed_rows = decode_topk_max_seq_len\n"
    "            index_topk = int(self.vllm_config.model_config.hf_config.index_topk)\n"
    "            if (\n"
    "                self.use_b12x_sparse_indexer\n"
    "                and self.dcp_world_size == 1\n"
    "                and num_decodes > 0\n"
    "                and num_prefills == 0\n"
    "                and not requires_padding\n"
    "                and self._decode_topk_refresh_interval > 0\n"
    "                and min_compressed_rows is not None\n"
    "                and min_compressed_rows > index_topk\n"
    "            ):\n"
    "                try:\n"
    "                    sig = (\n"
    "                        num_decodes,\n"
    "                        num_decode_tokens,\n"
    "                        tuple(int(x) for x in decode_lens_cpu.numpy())\n"
    "                    )\n"
    "                except Exception:\n"
    "                    sig = None\n"
    "                if sig is not None and sig == self._decode_topk_batch_sig:\n"
    "                    if (\n"
    "                        self._decode_topk_steps_since_refresh\n"
    "                        < self._decode_topk_refresh_interval\n"
    "                    ):\n"
    "                        topk_cache_refresh = False\n"
    "                    else:\n"
    "                        self._decode_topk_steps_since_refresh = 0\n"
    "                else:\n"
    "                    self._decode_topk_batch_sig = sig\n"
    "                    self._decode_topk_steps_since_refresh = 0\n"
    "                if self._decode_topk_batch_sig is not None:\n"
    "                    self._decode_topk_steps_since_refresh += 1\n"
    "            topk_cache_refresh = bool(topk_cache_refresh)\n"
    "            decode_metadata = DeepSeekV32IndexerDecodeMetadata(\n"
)

ANCHOR_FIELD_ARG = (
    "                indices=decode_indices,\n"
    "                global_seq_lens=global_seq_lens_for_decode,\n"
    "                active_width=active_width,\n"
    "            )\n"
)
PATCH_FIELD_ARG = (
    "                indices=decode_indices,\n"
    "                global_seq_lens=global_seq_lens_for_decode,\n"
    "                active_width=active_width,\n"
    "                topk_cache_refresh=topk_cache_refresh,\n"
    "            )\n"
)

CHECKS = [
    (ANCHOR_FIELD, PATCH_FIELD, "decode metadata field",
     "    topk_cache_refresh: bool = False"),
    (ANCHOR_INIT, PATCH_INIT, "builder reuse-cache state",
     "        self._decode_topk_batch_sig = None"),
    (ANCHOR_BUILD_CALL, PATCH_BUILD_CALL, "build() refresh decision",
     "            topk_cache_refresh = bool(topk_cache_refresh)"),
    (ANCHOR_FIELD_ARG, PATCH_FIELD_ARG, "decode metadata ctor field",
     "                topk_cache_refresh=topk_cache_refresh,"),
]


def validate_shape(text: str, what: str) -> None:
    for anchor, patch, label, marker in CHECKS:
        if what.startswith("patched"):
            # After patching, the anchor may legitimately be consumed by its
            # patch; require the patch's unique marker line to be present
            # exactly once.
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
    ap.add_argument("target", help="Path to indexer.py to patch")
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

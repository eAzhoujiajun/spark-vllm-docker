# DeepSeek V4 Flash Vision-Exp

Adds native image input support to this repository's `vllm-node-b12x` image.
The mod copies the checkpoint's DeepSeek V4 encoder, installs the Vision-Exp
ViT, aligner, and multimodal processor, maps the vision checkpoint weights, and
enables OpenAI Chat Completions `image_url` content.

## Provenance (combined from two upstream implementations)

This is a **combination of the two published DGX Spark ports**, not a single
vendored copy:

- **`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`** — the monkey-patch
  mechanism and most of the runtime plumbing:
  - `vision_exp/apply.py` — `apply_vision_exp()` monkey-patch: tower install,
    `fused_topk_bias_split_vl` bias_vl dual-route (text/image/mixed), CUDA-graph
    capture guard, per-forward routing-kind cell (issue #175),
    `requires_raw_input_tokens=True`, `SupportsMultiModal` injection, processor
    registration.
  - `vision_exp/image_processor.py` — official image preprocessor port
    (resize solver, patchify, N-layout block build, COMPRESS_PAD_TO=4),
    issue #172 hash salt, `as_pil` input normalisation.
  - `vision_exp/vision.py` — ViT (32 blocks, 2D RoPE, fp32 RMSNorm) + Aligner,
    pure PyTorch, not TP-sharded (~410M params replicated per rank).
  - `hotfix-dsv4-vision-exp.py` — idempotent startup patch: imports
    `apply_vision_exp` at the end of `nvidia/model.py`, patches the encoder
    (images only in user messages → 400 elsewhere), remaps DSpark draft
    `ffn.gate.bias_vl` → `e_score_correction_bias_vl`.
- **`tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark`**
  — fixes and disciplines adopted here:
  - `modelinfos/` inspection-cache clearing (trap #7; both per-boot and
    persistent `/models`-mounted cache) — see `run.sh`.
  - W1/W3 stacked-params-mapping guard so the ViT MLP / aligner `w1`/`w2` are
    not rewritten into `gate_up_proj` (trap #2).
  - `image_token_index` published from the tokenizer's `<|deepseek_image|>` id
    so the DSpark proposer finds the standard VLM field (trap #4).
  - Loader guard and hash-MoE `gate.bias` handling (traps #3/#12).

## a50ebee1d adaptation (why this differs from either upstream)

Both upstreams are written against a vLLM version whose processor dispatches
through a `_call_hf_processor` hook. This repository's `vllm-node-b12x` image
(a50ebee1d, vLLM `0.1.dev20489+ga50ebee1d`) has **no such hook**: its
`BaseMultiModalProcessor._apply_hf_processor_main`
(`vllm/multimodal/processing/processor.py:1226`) calls
`info.get_hf_processor()` unconditionally whenever mm data is present.

So `vision_exp/processor.py` overrides **`_apply_hf_processor`** (the real
engine-invoked entry) and builds the `MultiModalProcessingInfo` directly from
the tokenized prompt + image items — running the same tail the base would
(`from_hf_inputs` → `get_mm_hashes` → `_get_mm_prompt_updates`) without ever
touching `get_hf_processor`. The `_cached_apply_hf_processor` override keeps the
issue #172 hash salting and the image-count short-circuit.

## Runtime requirements

The model must be mounted at `/models/DeepSeek-V4-Flash-Vision-Exp`, or the
container environment variable `DEEPSEEK_VISION_MODEL_PATH` must point to it.
DSpark speculative tokens must be a multiple of the checkpoint's
`num_nextn_predict_layers=3` (use `num_speculative_tokens: 6`).

After starting a newly built image or clearing its JIT caches, prime the real
prefill scheduler paths before benchmarking:

```bash
./examples/warmup-deepseek-v4-prefill.sh
```

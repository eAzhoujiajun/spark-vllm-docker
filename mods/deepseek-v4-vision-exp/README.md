# DeepSeek V4 Flash Vision-Exp

Adds native image input support to this repository's `vllm-node-b12x` image.
The mod copies the checkpoint's DeepSeek V4 encoder, installs the Vision-Exp
ViT, aligner, and multimodal processor, maps the vision checkpoint weights, and
enables OpenAI Chat Completions `image_url` content.

The implementation is vendored from
`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`. It supports still images only;
the checkpoint does not include a video encoder.

The model must be mounted at `/models/DeepSeek-V4-Flash-Vision-Exp`, or the
container environment variable `DEEPSEEK_VISION_MODEL_PATH` must point to it.

After starting a newly built image or clearing its JIT caches, prime the real
prefill scheduler paths before benchmarking:

```bash
./examples/warmup-deepseek-v4-prefill.sh
```
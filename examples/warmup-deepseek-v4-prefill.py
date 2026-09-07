#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prime DeepSeek V4 prefill kernels through the live API."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    parser.add_argument(
        "--model-path", default="/models/DeepSeek-V4-Flash-Vision-Exp"
    )
    parser.add_argument(
        "--lengths",
        default="512,1024,2048,4096,8192,16384,32768,65536,131072",
    )
    return parser.parse_args()


def wait_until_healthy(url: str) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"Server did not become healthy within 1800s: {url}")


def main() -> int:
    args = parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_ids = tokenizer.encode(
        "Prefill kernel warmup 0123456789 abcdefghijklmnopqrstuvwxyz. ",
        add_special_tokens=False,
    )
    wait_until_healthy(args.url)

    for target in lengths:
        nonce_ids = tokenizer.encode(
            f"warmup-length={target} nonce={time.time_ns()} ",
            add_special_tokens=False,
        )
        token_ids = (
            nonce_ids
            + base_ids * ((target - len(nonce_ids)) // len(base_ids) + 1)
        )[:target]
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": tokenizer.decode(token_ids, skip_special_tokens=True),
                }
            ],
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False},
        }
        request = urllib.request.Request(
            f"{args.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=900) as response:
            response.read()
        print(
            f"warmed prefill_tokens={target} elapsed_s={time.monotonic() - started:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Emit the fixed 16-prompt, 2,048-token A/B qualification as JSONL."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "def add(a, b):",
    "def reverse_string(s):",
    "The Treaty of Westphalia, signed in 1648,",
    "Explain photosynthesis in two paragraphs.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Return a compact JSON object describing a red bicycle.",
    "Explain why there are infinitely many prime numbers.",
    "Write a safe Rust function that sums a slice of i64 values.",
    "Write SQL selecting the five highest paid employees by department.",
    "Translate 'Knowledge is power' into Spanish and French.",
    "A train travels 120 km in 80 minutes. Its average speed is",
    "Write a four-sentence science-fiction story about a lunar library.",
    "Create a minimal accessible HTML button labeled Submit.",
    "Explain conservation of momentum with one concrete example.",
]


def request_json(url: str, payload: dict, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def metrics_snapshot(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode(errors="replace")
    except OSError as error:
        return [f"metrics_unavailable:{type(error).__name__}"]
    return [
        line
        for line in text.splitlines()
        if "spec" in line.lower() and not line.startswith("# HELP")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, choices=("target", "dspark"))
    parser.add_argument("--api-key")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    completions_url = f"{base_url}/v1/completions"
    metrics_url = f"{base_url}/metrics"
    print(
        json.dumps(
            {
                "event": "metrics_before",
                "label": args.label,
                "lines": metrics_snapshot(metrics_url),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index, prompt in enumerate(PROMPTS):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "ignore_eos": True,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        started = time.perf_counter()
        result = request_json(completions_url, payload, args.api_key)
        wall = time.perf_counter() - started
        choice = result["choices"][0]
        encoded_tokens = choice["logprobs"]["tokens"]
        if not all(token.startswith("token_id:") for token in encoded_tokens):
            raise RuntimeError("server did not return exact token IDs")
        token_ids = [
            int(token.removeprefix("token_id:")) for token in encoded_tokens
        ]
        if len(token_ids) != 128:
            raise RuntimeError(f"prompt {index} returned {len(token_ids)} tokens")
        print(
            json.dumps(
                {
                    "event": "prompt",
                    "label": args.label,
                    "index": index,
                    "prompt": prompt,
                    "token_ids": token_ids,
                    "text_sha256": __import__("hashlib").sha256(
                        choice["text"].encode()
                    ).hexdigest(),
                    "text_bytes": len(choice["text"].encode()),
                    "wall_seconds": wall,
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "metrics_after",
                "label": args.label,
                "lines": metrics_snapshot(metrics_url),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

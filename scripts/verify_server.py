#!/usr/bin/env python3
"""Verify the sealed one-token parity contract through the local vLLM API."""

from __future__ import annotations

import argparse
import json
import urllib.request


PROMPT = "The capital of France is"
PROMPT_IDS = [1008, 10484, 318, 15383, 387]
EXPECTED_TEXT = " Paris"
EXPECTED_TOKEN_ID = 17374


def post(url: str, payload: dict, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    args = parser.parse_args()

    tokenized = post(
        f"{args.base_url}/tokenize",
        {"model": args.model, "prompt": PROMPT, "add_special_tokens": False},
        args.api_key,
    )
    assert tokenized["tokens"] == PROMPT_IDS, tokenized

    for run in range(1, 4):
        completion = post(
            f"{args.base_url}/v1/completions",
            {
                "model": args.model,
                "prompt": PROMPT_IDS,
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
            },
            args.api_key,
        )
        text = completion["choices"][0]["text"]
        assert text == EXPECTED_TEXT, {"run": run, "text": text}

    expected = post(
        f"{args.base_url}/tokenize",
        {"model": args.model, "prompt": EXPECTED_TEXT, "add_special_tokens": False},
        args.api_key,
    )
    assert expected["tokens"] == [EXPECTED_TOKEN_ID], expected
    print("PASS: sealed France token 17374 (' Paris') matched in 3/3 runs")


if __name__ == "__main__":
    main()

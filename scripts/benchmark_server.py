#!/usr/bin/env python3
"""Reproduce the exact published K3 template/fixed-output benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from jinja2 import BaseLoader, Environment


EXPECTED_TEMPLATE_SHA256 = (
    "05bb501f8ac31fa6b0bf04803b5ada49abf9cdd51c3c90a4719b739df0000722"
)
EXPECTED_TEMPLATE_BYTES = 24696
EXPECTED_PROMPT_TOKENS = 105
REFERENCE_OUTPUT_SHA256 = {
    64: "9af0ea8e070bdd0db2b36cf41fe790eea2599be9608e71e129abe1e507ccdd0c",
    256: "b85de5bae72080bed1eb3c28d68da4f19e80f2aa45bd7c0bf141f30cb30244b1",
}
USER_MESSAGE = (
    "Explain how photosynthesis works, including both the light-dependent "
    "reactions and the Calvin cycle."
)


def render_prompt(template_path: Path) -> tuple[str, str, int]:
    template_bytes = template_path.read_bytes()
    digest = hashlib.sha256(template_bytes).hexdigest()
    if digest != EXPECTED_TEMPLATE_SHA256 or len(template_bytes) != EXPECTED_TEMPLATE_BYTES:
        raise RuntimeError(
            "chat template does not match the pinned 24,696-byte K3 template"
        )
    environment = Environment(loader=BaseLoader())
    environment.globals["raise_exception"] = lambda message: (
        _ for _ in ()
    ).throw(Exception(message))
    template = environment.from_string(template_bytes.decode())
    prompt = template.render(
        messages=[{"role": "user", "content": USER_MESSAGE}],
        add_generation_prompt=True,
        tools=None,
        thinking_effort="max",
    )
    return prompt, digest, len(template_bytes)


def run_once(
    repetition: int,
    endpoint: str,
    model: str,
    rendered_prompt: str,
    template_path: Path,
    template_sha256: str,
    template_bytes: int,
    max_tokens: int,
    api_key: str | None,
) -> dict:
    payload = {
        "model": model,
        "prompt": rendered_prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first_emission = None
    last_emission = None
    emitted_chunks = 0
    output: list[str] = []
    usage = None
    finish_reason = None
    try:
        response_context = urllib.request.urlopen(request, timeout=1800)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error
    with response_context as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            piece = choice.get("text") or ""
            if not piece:
                continue
            now = time.perf_counter()
            first_emission = first_emission or now
            last_emission = now
            emitted_chunks += 1
            output.append(piece)
    finished = time.perf_counter()
    if first_emission is None or last_emission is None or usage is None:
        raise RuntimeError("stream did not return timed output and final usage")

    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    if prompt_tokens != EXPECTED_PROMPT_TOKENS:
        raise RuntimeError(
            f"rendered prompt contract changed: expected 105 tokens, got {prompt_tokens}"
        )
    if completion_tokens != max_tokens:
        raise RuntimeError(
            f"fixed output truncated: expected {max_tokens}, got {completion_tokens}"
        )

    ttft = first_emission - started
    decode_seconds = last_emission - first_emission
    decode_tokens = completion_tokens - 1
    output_text = "".join(output)
    output_sha256 = hashlib.sha256(output_text.encode()).hexdigest()
    return {
        "repetition": repetition,
        "contract": {
            "endpoint": endpoint,
            "model": model,
            "user_message": USER_MESSAGE,
            "chat_template": str(template_path),
            "chat_template_sha256": template_sha256,
            "chat_template_bytes": template_bytes,
            "thinking_effort": "max",
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "stream": True,
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "stream_emission_chunks": emitted_chunks,
        "time_to_first_token_seconds": ttft,
        "approx_prompt_processing_tokens_per_second": prompt_tokens / ttft,
        "decode_interval_seconds_first_to_last": decode_seconds,
        "decode_tokens_after_first": decode_tokens,
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "end_to_end_seconds": finished - started,
        "end_to_end_completion_tokens_per_second": completion_tokens
        / (finished - started),
        "finish_reason": finish_reason,
        "output_sha256": output_sha256,
        "matches_published_target_output": (
            output_sha256 == REFERENCE_OUTPUT_SHA256.get(max_tokens)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", required=True)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--tokens", type=int, choices=(64, 256), default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--api-key")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be at least one")

    prompt, template_sha, template_bytes = render_prompt(args.chat_template)
    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"
    runs = [
        run_once(
            repetition,
            endpoint,
            args.model,
            prompt,
            args.chat_template,
            template_sha,
            template_bytes,
            args.tokens,
            args.api_key,
        )
        for repetition in range(1, args.repetitions + 1)
    ]
    report = {
        "runs": runs,
        "median_decode_tokens_per_second": statistics.median(
            run["decode_tokens_per_second"] for run in runs
        ),
        "median_end_to_end_tokens_per_second": statistics.median(
            run["end_to_end_completion_tokens_per_second"] for run in runs
        ),
        "all_outputs_byte_identical": len(
            {run["output_sha256"] for run in runs}
        )
        == 1,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

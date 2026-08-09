#!/usr/bin/env python3
"""Combine A/B parity, fixed-output speed, and the promotion rule."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def qualification_rows(path: Path) -> list[dict]:
    return [
        row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row.get("event") == "prompt"
    ]


def benchmark_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_qualification", type=Path)
    parser.add_argument("dspark_qualification", type=Path)
    parser.add_argument("target_benchmark", type=Path)
    parser.add_argument("dspark_benchmark", type=Path)
    args = parser.parse_args()

    target_q = qualification_rows(args.target_qualification)
    dspark_q = qualification_rows(args.dspark_qualification)
    target_s = benchmark_runs(args.target_benchmark)
    dspark_s = benchmark_runs(args.dspark_benchmark)
    assert len(target_q) == len(dspark_q) == 16
    assert len(target_s) == len(dspark_s) == 3
    for expected, actual in zip(target_q, dspark_q, strict=True):
        assert expected["token_ids"] == actual["token_ids"]
        assert expected["text_sha256"] == actual["text_sha256"]
    for expected, actual in zip(target_s, dspark_s, strict=True):
        assert expected["output_sha256"] == actual["output_sha256"]

    target_median = statistics.median(
        row["decode_tokens_per_second"] for row in target_s
    )
    dspark_median = statistics.median(
        row["decode_tokens_per_second"] for row in dspark_s
    )
    point_speedup = dspark_median / target_median

    random_generator = random.Random(0)
    bootstrap: list[float] = []
    for _ in range(20_000):
        indices = [random_generator.randrange(16) for _ in range(16)]
        target_time = sum(target_q[index]["wall_seconds"] for index in indices)
        dspark_time = sum(dspark_q[index]["wall_seconds"] for index in indices)
        bootstrap.append(target_time / dspark_time)
    bootstrap.sort()
    lower_95 = bootstrap[int(0.025 * len(bootstrap))]

    result = {
        "target_median_decode_tokens_per_second": target_median,
        "dspark_median_decode_tokens_per_second": dspark_median,
        "point_speedup": point_speedup,
        "prompt_bootstrap_lower_95_speedup": lower_95,
        "exact_prompt_token_parity": "2048/2048",
        "fixed_256_output_hash_parity": "3/3",
        "promote": point_speedup >= 1.15 and lower_95 > 1.0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

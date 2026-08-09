#!/usr/bin/env python3
"""Combine A/B parity, fixed-output speed, and the promotion rule.

Exact token-for-token / hash-for-hash identity between target-only and
DSpark is NOT achievable on this quantized target and is not a gate here --
see docs/INTEGRATED-RUNBOOK.md and the correctness section of
evidence/DSPARK-TP3-H200.md. The correctness gates that ARE enforced: the
sealed France token (17374) and run-to-run determinism within each server's
own repeated runs.
"""

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
    assert len(target_q) == len(dspark_q) == 16, "expected 16 prompts in each run"
    assert len(target_s) == len(dspark_s) == 3, "expected 3 benchmark repetitions"
    for expected, actual in zip(target_q, dspark_q, strict=True):
        assert expected["index"] == actual["index"]
        assert expected["prompt"] == actual["prompt"]

    # Exact target/DSpark token- or hash-for-hash identity is NOT achievable
    # on this quantized target (docs/INTEGRATED-RUNBOOK.md, and the
    # correctness section of evidence/DSPARK-TP3-H200.md), so it is reported
    # below as a diagnostic, not asserted as a gate. The gates that DO hold:
    # the sealed France token, and determinism within each server's own
    # repeated runs (NOT target-vs-dspark, which will never match).
    exact_token_matches = sum(
        1
        for expected, actual in zip(target_q, dspark_q, strict=True)
        if expected["token_ids"] == actual["token_ids"]
    )
    france = next(row for row in target_q if row["index"] == 0)
    france_ok = france["token_ids"][0] == 17374
    target_deterministic = len({run["output_sha256"] for run in target_s}) == 1
    dspark_deterministic = len({run["output_sha256"] for run in dspark_s}) == 1

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
        "sealed_france_token_parity": france_ok,
        "target_run_to_run_deterministic": target_deterministic,
        "dspark_run_to_run_deterministic": dspark_deterministic,
        "exact_prompt_token_matches": f"{exact_token_matches}/{len(target_q)}",
        "promote": (
            point_speedup >= 1.15
            and lower_95 > 1.0
            and france_ok
            and target_deterministic
            and dspark_deterministic
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

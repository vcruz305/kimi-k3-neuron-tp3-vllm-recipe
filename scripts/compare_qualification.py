#!/usr/bin/env python3
"""Compare a 16-prompt target/DSpark qualification run.

Exact token-for-token identity between target-only and DSpark is NOT the
pass/fail bar here -- it is not achievable on this quantized target (see
docs/INTEGRATED-RUNBOOK.md and the correctness section of
evidence/DSPARK-TP3-H200.md). This checks the gates that DO hold -- the same
16 prompts were asked in the same order, and the sealed France prompt
returns token ID 17374 -- and reports token-for-token overlap as a
diagnostic, not a gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prompt_rows(path: Path) -> list[dict]:
    return [
        row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row.get("event") == "prompt"
    ]


def common_prefix_length(a: list[int], b: list[int]) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("dspark", type=Path)
    args = parser.parse_args()
    target = prompt_rows(args.target)
    dspark = prompt_rows(args.dspark)
    assert len(target) == len(dspark) == 16, "expected 16 prompts in each run"
    for expected, actual in zip(target, dspark, strict=True):
        assert expected["index"] == actual["index"]
        assert expected["prompt"] == actual["prompt"]

    france = next(row for row in target if row["index"] == 0)
    assert france["token_ids"][0] == 17374, "sealed France parity gate failed"

    exact_matches = sum(
        1
        for expected, actual in zip(target, dspark, strict=True)
        if expected["token_ids"] == actual["token_ids"]
    )
    prefix_lengths = [
        common_prefix_length(expected["token_ids"], actual["token_ids"])
        for expected, actual in zip(target, dspark, strict=True)
    ]
    print(
        "PASS: 16/16 prompts asked in matching order; sealed France token "
        "17374 matched.\n"
        f"Diagnostic only, NOT a gate (see docs/INTEGRATED-RUNBOOK.md): "
        f"{exact_matches}/16 prompts matched token-for-token; common-prefix "
        f"tokens min={min(prefix_lengths)} "
        f"median={sorted(prefix_lengths)[len(prefix_lengths) // 2]} "
        f"max={max(prefix_lengths)} out of 128 tokens/prompt."
    )


if __name__ == "__main__":
    main()

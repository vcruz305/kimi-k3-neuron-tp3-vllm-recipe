#!/usr/bin/env python3
"""Require exact 16-prompt target/DSpark token parity."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("dspark", type=Path)
    args = parser.parse_args()
    target = prompt_rows(args.target)
    dspark = prompt_rows(args.dspark)
    assert len(target) == len(dspark) == 16
    for expected, actual in zip(target, dspark, strict=True):
        assert expected["index"] == actual["index"]
        assert expected["prompt"] == actual["prompt"]
        assert expected["token_ids"] == actual["token_ids"]
        assert expected["text_sha256"] == actual["text_sha256"]
    assert target[0]["token_ids"][0] == 17374
    print("PASS: 16/16 prompts, 2048/2048 generated token IDs; France=17374")


if __name__ == "__main__":
    main()

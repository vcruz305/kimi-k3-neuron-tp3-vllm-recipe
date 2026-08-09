#!/usr/bin/env python3
"""Verify the four precision-sensitive Kimi-K3 constructor contracts."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


EXPECTED_PREFIXES = {
    "{prefix}.gate": "GateLinear",
    "{prefix}.self_attention_res_proj": "ReplicatedLinear",
    "{prefix}.mlp_res_proj": "ReplicatedLinear",
    "{prefix}.output_attn_res_proj": "ReplicatedLinear",
}


def string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue) and isinstance(
                value.value, ast.Name
            ):
                parts.append("{" + value.value.id + "}")
            else:
                return None
        return "".join(parts)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="path to vllm/models/kimi_k3/nvidia/model.py",
    )
    args = parser.parse_args()
    tree = ast.parse(args.source.read_text(encoding="utf-8"))
    found: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        prefix = string_value(keywords.get("prefix", ast.Constant(None)))
        expected_class = EXPECTED_PREFIXES.get(prefix)
        if expected_class is None:
            continue
        if node.func.id != expected_class:
            raise AssertionError(
                f"{prefix}: expected {expected_class}, found {node.func.id}"
            )
        dtype = keywords.get("params_dtype")
        if not (
            isinstance(dtype, ast.Attribute)
            and isinstance(dtype.value, ast.Name)
            and dtype.value.id == "torch"
            and dtype.attr == "float32"
        ):
            raise AssertionError(f"{prefix}: params_dtype is not torch.float32")
        if prefix == "{prefix}.gate":
            out_dtype = keywords.get("out_dtype")
            if not (
                isinstance(out_dtype, ast.Attribute)
                and isinstance(out_dtype.value, ast.Name)
                and out_dtype.value.id == "torch"
                and out_dtype.attr == "float32"
            ):
                raise AssertionError("router out_dtype is not torch.float32")
        found.add(prefix)

    missing = set(EXPECTED_PREFIXES) - found
    if missing:
        raise AssertionError(f"missing precision constructors: {sorted(missing)}")
    print("PASS: Kimi-K3 router and all AttnRes projections preserve FP32")


if __name__ == "__main__":
    main()

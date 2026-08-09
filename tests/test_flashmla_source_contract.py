"""Static CPU gate for the optional Hopper FlashMLA DSpark patch."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    path = args.source_root / "vllm/v1/attention/backends/mla/flashmla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    backend = class_node(tree, "FlashMLABackend")
    supports_non_causal = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "supports_non_causal"
    )
    assert isinstance(supports_non_causal.body[0], ast.Return)
    assert isinstance(supports_non_causal.body[0].value, ast.Constant)
    assert supports_non_causal.body[0].value.value is True

    builder = class_node(tree, "FlashMLAMetadataBuilder")
    non_causal_assignment = next(
        node
        for node in builder.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "supports_non_causal_multi_token_decode"
    )
    assert isinstance(non_causal_assignment.value, ast.Constant)
    assert non_causal_assignment.value.value is True

    init = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    head_assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "num_q_heads"
            for target in node.targets
        )
    ]
    assert len(head_assignments) == 1
    value = head_assignments[0].value
    assert isinstance(value, ast.Attribute) and value.attr == "num_heads"

    impl = class_node(tree, "FlashMLAImpl")
    forward = next(
        node
        for node in impl.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_mqa"
    )
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {"flash_mla_with_kvcache", "flash_mla_with_kvcache_fp8"}
    ]
    assert len(calls) == 2
    for call in calls:
        causal = next(
            keyword.value for keyword in call.keywords if keyword.arg == "causal"
        )
        assert isinstance(causal, ast.Attribute) and causal.attr == "causal"

    print("PASS: Hopper FlashMLA non-causal DSpark source contract")


if __name__ == "__main__":
    main()

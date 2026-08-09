"""CPU-only contract for independent DSpark draft config discovery."""

from __future__ import annotations

import argparse
import ast
import types
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()

    path = args.source_root / "vllm/config/speculative.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpeculativeConfig"
    )
    post_init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )

    candidates: list[ast.expr] = []
    for call in ast.walk(post_init):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "ModelConfig":
            continue
        for keyword in call.keywords:
            if keyword.arg == "config_format" and "draft_load_config" in ast.unparse(
                keyword.value
            ):
                candidates.append(keyword.value)

    assert len(candidates) == 1, (
        "expected one draft ModelConfig config_format expression tied to "
        f"draft_load_config, found {len(candidates)}"
    )
    expression = candidates[0]
    assert isinstance(expression, ast.IfExp)
    compiled = compile(
        ast.fix_missing_locations(ast.Expression(expression)), str(path), "eval"
    )

    target = types.SimpleNamespace(config_format="gguf")
    self_ = types.SimpleNamespace(
        draft_load_config=object(), target_model_config=target
    )
    assert eval(compiled, {}, {"self": self_}) == "auto"
    self_.draft_load_config = None
    assert eval(compiled, {}, {"self": self_}) == "gguf"

    print("PASS: explicit draft loader forces independent config discovery")


if __name__ == "__main__":
    main()

"""CPU-only source-contract tests for the K3 DSpark TP3 patch."""

from __future__ import annotations

import argparse
import ast
import types
from pathlib import Path
from typing import Iterable


class FakeTensor:
    def __init__(self, data: list[list[int]]) -> None:
        self.data = [list(row) for row in data]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.data), len(self.data[0]) if self.data else 0

    def new_zeros(self, shape: list[int]) -> "FakeTensor":
        return FakeTensor([[0] * shape[1] for _ in range(shape[0])])

    def split(self, size: int, dim: int = 0) -> tuple["FakeTensor", ...]:
        assert dim == 0
        assert self.shape[0] % size == 0
        return tuple(
            FakeTensor(self.data[start : start + size])
            for start in range(0, self.shape[0], size)
        )


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def cat(tensors: tuple[FakeTensor, ...], dim: int) -> FakeTensor:
        if dim == 0:
            return FakeTensor([row for tensor in tensors for row in tensor.data])
        assert dim == 1
        rows = []
        for parts in zip(*(tensor.data for tensor in tensors), strict=True):
            rows.append([value for part in parts for value in part])
        return FakeTensor(rows)


def extract_functions(path: Path, names: set[str], globals_: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    missing = names - {node.name for node in selected}
    assert not missing, f"Missing source functions in {path}: {sorted(missing)}"
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_zero_padding(source_root: Path) -> None:
    path = source_root / "vllm/models/kimi_k3/nvidia/dspark_mla.py"
    ns = extract_functions(
        path,
        {"_zero_pad_dimension", "_pad_k3_dspark_tp_weights"},
        {"torch": FakeTorch, "Iterable": Iterable},
    )
    pad = ns["_pad_k3_dspark_tp_weights"]
    config = types.SimpleNamespace(
        num_attention_heads=3,
        intermediate_size=3,
        _vllm_checkpoint_num_attention_heads=2,
        _vllm_checkpoint_intermediate_size=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
    )
    weights = [
        ("x.self_attn.q_b_proj.weight", FakeTensor([[1], [2], [3], [4]])),
        ("x.self_attn.kv_b_proj.weight", FakeTensor([[5], [6], [7], [8]])),
        ("x.self_attn.o_proj.weight", FakeTensor([[9, 10], [11, 12]])),
        ("x.mlp.gate_proj.weight", FakeTensor([[13], [14]])),
        ("x.mlp.up_proj.weight", FakeTensor([[15], [16]])),
        ("x.mlp.down_proj.weight", FakeTensor([[17, 18], [19, 20]])),
        (
            "x.mlp.gate_up_proj.weight",
            FakeTensor([[21], [22], [23], [24]]),
        ),
    ]
    padded = dict(pad(weights, config))
    assert padded["x.self_attn.q_b_proj.weight"].data[-2:] == [[0], [0]]
    assert padded["x.self_attn.kv_b_proj.weight"].data[-2:] == [[0], [0]]
    assert padded["x.self_attn.o_proj.weight"].data == [
        [9, 10, 0],
        [11, 12, 0],
    ]
    assert padded["x.mlp.gate_proj.weight"].data == [[13], [14], [0]]
    assert padded["x.mlp.up_proj.weight"].data == [[15], [16], [0]]
    assert padded["x.mlp.down_proj.weight"].data == [
        [17, 18, 0],
        [19, 20, 0],
    ]
    assert padded["x.mlp.gate_up_proj.weight"].data == [
        [21],
        [22],
        [0],
        [23],
        [24],
        [0],
    ]


def test_config_padding(source_root: Path) -> None:
    path = source_root / "vllm/config/speculative.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpeculativeConfig"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pad_k3_dspark_for_tp"
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    log_messages: list[tuple] = []
    logger = types.SimpleNamespace(
        warning=lambda *args: log_messages.append(args),
    )
    namespace = {"ModelConfig": object, "logger": logger}
    exec(compile(module, str(path), "exec"), namespace)

    hf = types.SimpleNamespace(num_attention_heads=64, intermediate_size=14336)
    arch = types.SimpleNamespace(total_num_attention_heads=64)
    model = types.SimpleNamespace(hf_config=hf, model_arch_config=arch)
    namespace["_pad_k3_dspark_for_tp"](model, 3)
    assert hf.num_attention_heads == 66
    assert hf.intermediate_size == 14337
    assert hf._vllm_checkpoint_num_attention_heads == 64
    assert hf._vllm_checkpoint_intermediate_size == 14336
    assert arch.total_num_attention_heads == 66
    assert len(log_messages) == 1


def test_draft_loader_isolation(source_root: Path) -> None:
    path = source_root / "vllm/model_executor/models/utils.py"

    class VllmConfig:
        @staticmethod
        def get_quantization_config(model_config, load_config):
            return model_config, load_config

    ns = extract_functions(
        path,
        {"get_draft_quant_config"},
        {"VllmConfig": VllmConfig},
    )
    get_quant = ns["get_draft_quant_config"]
    draft_model = object()
    spec = types.SimpleNamespace(
        draft_model_config=draft_model,
        draft_load_config="safetensors",
    )
    config = types.SimpleNamespace(speculative_config=spec, load_config="gguf")
    assert get_quant(config) == (draft_model, "safetensors")
    spec.draft_load_config = None
    assert get_quant(config) == (draft_model, "gguf")

    loader_path = (
        source_root / "vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
    )
    tree = ast.parse(loader_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_dspark_model"
    )
    get_model_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_model"
    ]
    assert len(get_model_calls) == 1
    keywords = {keyword.arg for keyword in get_model_calls[0].keywords}
    assert {"vllm_config", "model_config", "load_config"} <= keywords


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    test_zero_padding(args.source_root)
    test_config_padding(args.source_root)
    test_draft_loader_isolation(args.source_root)
    print("PASS: K3 DSpark TP3 source contract (3 tests)")


if __name__ == "__main__":
    main()

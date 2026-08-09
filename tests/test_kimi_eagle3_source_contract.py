"""CPU-only contract tests for KimiLinear's EAGLE3 target bridge."""

from __future__ import annotations

import argparse
import ast
import types
from pathlib import Path
from typing import ClassVar, Literal, Protocol, runtime_checkable


def find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def base_names(cls: ast.ClassDef) -> set[str]:
    return {ast.unparse(base) for base in cls.bases}


def test_runtime_protocol_and_default_method(source_root: Path) -> None:
    interfaces_path = source_root / "vllm/model_executor/models/interfaces.py"
    interfaces_tree = ast.parse(interfaces_path.read_text(encoding="utf-8"))
    mixin = find_class(interfaces_tree, "EagleModelMixin")
    protocol = find_class(interfaces_tree, "SupportsEagle3")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            mixin,
            protocol,
        ],
        type_ignores=[],
    )

    @runtime_checkable
    class SupportsEagleBase(Protocol):
        pass

    namespace = {
        "ClassVar": ClassVar,
        "Literal": Literal,
        "Protocol": Protocol,
        "SupportsEagleBase": SupportsEagleBase,
        "runtime_checkable": runtime_checkable,
    }
    exec(compile(ast.fix_missing_locations(module), str(interfaces_path), "exec"), namespace)

    EagleModelMixin = namespace["EagleModelMixin"]
    SupportsEagle3 = namespace["SupportsEagle3"]

    class Inner(EagleModelMixin):
        def __init__(self) -> None:
            self.layers = [object()] * 93

    class Target(SupportsEagle3):
        def __init__(self) -> None:
            self.model = Inner()

    target = Target()
    assert isinstance(target, SupportsEagle3)
    boundaries = (3, 24, 48, 72, 90)
    target.set_aux_hidden_state_layers(boundaries)
    assert target.model.aux_hidden_state_layers == boundaries
    assert target.get_eagle3_default_aux_hidden_state_layers() == (2, 46, 90)


def test_kimi_hidden_state_contract(source_root: Path) -> None:
    model_path = source_root / "vllm/models/kimi_k3/nvidia/model.py"
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    inner = find_class(tree, "KimiLinearModel")
    wrapper = find_class(tree, "KimiLinearForCausalLM")
    assert "EagleModelMixin" in base_names(inner)
    assert "SupportsEagle3" in base_names(wrapper)

    forward = find_method(inner, "forward")
    comparisons = {
        ast.unparse(node.left)
        for node in ast.walk(forward)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(op, ast.In) for op in node.ops
        )
        and any(
            ast.unparse(comparator) == "self.aux_hidden_state_layers"
            for comparator in node.comparators
        )
    }
    assert {"self.start_layer", "layer_idx + 1"} <= comparisons

    tuple_returns = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and ast.unparse(node.value) == "(hidden_states, aux_hidden_states)"
    ]
    assert len(tuple_returns) == 1

    # The increasing decoder loop plus append gives the context projection the
    # exact training order: after target layers 2, 23, 47, 71, and 89.
    loops = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.For)
        and "enumerate(self.layers[self.start_layer:self.end_layer]" in ast.unparse(
            node.iter
        ).replace(" ", "")
    ]
    assert len(loops) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "aux_hidden_states.append"
        for node in ast.walk(loops[0])
    )


def test_dspark_layer_index_conversion(source_root: Path) -> None:
    path = (
        source_root
        / "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_eagle3_aux_layers_from_config"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    get_layers = namespace["get_eagle3_aux_layers_from_config"]

    hf_config = types.SimpleNamespace(target_layer_ids=[2, 23, 47, 71, 89])
    spec = types.SimpleNamespace(
        draft_model_config=types.SimpleNamespace(hf_config=hf_config)
    )
    assert get_layers(spec) == (3, 24, 48, 72, 90)

    draft_path = source_root / "vllm/models/kimi_k3/nvidia/dspark_mla.py"
    draft_tree = ast.parse(draft_path.read_text(encoding="utf-8"))
    draft_model = find_class(draft_tree, "K3DSparkModel")
    draft_init = find_method(draft_model, "__init__")
    assert (
        "self.config.target_hidden_size * self.config.num_target_layers"
        in ast.unparse(draft_init)
    )
    assert len(hf_config.target_layer_ids) * 7168 == 35840


def test_cuda_graph_and_draft_wiring(source_root: Path) -> None:
    graph_path = source_root / "vllm/v1/worker/gpu/cudagraph_utils.py"
    graph_tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    graph_cls = find_class(graph_tree, "ModelCudaGraphManager")
    run_fullgraph = find_method(graph_cls, "run_fullgraph")
    graph_source = ast.unparse(run_fullgraph)
    assert "self.use_aux_hidden_state_outputs" in graph_source
    assert "return(hidden_states,[x[:desc.num_tokens]" in graph_source.replace(
        " ", ""
    )

    spec_path = (
        source_root
        / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
    )
    spec_tree = ast.parse(spec_path.read_text(encoding="utf-8"))
    spec_cls = find_class(spec_tree, "DFlashSpeculator")
    propose = find_method(spec_cls, "propose")
    propose_source = ast.unparse(propose).replace(" ", "")
    assert "torch.cat(aux_hidden_states,dim=-1)" in propose_source
    assert "self.model.combine_hidden_states" in propose_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    test_runtime_protocol_and_default_method(args.source_root)
    test_kimi_hidden_state_contract(args.source_root)
    test_dspark_layer_index_conversion(args.source_root)
    test_cuda_graph_and_draft_wiring(args.source_root)
    print("PASS: KimiLinear EAGLE3 bridge source contract (4 tests)")


if __name__ == "__main__":
    main()

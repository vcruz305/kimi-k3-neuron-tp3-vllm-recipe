"""Hermetic CPU check for the graph-mode resolver in patch 0010.

This deliberately extracts only the pure resolver function from the vLLM
source so validation does not require importing CUDA, torch, or vLLM.
"""

from __future__ import annotations

import argparse
import ast
from enum import Enum
from pathlib import Path


class CUDAGraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        if isinstance(self.value, tuple):
            return CUDAGraphMode(self.value[0])
        return self


def load_resolver(source: Path):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_dflash_cudagraph_mode"
    ]
    if len(funcs) != 1:
        raise AssertionError(f"expected one resolver, found {len(funcs)}")
    module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    namespace = {"CUDAGraphMode": CUDAGraphMode}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_resolve_dflash_cudagraph_mode"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    resolve = load_resolver(args.source)

    cases = [
        (CUDAGraphMode.PIECEWISE, True, True, CUDAGraphMode.FULL_DECODE_ONLY),
        (CUDAGraphMode.PIECEWISE, True, False, CUDAGraphMode.NONE),
        (CUDAGraphMode.PIECEWISE, False, True, CUDAGraphMode.NONE),
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            True,
            False,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        (
            CUDAGraphMode.FULL_DECODE_ONLY,
            True,
            False,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ),
        (CUDAGraphMode.NONE, True, True, CUDAGraphMode.NONE),
    ]
    for target, supported, opt_in, expected in cases:
        actual = resolve(target, supported, opt_in)
        if actual is not expected:
            raise AssertionError(
                f"{target.name}, supported={supported}, opt_in={opt_in}: "
                f"expected {expected.name}, got {actual.name}"
            )
    print(f"PASS: {len(cases)} graph-mode cases")


if __name__ == "__main__":
    main()

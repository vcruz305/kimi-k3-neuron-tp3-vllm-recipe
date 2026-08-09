#!/usr/bin/env python3
"""Print and validate the installed K3 vLLM overlay without reading secrets.

The default JSON report is safe to paste into a public bug report: no
tokens, no absolute filesystem paths (which can contain a username), no
model files. Pass --show-install-paths to include the absolute vllm and
vllm_gguf_plugin install paths for local debugging.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import sys


PIN_ROOT = Path(
    os.environ.get(
        "K3_PIN_INSTALL_DIR",
        str(Path(sys.prefix) / "share/k3-neuron-vllm-pins"),
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument(
        "--show-install-paths",
        action="store_true",
        help=(
            "Include absolute install paths for vllm and vllm_gguf_plugin. "
            "These can contain a username; omit this flag for a report "
            "meant to be pasted into a public bug report."
        ),
    )
    args = parser.parse_args()

    import torch
    import vllm
    import vllm_gguf_plugin

    model_module = importlib.import_module("vllm.models.kimi_k3.nvidia.model")
    dspark_module = importlib.import_module("vllm.models.kimi_k3.nvidia.dspark_mla")
    speculator_module = importlib.import_module(
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
    )

    model_class = getattr(model_module, "KimiLinearForCausalLM")
    assert hasattr(model_class, "set_aux_hidden_state_layers")
    assert hasattr(dspark_module, "K3DSparkModel")
    assert hasattr(speculator_module, "_resolve_dflash_cudagraph_mode")

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not args.allow_no_gpu and visible_gpus != 3:
        raise SystemExit(f"expected exactly 3 visible GPUs, found {visible_gpus}")

    report = {
        "vllm_version": vllm.__version__,
        "vllm_importable": True,
        "gguf_plugin_importable": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_nccl_version": (
            ".".join(str(part) for part in torch.cuda.nccl.version())
            if torch.cuda.is_available()
            else None
        ),
        "visible_gpus": visible_gpus,
        "vllm_commit": (PIN_ROOT / "VLLM_COMMIT").read_text().strip(),
        "gguf_plugin_commit": (
            PIN_ROOT / "GGUF_PLUGIN_COMMIT"
        ).read_text().strip(),
        "kimi_linear_eagle3_bridge": True,
        "dspark_graph_resolver": True,
    }
    if args.show_install_paths:
        report["vllm_path"] = inspect.getfile(vllm)
        report["gguf_plugin_path"] = inspect.getfile(vllm_gguf_plugin)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

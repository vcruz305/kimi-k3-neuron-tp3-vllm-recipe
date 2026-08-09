#!/usr/bin/env python3
"""Compare one real Kimi-K3 IQ1_S expert with its dense dequantized oracle.

The script is intentionally single-GPU and reads only three tensors for one
expert.  It reproduces the exact TP=3 packed slices used by the GGUF plugin,
without constructing or restarting the model server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F

import vllm_gguf_plugin.ops as gguf_ops


TENSOR_SUFFIXES = {
    "w1": "ffn_gate_exps.weight",
    "w3": "ffn_up_exps.weight",
    "w2": "ffn_down_exps.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def find_tensors(
    model_dir: Path, layer: int
) -> dict[str, gguf.ReaderTensor]:
    wanted = {
        kind: f"blk.{layer}.{suffix}" for kind, suffix in TENSOR_SUFFIXES.items()
    }
    found: dict[str, gguf.ReaderTensor] = {}
    files = sorted(model_dir.glob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"no .gguf files in {model_dir}")
    for path in files:
        reader = gguf.GGUFReader(path)
        by_name = {tensor.name: tensor for tensor in reader.tensors}
        for kind, name in wanted.items():
            if name in by_name:
                found[kind] = by_name[name]
        if len(found) == len(wanted):
            break
    missing = [name for kind, name in wanted.items() if kind not in found]
    if missing:
        raise RuntimeError(f"missing GGUF tensors: {missing}")
    return found


def packed_expert(tensor: gguf.ReaderTensor, expert: int) -> np.ndarray:
    if expert < 0 or expert >= tensor.data.shape[0]:
        raise IndexError(
            f"expert {expert} outside packed tensor shape {tensor.data.shape}"
        )
    return np.ascontiguousarray(tensor.data[expert])


def dense_expert(tensor: gguf.ReaderTensor, expert: int) -> np.ndarray:
    return gguf.quants.dequantize(
        packed_expert(tensor, expert), tensor.tensor_type
    ).astype(np.float32, copy=False)


def situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate32 = gate.float()
    up32 = up.float()
    result = (
        4.0
        * torch.tanh(gate32 / 4.0)
        * torch.sigmoid(gate32)
        * (25.0 * torch.tanh(up32 / 25.0))
    )
    return result.to(gate.dtype)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual32 = actual.float()
    expected32 = expected.float()
    delta = actual32 - expected32
    cosine = F.cosine_similarity(
        actual32.reshape(1, -1), expected32.reshape(1, -1)
    ).item()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rms": delta.square().mean().sqrt().item(),
        "reference_rms": expected32.square().mean().sqrt().item(),
        "cosine": cosine,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.tp_size != 3:
        raise ValueError("this sealed diagnostic currently targets TP=3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the IQ1_S kernel oracle")

    tensors = find_tensors(args.model_dir, args.layer)
    qtypes = {int(tensor.tensor_type) for tensor in tensors.values()}
    iq1_s = int(gguf.GGMLQuantizationType.IQ1_S)
    if qtypes != {iq1_s}:
        raise RuntimeError(f"expected all IQ1_S tensors, got qtypes={qtypes}")

    packed = {
        kind: packed_expert(tensor, args.expert)
        for kind, tensor in tensors.items()
    }
    dense = {
        kind: dense_expert(tensor, args.expert)
        for kind, tensor in tensors.items()
    }
    if dense["w1"].shape != dense["w3"].shape:
        raise RuntimeError("gate/up logical shapes differ")
    hidden = dense["w1"].shape[1]
    intermediate = dense["w1"].shape[0]
    if dense["w2"].shape != (hidden, intermediate):
        raise RuntimeError(
            f"unexpected down shape {dense['w2'].shape}; "
            f"expected {(hidden, intermediate)}"
        )
    if intermediate % args.tp_size:
        raise RuntimeError("intermediate width is not divisible by TP size")

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn((1, hidden), generator=generator, device=device)
    x = (x * args.input_scale).to(torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32, device=device)

    local_intermediate = intermediate // args.tp_size
    if packed["w1"].shape[0] != intermediate:
        raise RuntimeError("gate packed rows do not match logical rows")
    if packed["w2"].shape[1] % args.tp_size:
        raise RuntimeError("down packed columns are not divisible by TP size")
    local_packed_down = packed["w2"].shape[1] // args.tp_size

    per_rank: list[dict[str, object]] = []
    kernel_outputs: list[torch.Tensor] = []
    dense_outputs: list[torch.Tensor] = []

    for rank in range(args.tp_size):
        logical = slice(
            rank * local_intermediate, (rank + 1) * local_intermediate
        )
        packed_down = slice(
            rank * local_packed_down, (rank + 1) * local_packed_down
        )

        w13_raw_np = np.concatenate(
            (packed["w1"][logical], packed["w3"][logical]), axis=0
        )
        w2_raw_np = packed["w2"][:, packed_down]
        w13_raw = torch.from_numpy(np.ascontiguousarray(w13_raw_np)).to(device)
        w2_raw = torch.from_numpy(np.ascontiguousarray(w2_raw_np)).to(device)
        w13_raw = w13_raw.unsqueeze(0)
        w2_raw = w2_raw.unsqueeze(0)

        w1_dense = torch.from_numpy(np.ascontiguousarray(dense["w1"][logical]))
        w3_dense = torch.from_numpy(np.ascontiguousarray(dense["w3"][logical]))
        w2_dense = torch.from_numpy(
            np.ascontiguousarray(dense["w2"][:, logical])
        )
        w1_dense = w1_dense.to(device=device, dtype=torch.bfloat16)
        w3_dense = w3_dense.to(device=device, dtype=torch.bfloat16)
        w2_dense = w2_dense.to(device=device, dtype=torch.bfloat16)

        kernel_w13 = gguf_ops.ggml_moe_a8_vec(
            x,
            w13_raw,
            topk_ids,
            1,
            iq1_s,
            2 * local_intermediate,
            1,
        ).reshape(1, 2 * local_intermediate)
        dense_gate = F.linear(x, w1_dense)
        dense_up = F.linear(x, w3_dense)
        dense_w13 = torch.cat((dense_gate, dense_up), dim=-1)

        kernel_hidden = situ(
            kernel_w13[:, :local_intermediate],
            kernel_w13[:, local_intermediate:],
        )
        dense_hidden = situ(dense_gate, dense_up)

        kernel_w2 = gguf_ops.ggml_moe_a8_vec(
            kernel_hidden,
            w2_raw,
            topk_ids,
            1,
            iq1_s,
            hidden,
            1,
        ).reshape(1, hidden)
        dense_w2_from_kernel_hidden = F.linear(kernel_hidden, w2_dense)
        dense_end_to_end = F.linear(dense_hidden, w2_dense)

        per_rank.append(
            {
                "rank": rank,
                "w13_kernel_vs_dequant_bf16": error_metrics(
                    kernel_w13, dense_w13
                ),
                "w2_kernel_vs_dequant_bf16_same_input": error_metrics(
                    kernel_w2, dense_w2_from_kernel_hidden
                ),
                "end_to_end_kernel_vs_dense": error_metrics(
                    kernel_w2, dense_end_to_end
                ),
            }
        )
        kernel_outputs.append(kernel_w2)
        dense_outputs.append(dense_end_to_end)

    kernel_tp = torch.stack(kernel_outputs).float().sum(dim=0)
    dense_tp = torch.stack(dense_outputs).float().sum(dim=0)
    report = {
        "contract": {
            "layer": args.layer,
            "expert": args.expert,
            "tp_size": args.tp_size,
            "hidden": hidden,
            "intermediate": intermediate,
            "dtype": "bfloat16",
            "qtype": "IQ1_S",
            "device": str(device),
            "seed": args.seed,
        },
        "packed_shapes": {kind: list(value.shape) for kind, value in packed.items()},
        "logical_shapes": {kind: list(value.shape) for kind, value in dense.items()},
        "per_rank": per_rank,
        "tp_sum_kernel_vs_dense": error_metrics(kernel_tp, dense_tp),
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json is not None:
        args.json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

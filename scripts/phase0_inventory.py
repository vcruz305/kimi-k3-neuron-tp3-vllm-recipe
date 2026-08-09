#!/usr/bin/env python3
"""Phase 0: attribute per-rank residency for the Neuron IQ1_S GGUF.

Answers: where do the 23.6 GiB/rank above the naive weight share go, and what
per-rank residency would TP3 / TP4 give on a 128 GB DGX Spark?

Pure metadata read of the GGUF headers - no GPU, no dequantization.
"""
from __future__ import annotations

import glob
import re
from collections import defaultdict

from gguf import GGUFReader

SHARDS = sorted(glob.glob("/work/models/Neuron-IQ1S/k3-neuron-iq1s-*-of-00009.gguf"))
GiB = 1024 ** 3


def bucket(name: str) -> str:
    n = name.lower()
    if "ffn_gate_exps" in n or "ffn_up_exps" in n or "ffn_down_exps" in n:
        return "routed_expert_ffn"
    if "ffn_gate_shexp" in n or "ffn_up_shexp" in n or "ffn_down_shexp" in n:
        return "shared_expert_ffn"
    if "ffn_gate_inp" in n or "exp_probs" in n or "router" in n:
        return "router"
    if "token_embd" in n:
        return "token_embedding"
    if "output.weight" in n or n.endswith("output.weight"):
        return "lm_head"
    if any(k in n for k in ("attn_q", "attn_k", "attn_v", "attn_o", "attn_kv",
                            "wq", "wk", "wv", "wo", "kv_a", "kv_b", "q_a", "q_b")):
        return "attention_latent"
    if "kda" in n or "conv" in n or "dt_" in n or "a_log" in n:
        return "kda"
    if "norm" in n:
        return "norm"
    return "other"


def main() -> None:
    by_bucket: dict[str, int] = defaultdict(int)
    by_bucket_n: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    bucket_types: dict[str, set] = defaultdict(set)
    total = 0

    for s in SHARDS:
        r = GGUFReader(s)
        for t in r.tensors:
            nbytes = int(t.n_bytes)
            b = bucket(t.name)
            by_bucket[b] += nbytes
            by_bucket_n[b] += 1
            tname = str(t.tensor_type).split(".")[-1]
            by_type[tname] += nbytes
            bucket_types[b].add(tname)
            total += nbytes
        del r

    print(f"shards: {len(SHARDS)}   total tensor bytes: {total:,}  ({total/GiB:.2f} GiB)\n")

    print(f"{'bucket':22} {'GiB':>9} {'%':>6} {'tensors':>9}  dtypes")
    for b in sorted(by_bucket, key=lambda x: -by_bucket[x]):
        print(f"{b:22} {by_bucket[b]/GiB:9.2f} {100*by_bucket[b]/total:5.1f}% "
              f"{by_bucket_n[b]:9,}  {','.join(sorted(bucket_types[b]))}")

    print(f"\n{'ggml type':22} {'GiB':>9} {'%':>6}")
    for t in sorted(by_type, key=lambda x: -by_type[x]):
        print(f"{t:22} {by_type[t]/GiB:9.2f} {100*by_type[t]/total:5.1f}%")

    # ---- what patch 0005 costs: dequantizing latent projections to bf16 -----
    lat = by_bucket["attention_latent"]
    # IQ1_S ~1.6 bit/weight effective; bf16 is 16 bit -> ~10x inflation
    print("\n--- patch 0005 (dequantize latent projections to bf16) ---")
    print(f"attention_latent on disk        : {lat/GiB:.2f} GiB")
    for ratio, label in ((10.0, "if stored IQ1_S (~1.6b) -> bf16"),
                         (4.0, "if stored ~4b -> bf16"),
                         (2.0, "if stored ~8b -> bf16")):
        print(f"  {label:32}: +{lat*(ratio-1)/GiB:7.2f} GiB total, "
              f"+{lat*(ratio-1)/3/GiB:6.2f} GiB/rank TP3, "
              f"+{lat*(ratio-1)/4/GiB:6.2f} GiB/rank TP4")

    # ---- residency model -----------------------------------------------------
    replicated = by_bucket["norm"] + by_bucket["router"]
    shardable = total - replicated
    print("\n--- per-rank residency model (weights only) ---")
    print(f"{'':16}{'TP3':>12}{'TP4':>12}")
    for tp in (3, 4):
        pass
    r3 = shardable / 3 + replicated
    r4 = shardable / 4 + replicated
    print(f"{'sharded/N':16}{shardable/3/GiB:11.2f}{shardable/4/GiB:12.2f}")
    print(f"{'+replicated':16}{replicated/GiB:11.2f}{replicated/GiB:12.2f}")
    print(f"{'= weights/rank':16}{r3/GiB:11.2f}{r4/GiB:12.2f}")
    print(f"\nmeasured H200 TP3 residency: 126.10 GiB/rank")
    print(f"unexplained above weight model (TP3): {126.10 - r3/GiB:.2f} GiB/rank")

    print("\n--- DGX Spark fit (128 GB = 119.2 GiB, ~110 GiB usable to the process) ---")
    for tp, r in ((3, r3), (4, r4)):
        w = r / GiB
        print(f"TP{tp}: weights {w:.2f} GiB/rank -> headroom vs 110 GiB = {110 - w:+.2f} GiB "
              f"{'FITS' if w < 100 else 'TIGHT/NO'}")

    # ---- k-scaling: what if expert FFN width changes -------------------------
    exp = by_bucket["routed_expert_ffn"] + by_bucket["shared_expert_ffn"]
    fixed = total - exp
    print(f"\n--- k-scaling (expert FFN = {exp/GiB:.2f} GiB, fixed = {fixed/GiB:.2f} GiB) ---")
    print(f"{'k':>6} {'retain':>8} {'total GiB':>10} {'TP3 GiB/rk':>11} {'TP4 GiB/rk':>11}")
    for k in (512, 768, 1024, 1280, 1536, 1792, 2048):
        scale = k / 1536.0
        tot = exp * scale + fixed
        sh = (tot - replicated)
        print(f"{k:>6} {100*k/3072:7.1f}% {tot/GiB:10.2f} "
              f"{(sh/3+replicated)/GiB:11.2f} {(sh/4+replicated)/GiB:11.2f}")


if __name__ == "__main__":
    main()

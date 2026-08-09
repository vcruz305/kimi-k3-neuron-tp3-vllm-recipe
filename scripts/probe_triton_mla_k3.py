#!/usr/bin/env python3
"""Does Kimi-K3's MLA geometry survive TRITON_MLA on this GPU? ~1 GiB, no model.

WHY
---
vLLM issue #26211 reports DeepSeek-series models failing on RTX PRO 6000 /
SM120. Kimi-K3 is DeepSeek-lineage MLA, and on Blackwell consumer/workstation
parts TRITON_MLA is the *only* MLA backend that accepts compute capability
12.0 -- FLASH_ATTN_MLA requires major==9, FLASHMLA major in [9,10],
FLASHINFER_MLA major==10. So on sm_120 there is no fallback: if the Triton
decode kernel cannot handle K3's shapes, K3 cannot serve there at all.

That question is normally gated behind loading a 300+ GiB model. It does not
have to be. The decode kernel takes tensors, not a model, so K3's *geometry*
can be pushed through it for about a gigabyte of VRAM.

WHAT THIS DOES AND DOES NOT ESTABLISH
-------------------------------------
Establishes, if it passes: the Triton MLA decode kernel compiles and runs at
K3's exact head/latent geometry on this GPU, producing finite, correctly shaped
output that varies with the KV contents.

Does NOT establish: end-to-end correctness of a real K3 serve, numerical
agreement with FLASH_ATTN_MLA, throughput, or anything about the prefill path.
A PASS here removes a specific blocker; it is not a green light.

If it FAILS, that is the more decisive result: it means #26211 does apply to
K3's shapes on this card, and no amount of re-quantizing or narrowing the model
will help, because the failure is in the attention kernel and not the weights.

USAGE
-----
Inside the vLLM container, on the target GPU:

    python probe_triton_mla_k3.py                 # defaults: TP4 (24 heads/rank)
    python probe_triton_mla_k3.py --tp 8          # 12 heads/rank
    python probe_triton_mla_k3.py --json          # machine-readable

Exit code 0 = PASS, 1 = FAIL, 2 = could not run the probe (import/env problem,
which is NOT evidence either way).
"""

import argparse
import json
import math
import sys
import traceback

# --- Kimi-K3 MLA geometry, read directly off the shipped GGUF's tensor shapes.
#     attn_kv_a_mqa.weight [7168, 576]  -> kv_lora_rank 512 + qk_rope 64
#     attn_k_b.weight      [128, 512, 96] -> 96 heads, qk_nope 128
#     attn_v_b.weight      [512, 128, 96] -> v_head_dim 128
#     attn_q_b.weight      [1536, 18432]  -> q_lora_rank 1536, 96*(128+64)=18432
K3 = dict(
    total_heads=96,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    q_lora_rank=1536,
    n_mla_layers=24,      # of 93 blocks; the other 69 are KDA
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size (default 4)")
    ap.add_argument("--batch", type=int, default=4, help="decode rows (default 4)")
    ap.add_argument("--seq-len", type=int, default=4096, help="KV length per row")
    ap.add_argument("--page-size", type=int, default=64, help="KV cache page size")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report: dict = {"probe": "triton_mla_k3_geometry", "tp": args.tp}

    def emit(status: str, code: int) -> int:
        report["status"] = status
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"\nVERDICT: {status}")
            if status == "PASS":
                print("  The Triton MLA decode kernel runs at K3's geometry on this GPU.")
                print("  #26211 does not block K3 at the decode-kernel level here.")
                print("  NOT established: end-to-end serve, numerics vs FLASH_ATTN_MLA,")
                print("  throughput, or the prefill path.")
            elif status == "FAIL":
                print("  K3's MLA geometry does NOT survive the Triton decode kernel here.")
                print("  On sm_120 there is no other MLA backend, so this is a hard blocker:")
                print("  re-quantizing or narrowing the model cannot fix an attention kernel.")
            else:
                print("  Probe could not run. This is NOT evidence either way.")
        return code

    # ---- argument preconditions (no GPU or torch needed) ------------------
    # Checked before touching the environment so an illegal TP reports as an
    # illegal TP, not as "torch is missing" -- and so this path stays testable
    # on a machine with no CUDA at all.
    if K3["total_heads"] % args.tp:
        legal = [t for t in range(1, 17) if K3["total_heads"] % t == 0]
        report["error"] = (f"{K3['total_heads']} attention heads do not divide by "
                           f"TP={args.tp}. Legal TP values <=16: {legal}")
        return emit("INCONCLUSIVE", 2)

    # ---- environment ------------------------------------------------------
    try:
        import torch
    except Exception as e:
        report["error"] = f"torch import failed: {e}"
        return emit("INCONCLUSIVE", 2)

    if not torch.cuda.is_available():
        report["error"] = "no CUDA device visible"
        return emit("INCONCLUSIVE", 2)

    cap = torch.cuda.get_device_capability()
    report["device"] = torch.cuda.get_device_name()
    report["capability"] = f"{cap[0]}.{cap[1]}"
    report["torch"] = torch.__version__
    report["arch_list"] = torch.cuda.get_arch_list()

    try:
        from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
    except Exception as e:
        report["error"] = f"could not import decode_attention_fwd: {e}"
        return emit("INCONCLUSIVE", 2)

    if not args.json:
        print(f"device     : {report['device']}  sm_{cap[0]}{cap[1]}")
        print(f"torch      : {report['torch']}")
        print(f"arch_list  : {report['arch_list']}")

    # ---- shapes -----------------------------------------------------------
    H = K3["total_heads"] // args.tp
    R = K3["kv_lora_rank"]                       # 512
    ROPE = K3["qk_rope_head_dim"]                # 64
    DIM = R + ROPE                               # 576, the cached latent width
    B, S, PAGE = args.batch, args.seq_len, args.page_size
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    dev = "cuda"

    # scale is over the full QK head dim (nope + rope), as the model config does
    scale = 1.0 / math.sqrt(K3["qk_nope_head_dim"] + ROPE)

    pages_per_row = (S + PAGE - 1) // PAGE
    n_blocks = B * pages_per_row

    report["shapes"] = dict(heads_per_rank=H, kv_lora_rank=R, latent_dim=DIM,
                            batch=B, seq_len=S, page_size=PAGE, n_blocks=n_blocks)
    if not args.json:
        print(f"\nK3 geometry at TP{args.tp}: {H} heads/rank, latent {DIM} "
              f"(= {R} kv_lora + {ROPE} rope), scale {scale:.6f}")
        print(f"synthetic KV: {n_blocks} pages x {PAGE} tokens, {args.dtype}")

    try:
        torch.manual_seed(0)
        # Paged latent KV cache: [n_blocks, PAGE, DIM]. forward_mqa unsqueezes a
        # head dim of 1, so mirror that here -- this is MQA over one latent head.
        kv = torch.randn(n_blocks, PAGE, DIM, dtype=dt, device=dev) * 0.1
        kv4 = kv.unsqueeze(2)                     # [n_blocks, PAGE, 1, DIM]
        kv_c = kv4[..., :R]                       # compressed part only

        q = torch.randn(B, H, DIM, dtype=dt, device=dev) * 0.1
        o = torch.zeros(B, H, R, dtype=dt, device=dev)
        lse = torch.zeros(B, H, dtype=dt, device=dev)

        block_table = torch.arange(n_blocks, dtype=torch.int32, device=dev).view(B, pages_per_row)
        seq_lens = torch.full((B,), S, dtype=torch.int32, device=dev)

        # Mirrors TritonMLAMetadataBuilder's sizing: power-of-two splits capped
        # by 2x SM count.
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        ideal = 1
        while ideal < max(1, S // 512):
            ideal *= 2
        num_kv_splits = min(ideal, sm * 2)
        attn_logits = torch.empty(B, H, num_kv_splits, R + 1, dtype=torch.float32, device=dev)
        report["num_kv_splits"] = num_kv_splits
        report["sm_count"] = sm
        report["workspace_MiB"] = round(attn_logits.numel() * 4 / 2**20, 2)
        if not args.json:
            print(f"num_kv_splits={num_kv_splits} (SMs={sm}), "
                  f"attn_logits workspace {report['workspace_MiB']} MiB fp32")

        ones = torch.ones(1, dtype=torch.float32, device=dev)
        decode_attention_fwd(q, kv4, kv_c, o, lse, block_table, seq_lens,
                             attn_logits, num_kv_splits, scale, PAGE,
                             k_scale=ones, v_scale=ones, is_mla=True)
        torch.cuda.synchronize()
    except Exception:
        report["error"] = traceback.format_exc(limit=6)
        if not args.json:
            print("\n--- kernel raised ---")
            print(report["error"])
        return emit("FAIL", 1)

    # ---- did it actually compute something? -------------------------------
    checks = {}
    checks["output_shape"] = (tuple(o.shape) == (B, H, R))
    checks["all_finite"] = bool(torch.isfinite(o).all().item())
    checks["not_all_zero"] = bool(o.abs().sum().item() > 0)
    checks["lse_finite"] = bool(torch.isfinite(lse).all().item())

    # A kernel that silently ignores the cache would produce identical output
    # for different KV. Perturb the cache and require the result to move.
    try:
        kv2 = (kv * 2.0).unsqueeze(2)
        o2 = torch.zeros_like(o)
        lse2 = torch.zeros_like(lse)
        attn_logits.zero_()
        decode_attention_fwd(q, kv2, kv2[..., :R], o2, lse2, block_table, seq_lens,
                             attn_logits, num_kv_splits, scale, PAGE,
                             k_scale=ones, v_scale=ones, is_mla=True)
        torch.cuda.synchronize()
        checks["responds_to_kv"] = bool((o2 - o).abs().max().item() > 1e-4)
    except Exception:
        checks["responds_to_kv"] = False
        report["perturbation_error"] = traceback.format_exc(limit=4)

    report["checks"] = checks
    if not args.json:
        print("\nchecks:")
        for k, v in checks.items():
            print(f"  {k:<18} {'ok' if v else 'FAILED'}")
        print(f"\noutput |mean|={o.float().abs().mean().item():.6f} "
              f"max={o.float().abs().max().item():.6f}")

    return emit("PASS" if all(checks.values()) else "FAIL",
                0 if all(checks.values()) else 1)


if __name__ == "__main__":
    sys.exit(main())

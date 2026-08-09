# Running this recipe on other GPU architectures

The measured results in this repository come from **one** architecture: sm_90
(H200). This document gives per-architecture instructions and states plainly
what is proven versus assumed on each.

**Always run the preflight first.** It probes the actual install rather than
trusting any table, including the ones below:

```bash
python scripts/preflight_arch.py          # human-readable, exits non-zero on NO-GO
python scripts/preflight_arch.py --json   # machine-readable, safe to paste in a bug report
```

## Two ways to get a runnable vLLM

**Path A - container overlay (minutes).** Every patch in `patches/` is pure
Python (the patches touch zero `.cu`/`.cpp` files). Note the GGUF *plugin*
itself is not pure Python - installing it compiles a CUDA extension - but its
build targets sm_120 natively and takes about 70 seconds, not hours. If a prebuilt vLLM image already carries kernels for your architecture,
you do not need to compile vLLM at all: overlay the patched files onto the
installed package and install the GGUF plugin.

Precondition: the image's vLLM commit must be close enough to the pinned base
that none of the patched files moved. Verify, do not assume:

```bash
git -C /path/to/vllm diff --name-only <image_commit> 75231eff2f3873e2bce7cc9558bb5227ea70b808 \
  -- vllm/config/speculative.py vllm/model_executor/models/utils.py \
     vllm/models/kimi_k3/nvidia/dspark_mla.py vllm/models/kimi_k3/nvidia/model.py \
     vllm/v1/worker/gpu/spec_decode/dflash/speculator.py \
     vllm/v1/worker/gpu/spec_decode/dspark/utils.py
```

Empty output means the overlay is byte-identical to a clean source build. On
sm_90 this path saved roughly an hour versus compiling.

**Path B - source build (hours).** Required when no prebuilt image carries your
architecture. `scripts/build_from_source.sh` defaults to
`TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX"`; override as needed.

The `+PTX` matters: it embeds `compute_120` PTX that JITs forward onto sm_121
(DGX Spark GB10), so one build can serve sm_90, sm_120 and sm_121. vLLM's own
`CMakeLists.txt` lists `12.0;12.1` together in `CUDA_SUPPORTED_ARCHS`,
corroborating the same-family assumption. Cost: a one-time JIT delay on first
load for the arch that lacks a native cubin.

## sm_90 - H200, H100 (VALIDATED)

The only architecture with measured results. vLLM selects `FLASH_ATTN_MLA`
automatically. Optional patch `patches/optional/0011` applies here **and only
here** - it is a Hopper FlashMLA wrapper. `scripts/apply_patches.sh` now refuses
it on other architectures unless `K3_SKIP_HOPPER_ARCH_CHECK=1`.

See `evidence/DSPARK-TP3-H200.md` for the numbers.

## sm_120 - RTX PRO 6000 Blackwell, RTX 5090 (UNTESTED, environment verified)

Measured on real sm_120 hardware - see
[`evidence/SM120-PREFLIGHT.md`](../evidence/SM120-PREFLIGHT.md):

- **The stock `vllm/vllm-openai:nightly` already contains native sm_120
  cubins.** A source build is not required. Use Path A.
- **`TRITON_MLA` is the only MLA backend that accepts capability 12.0.**
  `FLASH_ATTN_MLA` (`major == 9`), `FLASHMLA` (`major in [9,10]`) and
  `FLASHINFER_MLA` (`major == 10`) all refuse it, by their own
  `supports_compute_capability()`.

```bash
docker pull vllm/vllm-openai:nightly
python scripts/preflight_arch.py            # expect GO-WITH-CAVEATS, TRITON_MLA

# NOTE: on the nightly tested 2026-08-09, vLLM did NOT recognise this variable
# ("Unknown vLLM environment variable detected"). Verify the current knob name
# before relying on it. Generation worked regardless on that build.
export VLLM_FLASH_ATTN_VERSION=2            # FA3 has no Blackwell support
# then serve as usual, but pin the backend in the speculative config:
#   "attention_backend": "TRITON_MLA"
```

Two risks to keep in view:

1. **Performance is unmeasured.** Every throughput figure in this repository was
   produced with `FLASH_ATTN_MLA`. `TRITON_MLA` is the arch-agnostic fallback
   and its cost on this model is unknown. Do not quote sm_90 numbers for sm_120.
2. **Upstream [vLLM #26211](https://github.com/vllm-project/vllm/issues/26211)**
   reports DeepSeek-series models failing on RTX PRO 6000 / SM120. Kimi-K3 is
   DeepSeek-lineage MLA, so this plausibly applies.

## sm_100 - B200, GB200 (UNTESTED, but the better Blackwell path)

`sm_100` is also present in the nightly's arch list, and it is the one Blackwell
family that `FLASHINFER_MLA` accepts (`major == 10`). If you have a choice of
Blackwell hardware, this is the one likely to avoid the Triton fallback.

## sm_121 - DGX Spark GB10 (UNTESTED, two extra blockers)

1. **aarch64.** Spark is ARM. Stock PyTorch ships CUDA kernels only through
   sm_120, and aarch64 wheel availability is the practical constraint - see
   [vLLM #36821](https://github.com/vllm-project/vllm/issues/36821). Either rely
   on the `12.0+PTX` JIT path or rebuild from source for aarch64.
2. **Memory.** Per `docs/DGX-SPARK-PORT.md`, the 307.49 GiB model measures
   **84.06 GiB/rank at TP4**, so it needs **4 Sparks, not 3**. A larger
   IQ2-class build (~353.71 GiB) scales that to roughly **88 GiB/rank at TP4**
   and does **not** fit 3 Sparks at all.

Backend expectation is the same as sm_120: `TRITON_MLA` only.

## sm_80 - A100 (UNTESTED)

Also unvalidated. The gap is not Blackwell-specific. `FLASH_ATTN_MLA` requires
`major == 9`, so A100 falls back to `TRITON_MLA` as well. Run the preflight.

## Summary

| arch | example | build path | MLA backend | status |
|---|---|---|---|---|
| sm_90 | H200 | overlay or source | `FLASH_ATTN_MLA` | **VALIDATED** |
| sm_100 | B200 | overlay | `FLASHINFER_MLA` | untested |
| sm_120 | RTX PRO 6000, 5090 | **overlay (verified)** | `TRITON_MLA` only | untested, env verified |
| sm_121 | DGX Spark | source, aarch64 | `TRITON_MLA` only | untested, + memory |
| sm_80 | A100 | overlay | `TRITON_MLA` | untested |

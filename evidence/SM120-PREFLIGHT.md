# sm_120 preflight: measured on real Blackwell hardware

Sanitized capture from `scripts/preflight_arch.py` run inside
`vllm/vllm-openai:nightly` on a 2 x NVIDIA GeForce RTX 5090 host, 2026-08-09.
No pod identity, address, or credential included.

RTX 5090 and RTX PRO 6000 Blackwell are **both compute capability 12.0**, so
this capture answers the RTX PRO 6000 question directly at a fraction of the
rental cost.

## Headline result: the nightly container already carries sm_120 kernels

```text
torch      : 2.13.0+cu130
arch_list  : ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
capability : (12, 0)
device     : NVIDIA GeForce RTX 5090
vllm       : 0.26.1rc1.dev528+gf8d03e774
```

**A from-source vLLM build is NOT required for sm_120.** `sm_120` is present as
a native cubin in the stock nightly image. Because every patch in this
repository is pure Python, the container-overlay path in `README.md` applies:
copy the patched files over the installed package and install the GGUF plugin.
Setup is minutes, not the multi-hour compile the source path implies.

Note `sm_100` is also present, so B200-class datacenter Blackwell is covered by
the same image — and, per the backend table below, would additionally get
`FLASHINFER_MLA`, which sm_120 does not.

## Preflight output

```text
python: 3.12.3  platform: Linux/x86_64
torch: 2.13.0+cu130 (cuda build 13.0)
torch.cuda.get_arch_list(): ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
torch.cuda.is_available(): True

GPUs:
  [0] NVIDIA GeForce RTX 5090  sm_120  kernel_coverage=native
  [1] NVIDIA GeForce RTX 5090  sm_120  kernel_coverage=native

MLA attention backends in this install:
  FLASH_ATTN_MLA   importable, self-reports: REFUSES this capability
  FLASHMLA         importable, self-reports: REFUSES this capability
  FLASHINFER_MLA   importable, self-reports: REFUSES this capability
  TRITON_MLA       importable, self-reports: SUPPORTS this capability

VERDICT: GO-WITH-CAVEATS
Recommended attention_backend: TRITON_MLA
Recommended environment variables:
  export VLLM_FLASH_ATTN_VERSION=2
```

## What this establishes, and what it does not

**Established (measured):**

- sm_120 kernel coverage is native in the stock nightly; no source build needed.
- Of the four MLA backends, **only `TRITON_MLA` accepts capability 12.0.** The
  other three refuse it via their own `supports_compute_capability()`, so this
  is the backends' own report, not an assumption by this repository.
- The gates, read from vLLM at the pinned base commit: `FLASH_ATTN_MLA`
  requires `major == 9`; `FLASHMLA` requires `major in [9, 10]`;
  `FLASHINFER_MLA` requires `major == 10`; `TRITON_MLA` returns `True`
  unconditionally.

**Not established (still untested):**

- **Whether Kimi-K3 actually serves correctly on sm_120.** This preflight
  probes the environment, not the model. The 330 GB target does not fit on
  2 x 32 GiB, so no load was attempted. Upstream
  [vLLM #26211](https://github.com/vllm-project/vllm/issues/26211)
  ("vLLM does not support DeepSeek series on RTX PRO 6000/SM120") remains an
  open risk for DeepSeek-lineage MLA models, which Kimi-K3 is.
- **The performance cost of `TRITON_MLA` versus `FLASH_ATTN_MLA`.** Every speed
  number in `evidence/DSPARK-TP3-H200.md` was produced with `FLASH_ATTN_MLA`,
  which vLLM selected automatically on sm_90. Triton is the arch-agnostic
  fallback and its cost on the target model has never been measured. Do not
  assume sm_120 reproduces the sm_90 throughput figures.
- Nightly drift: this capture is vLLM `dev528+gf8d03e774`, while this
  repository pins base `75231eff`. Before overlaying, confirm none of the six
  patched files moved between the two commits (`git diff --name-only`), exactly
  as the README's fast-path precondition requires.

## Follow-up: Path A validated end-to-end on sm_120 (measured 2026-08-09)

A second session ran the full overlay path on the same 2 x RTX 5090 host.

**Overlay precondition: CLEAN.** Installed image commit `f8d03e77416bf90c...`
versus pinned base `75231eff2f3873e2...`:
`git diff --name-only <base> <image> -- <the six patched files>` returned
**empty**. Verified non-vacuously: all six files confirmed present at BOTH
commits via `git cat-file -e`. The overlay is byte-identical to a clean source
build at this image revision.

**GGUF plugin install: 69 seconds** end to end (clone 1s, 4 mandatory patches
applied with zero fuzz, `gguf==0.19.0` 1s, plugin build+install 67s).

**Correction - the plugin is NOT pure Python.** Installing it compiles a real
CUDA extension: `nvcc` builds `vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu` and
`c++` builds `torch_bindings.cpp`, linked into `_C_gguf.abi3.so`. Crucially its
gencode list **explicitly includes `-gencode=arch=compute_120,code=sm_120`**
(alongside 75/80/86/89/90/100), so the plugin targets sm_120 natively rather
than relying on PTX JIT. What IS pure Python is *this repository's patch set* -
the patches touch zero `.cu`/`.cpp` files - which is what makes the overlay
possible. Do not conflate the two.

**Real inference on sm_120.** `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
(`q4_k_m`, 491,400,032 bytes) served through the patched vLLM + plugin, TP1.
Server ready in 144s (torch.compile 10.84s, CUDA graph capture PIECEWISE 51/51
and FULL 35/35). Sample, unedited:

```text
"The capital of France is"  ->  " Paris. It was founded in 787 AD by the Romans, and it has been a"
"Q: What is 2+2?
A:"       ->  " 4"
```

Backend selected for that (non-MLA) model, from the startup log:
`Using FLASH_ATTN attention backend out of potential backends:
['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`.

**Two operational gotchas found the hard way:**

1. **`--tokenizer` is mandatory.** Without it, startup dies with
   `Unrecognized model in ...  Should have a model_type key in its config.json`
   - a GGUF carries no HF `config.json`. This repository's `serve_dspark.sh`
   already requires `--tokenizer`; the failure is a rediscovery of that
   requirement, not a new gap.
2. **`VLLM_FLASH_ATTN_VERSION` is NOT recognized by this nightly.** It logs
   `WARNING envs.py: Unknown vLLM environment variable detected:
   VLLM_FLASH_ATTN_VERSION`. Generation still worked (this model auto-selected
   FlashAttention 2 anyway), but **do not rely on that env var on this build** -
   it appears to have been renamed or removed. Verify the current knob before
   depending on it for the MLA path.

**TRITON_MLA probed live, deeper than the preflight goes.** On measured
capability (12, 0): `TRITON_MLA.supports_compute_capability()` returns True;
FLASH_ATTN_MLA / FLASHMLA / FLASHINFER_MLA all return False. Its MRO is a real
chain (`TritonMLABackend -> MLACommonBackend -> AttentionBackend -> ABC`), and
`get_impl_cls()` / `get_builder_cls()` resolve to concrete `TritonMLAImpl` /
`TritonMLAMetadataBuilder`, not stubs.

**Still untested (unchanged):** whether Kimi-K3 itself loads and serves on
sm_120 - never attempted, 330+ GiB does not fit in 64 GiB. Kimi-K3's MLA
dimensions are passed inside `TritonMLAImpl`'s `mla_args` bundle rather than as
flat kwargs; the class is architected to accept arbitrary MLA configs, but K3's
specific shapes were NOT constructed. And TRITON_MLA's numerical correctness and
performance cost on sm_120 remain unmeasured.

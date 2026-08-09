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

# Kimi-K3 Neuron TP3 vLLM recipe

Patch recipe for serving the 330.2 GB Kimi-K3 Neuron IQ1_S GGUF through
vLLM on three H200 GPUs, with breakable CUDA graphs and an experimental
bridge to the released Kimi-K3 DSpark speculative draft.

> **DSpark status: GPU-QUALIFIED (2026-08-09).** The combined GGUF target +
> DSpark path now constructs, serves, and clears the 1.15x promotion gate on
> 3 x H200: **42.464 token/s** single-stream at `num_speculative_tokens: 2`,
> **1.218x** a contemporaneous target-only baseline of 34.875 token/s.
> Full receipt: [`evidence/DSPARK-TP3-H200.md`](evidence/DSPARK-TP3-H200.md).
>
> **Set `num_speculative_tokens: 2`, not the draft config's default of 7.**
> On this IQ1_S target, N=7 is the *worst* point on the curve (36.056 token/s);
> speculative positions 5 and 6 contribute no measurable accepted tokens while
> still costing about 8 ms per step.
>
> **Exact token-ID parity with target-only is not achievable on this model**,
> and that is a property of quantization, not a bug — see the correctness
> section of the receipt before writing any equality-based test.

This is a **custom source overlay**, not an official vLLM release, a forked
wheel, or a complete vLLM fork. It pins upstream source commits and carries the
small patch set needed for this pruned GGUF. Build from those sources or apply
the patches to exact clean checkouts.

| | |
|---|---|
| **This recipe** | `vcruz305/kimi-k3-neuron-tp3-vllm-recipe` |
| **GGUF** | [vcruz305/Kimi-K3-Neuron-IQ1S-GGUF](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) |
| **Draft** | [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) |
| vLLM base | `75231eff2f3873e2bce7cc9558bb5227ea70b808` |
| GGUF plugin base | `d94067060884ea87766f12010c3a8b9c2d6715cc` |
| Hardware qualified so far | 3 x NVIDIA H200, TP3, single sequence |

The sanitized measurement contract is preserved in
[`evidence/TARGET-ONLY-H200.md`](evidence/TARGET-ONLY-H200.md).

## Measured target-only performance

The target-only measurements below used the same 330.2 GB / 307.49 GiB
Neuron GGUF, TP3, BF16 compute, one sequence, and `--disable-custom-all-reduce`.
The graph runs used capture size `[1]` only.

| Runtime contract | Sustained decode | Status |
|---|---:|---|
| vLLM eager | 6.632 token/s | GPU measured |
| vLLM PIECEWISE graph, 3 serial reps x 64 output tokens | 30.092 token/s | GPU measured |
| vLLM PIECEWISE graph, 3 serial reps x 256 output tokens | **34.339 token/s** | GPU measured |
| llama.cpp target-only | about 20 token/s | approximate, non-contemporaneous reference |
| **vLLM graph + DSpark, N=2** | **42.464 token/s** | **GPU measured 2026-08-09** |

## Measured DSpark performance

Contemporaneous target-only baseline on the same build: **34.875 token/s**
(reproduces the published `b85de5ba…` output hash exactly).

| `num_speculative_tokens` | Sustained decode | vs target-only |
|---:|---:|---:|
| 1 | 39.113 token/s | 1.122x |
| **2** | **42.464 token/s** | **1.218x** |
| 3 | 41.325 token/s | 1.185x |
| 5 | 40.028 token/s | 1.148x |
| 7 (draft-config default) | 36.056 token/s | 1.034x |

Aggregate throughput, DSpark N=2: **51.352 token/s** at batch 2, **76.857
token/s** at batch 4. Target-only reaches **88.546 token/s** at batch 8.

**Break-even rule.** Fitted cost is `step(N) ~= 37.2 + 5.95*N` ms against a
28.67 ms standalone decode, so speculative position *i* pays only when its
acceptance probability exceeds `5.95 / 28.67 = 0.2075`. Measured acceptance
crosses that threshold between positions 2 and 3, which is exactly where the
optimum was measured. Use this rule to retune N for any other quantization.

Levers that measured as **null** on this target — recorded so they are not
repeated: the optional Hopper FlashMLA draft patch (+1.4%), `probabilistic` +
`block` sampling (untestable at temperature 0), `--async-scheduling` (+0.02%),
target `cudagraph_mode: FULL` (refused by the KDA backend), and symmetric-memory
all-reduce (unavailable at world size 3).

The sealed France prompt returned token ID `17374` (` Paris`) in 3/3 runs.
A stricter five-prompt cross-engine gate matched 4/5; the fifth prompt was a
near log-probability tie, so this repository does not claim byte-for-byte
identity with llama.cpp on every prompt.

## What the patches add

- a tensor-name and transform adapter for text-only Kimi-K3 Neuron GGUF;
- TP3-safe vocabulary padding and correct distributed fused-loader rank use;
- BF16 dequantization for the latent routed projections used by native K3;
- FP32 preservation for routers and AttnRes projections;
- a CUDA grid-limit guard for top-16 MoE dispatch;
- exact zero padding of the 64-head / 14,336-wide DSpark draft to TP3;
- separate GGUF target and safetensors draft configuration/load paths;
- the KimiLinear EAGLE3 auxiliary-state bridge;
- a FULL M=7 draft graph alongside a PIECEWISE M=1/M=8 target graph.

Superseded experiments `0002`, `0003`, and their `0006` reversion are
intentionally absent. The optional Hopper FlashMLA patch is quarantined under
[`patches/optional`](patches/optional) and is not part of the first DSpark run.

## Prerequisites

- Linux with three H200-class GPUs visible in one host;
- a CUDA 13.0 development stack and Python 3.12 for source builds;
- at least 350 GB of local model storage;
- Hugging Face access to the manually gated GGUF repository;
- `git`, Python, and a compiler toolchain. Docker is recommended.

The source build is reproducible at the Git level, but it is large. The
target-only GPU receipt used vLLM
`0.26.1rc1.dev511+g700d39b55` with PyTorch `2.13.0+cu130` and NCCL `2.29.7`;
the complete toolkit and driver versions were not captured. The public patch
base is seven commits later and does not change the relevant K3/DSpark files.
Clean patch application and source contracts are validated at the public base,
while the integrated DSpark GPU run at that base remains pending.

## 1. Download pinned assets

Authenticate without putting a token in a command, shell history, or this
repository:

```bash
hf auth login
./scripts/download_assets.sh /models/k3-neuron /models/k3-tokenizer /models/k3-dspark
```

The tokenizer executes pinned Moonshot tokenizer code. It is downloaded from
the immutable revision in `pins.env`, checked against the hashes in
`TOKENIZER-SHA256SUMS`, and then loaded from the local directory with
`--trust-remote-code`. Review those four small files before use if your threat
model requires it. A static audit of the pinned files found no subprocess,
shell, network, dynamic-evaluation, deletion, or arbitrary-write path; the only
write helper is the normal vocabulary-save copy operation.

## 2. Build the pinned overlay

### Docker

The Dockerfile performs a full source build; it does not depend on a moving
`nightly` image:

```bash
docker build --build-arg MAX_JOBS=16 -t k3-neuron-vllm:tp3 .
```

Run target-only with host networking so the server's in-container loopback bind
remains loopback-only. Mount all downloaded artifacts read-only:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  -v /models/k3-neuron:/models/k3-neuron:ro \
  -v /models/k3-tokenizer:/models/k3-tokenizer:ro \
  -e TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf \
  -e TARGET_TOKENIZER=/models/k3-tokenizer \
  k3-neuron-vllm:tp3 \
  /opt/recipe/scripts/serve_target.sh
```

If you deliberately bind to a non-loopback interface, the launchers fail
closed unless `VLLM_API_KEY` is set or the explicit insecure-network
acknowledgement is supplied. Be aware that a CLI API key can be visible to
other users on the same host through process inspection.

### Existing CUDA environment

Run this inside an isolated CUDA build environment:

```bash
./scripts/prepare_sources.sh /opt/k3-sources
MAX_JOBS=16 ./scripts/build_from_source.sh /opt/k3-sources
./scripts/assert_runtime.py
```

The plugin install uses `--no-deps` so it cannot silently replace the runtime's
PyTorch/NCCL packages. See [`APPLY.md`](APPLY.md) for the exact order and
manual path.

## 3. Run target-only first

The server binds only to loopback by default. Do not expose an unauthenticated
vLLM endpoint to a public network.

```bash
export TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf
export TARGET_TOKENIZER=/models/k3-tokenizer
./scripts/serve_target.sh
```

In another shell:

```bash
./scripts/verify_server.py --model k3-neuron
./scripts/benchmark_server.py \
  --model k3-neuron \
  --chat-template /models/k3-neuron/k3_chat_template.jinja \
  --tokens 256 \
  --repetitions 3
```

The benchmark reproduces the exact published photosynthesis contract: shipped
template, maximum thinking effort, 105 rendered prompt tokens, temperature
zero, and fixed 64- or 256-token output. It reports HTTP-observed rates; it is
not a substitute for an engine profiler.

## 4. Qualify DSpark

Read [`docs/INTEGRATED-RUNBOOK.md`](docs/INTEGRATED-RUNBOOK.md) before spending
GPU time. The first run uses `TRITON_MLA`, greedy drafting, and standard
rejection so target token preservation can be checked before optimizing
sampling.

```bash
export TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf
export TARGET_TOKENIZER=/models/k3-tokenizer
export DRAFT_MODEL=/models/k3-dspark
export K3_ACK_DSPARK_UNVERIFIED=1
./scripts/serve_dspark.sh
```

Promotion requires exact target-token preservation on the fixed prompt suite,
at least 1.15x the contemporaneous target-only median, a bootstrap lower bound
above 1.00x, at least 1 GiB HBM free per rank after graph capture, and no ECC,
OOM, graph fallback, or draft-loader errors.

## Repository map

| Path | Purpose |
|---|---|
| [`APPLY.md`](APPLY.md) | exact clone, patch, install, and assertion sequence |
| [`config`](config) | text-only pruned Kimi-K3 configuration |
| [`patches/gguf-plugin`](patches/gguf-plugin) | mandatory GGUF plugin patches |
| [`patches/vllm`](patches/vllm) | mandatory vLLM, graph, and DSpark patches |
| [`patches/optional`](patches/optional) | unqualified Hopper FlashMLA experiment |
| [`scripts`](scripts) | build, launch, integrity, parity, and benchmark tools |
| [`tests`](tests) | hermetic source-contract checks |
| [`docs/INTEGRATED-RUNBOOK.md`](docs/INTEGRATED-RUNBOOK.md) | paid-run gates and rollback rules |

## Non-claims

- This is not upstream vLLM support or an official vLLM image.
- DSpark speed, acceptance, and graph replay are not yet GPU-proven here.
- The 34.339 token/s result used target capture `[1]`, not the unmeasured
  integrated target capture `[1,8]`.
- Native K3 MXFP4/GB300 throughput does not transfer directly to IQ1_S/TP3
  H200.
- This recipe does not redistribute weights, draft weights, or tokenizer
  artifacts.

## License and attribution

Original recipe scripts, documentation, and patch contributions are provided
under Apache-2.0. `config/config.json` is model-derived metadata and remains
subject to the applicable Kimi/model terms. Upstream projects and model
artifacts retain their own terms. See [`NOTICE`](NOTICE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

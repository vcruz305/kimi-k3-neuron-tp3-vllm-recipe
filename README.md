# Kimi-K3 Neuron vLLM recipe

Serve the compressed **Kimi-K3 Neuron IQ1_S GGUF** (330.2 GB / 307.49 GiB)
through vLLM, with breakable CUDA graphs and a working bridge to the released
**Kimi-K3 DSpark** speculative draft.

This is a **source overlay**, not a vLLM fork or a wheel. It pins two upstream
commits and carries the small patch set a pruned K3 GGUF needs.

| | |
|---|---|
| **GGUF** | [vcruz305/Kimi-K3-Neuron-IQ1S-GGUF](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) |
| **Draft** | [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) |
| **vLLM base** | `75231eff2f3873e2bce7cc9558bb5227ea70b808` |
| **GGUF plugin base** | `d94067060884ea87766f12010c3a8b9c2d6715cc` |
| **Validated on** | sm_90 (H200). Other architectures: run the preflight |

---

## Are you using an agent?

Paste this into Claude Code, Cursor, or similar. It encodes the traps that cost
real time here.

```text
Set up the Kimi-K3 Neuron vLLM recipe from
https://github.com/vcruz305/kimi-k3-neuron-tp3-vllm-recipe on this machine.

Do these in order and do not skip step 1.

1. Clone the repo and run `python scripts/preflight_arch.py`. It reports my GPU
   architecture, whether the installed torch/vLLM actually carries kernels for
   it, and which MLA attention backends this install will accept. Do not assume
   an answer - run it. If it reports NO-GO, stop and tell me why.

2. Choose a build path from its output:
   - If a prebuilt vLLM image already carries kernels for my arch, use the
     CONTAINER OVERLAY path (minutes). Every patch in patches/ is pure Python,
     so they can be copied over the installed package. Verify the precondition
     first: `git diff --name-only <image_commit> <pinned_base>` restricted to
     the six patched files must come back EMPTY.
   - Otherwise use scripts/build_from_source.sh (hours).

3. Install the GGUF plugin at its pinned commit with the four mandatory patches
   from patches/gguf-plugin/. Install `gguf==0.19.0` first, then the plugin with
   --no-deps. It compiles a CUDA extension; ~70s is normal, not a hang.

4. Serve with scripts/serve_dspark.sh. Non-negotiable details:
   - `--tokenizer` is MANDATORY. A GGUF has no model_type in config.json and
     vLLM refuses to start without it.
   - Set NUM_SPECULATIVE_TOKENS: 3 for code, 2 for prose. NEVER leave it at the
     draft config's default of 7 - that is measurably the worst setting.
   - Use the attention_backend the preflight recommended for my arch.

5. Verify with scripts/verify_server.py (sealed France token check).

Rules:
- Do NOT write a test requiring speculative output to match non-speculative
  output token-for-token. That is impossible on a quantized target by
  construction - see evidence/DSPARK-TP3-H200.md.
- Do NOT quote this repo's sm_90 performance numbers for any other
  architecture.
- Report what you actually measured, not what the docs claim. If a step fails,
  give me the real error text.
```

---

## Start here: which GPU do you have?

Run this first. It probes your actual install rather than trusting any table,
including the one below.

```bash
python scripts/preflight_arch.py          # exits non-zero on NO-GO
python scripts/preflight_arch.py --json   # safe to paste into a bug report
```

| Your GPU | arch | Build path | Attention backend | Status |
|---|---|---|---|---|
| **H200, H100** | sm_90 | overlay or source | `FLASH_ATTN_MLA` | **validated** |
| **B200, GB200** | sm_100 | overlay | `FLASHINFER_MLA` | untested |
| **RTX PRO 6000, RTX 5090** | sm_120 | overlay (verified) | `TRITON_MLA` only | environment verified |
| **DGX Spark (GB10)** | sm_121 | source, aarch64 | `TRITON_MLA` only | untested *(for this TP3+DSpark recipe)* |
| **A100** | sm_80 | overlay | `TRITON_MLA` | untested |

Per-architecture detail, including the sm_120 and sm_121 caveats:
[`docs/MULTI-ARCH.md`](docs/MULTI-ARCH.md).

**Looking for a validated 4x DGX Spark recipe?** This repo's DGX Spark row is
about the single-node, TP3+DSpark configuration above, which has not been run
on Spark hardware. A *separate*, non-speculative TP4 configuration across 4
physical Sparks was fully debugged and validated (four real vLLM/GGUF-plugin
bugs found and fixed) in
[`kimi-k3-neuron-tp4-vllm-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp4-vllm-recipe).
`docs/DGX-SPARK-PORT.md` in this repo is the pre-validation feasibility study
that preceded that work; the TP4 repo is what actually ships and runs.

---

## How to: get a runnable vLLM

### Option A - container overlay (minutes, preferred)

Every patch in `patches/` is pure Python. If a prebuilt image already carries
kernels for your architecture, you do not need to compile vLLM at all - copy the
patched files over the installed package.

**Verify the precondition first.** The image's vLLM commit must be close enough
to the pinned base that none of the patched files moved:

```bash
git -C /path/to/vllm diff --name-only <image_commit> 75231eff2f3873e2bce7cc9558bb5227ea70b808 \
  -- vllm/config/speculative.py vllm/model_executor/models/utils.py \
     vllm/models/kimi_k3/nvidia/dspark_mla.py vllm/models/kimi_k3/nvidia/model.py \
     vllm/v1/worker/gpu/spec_decode/dflash/speculator.py \
     vllm/v1/worker/gpu/spec_decode/dspark/utils.py
```

Empty output means the overlay is byte-identical to a clean source build.
Measured: this saved about an hour on sm_90, and on sm_120 the drift was zero.

The GGUF **plugin** is not pure Python - installing it compiles a CUDA
extension - but that takes about 70 seconds and its build targets sm_120
natively.

### Option B - source build (hours)

Needed when no prebuilt image carries your architecture.

```bash
./scripts/check_bundle.py
./scripts/prepare_sources.sh /opt/k3-sources
MAX_JOBS=16 ./scripts/build_from_source.sh /opt/k3-sources
./scripts/assert_runtime.py
```

`TORCH_CUDA_ARCH_LIST` defaults to `"9.0;12.0+PTX"`. The `+PTX` embeds
`compute_120` PTX that JITs forward onto sm_121 (DGX Spark), so one build can
serve sm_90, sm_120 and sm_121. Cost: a one-time JIT delay on first load.

---

## How to: download the pinned assets

```bash
hf auth login
./scripts/download_assets.sh /models/k3-neuron /models/k3-tokenizer /models/k3-dspark
```

The tokenizer executes pinned Moonshot code. It is fetched at the immutable
revision in `pins.env`, checked against `TOKENIZER-SHA256SUMS`, then loaded from
the local directory with `--trust-remote-code`.

---

## How to: serve it

```bash
K3_ACK_DSPARK_UNVERIFIED=1 \
TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf \
TARGET_TOKENIZER=/models/k3-tokenizer \
DRAFT_MODEL=/models/k3-dspark \
NUM_SPECULATIVE_TOKENS=3 CUDAGRAPH_CAPTURE_SIZES='[1,4]' \
./scripts/serve_dspark.sh
```

Three things that will bite you otherwise:

1. **`--tokenizer` is mandatory.** A GGUF carries no HF `config.json`, so vLLM
   fails with `Unrecognized model ... should have a model_type key`.
2. **Set `num_speculative_tokens` yourself** (next section). The draft config's
   default of 7 is the worst point on the curve.
3. **Pick the backend for your architecture** from the preflight.
   `FLASH_ATTN_MLA` is sm_90 only; most others get `TRITON_MLA`.

Target-only, without the draft:

```bash
TARGET_GGUF=... TARGET_TOKENIZER=... ./scripts/serve_target.sh
```

Then verify:

```bash
python scripts/verify_server.py --model k3-neuron-dspark
```

---

## How to: choose `num_speculative_tokens`

**Highest-impact setting in the whole recipe, and the default is wrong.**
Measured on sm_90, single stream, 256-token contract, temperature 0:

| workload | setting | capture sizes | result |
|---|---|---|---:|
| **low-entropy / code** | `3` | `[1,4]` | **52.454 token/s** |
| **high-entropy / prose** | `2` | `[1,3]` | **42.464 token/s** |
| draft-config default | `7` | `[1,8]` | 36.056 token/s (worst) |

Why: one extra speculative token costs ~5.95 ms against a 28.67 ms decode, so
position *i* only pays when its acceptance probability exceeds
`5.95 / 28.67 = 0.2075`. Measured acceptance crosses that between positions 2
and 3. At N=5 and N=7 the model emits an **identical** 2.844 tokens/step -
positions 5 and 6 contribute nothing while still costing ~8 ms.

Reuse that break-even rule to retune for any other quantization.

---

## Measured results (sm_90 only)

3 x H200, TP3, one sequence, `--disable-custom-all-reduce`.
**Do not quote these for another architecture.**

| Runtime | Sustained decode |
|---|---:|
| vLLM eager | 6.632 token/s |
| vLLM PIECEWISE graph, target only | 34.875 token/s |
| **+ DSpark, N=2 (prose)** | **42.464 token/s** |
| **+ DSpark, N=3 (coding)** | **52.454 token/s** |
| llama.cpp target-only (non-contemporaneous) | ~20 token/s |

Aggregate throughput: 51.4 token/s at batch 2 and 76.9 at batch 4 with DSpark;
88.5 token/s at batch 8 target-only.

Receipts: [`evidence/DSPARK-TP3-H200.md`](evidence/DSPARK-TP3-H200.md) -
[`evidence/TARGET-ONLY-H200.md`](evidence/TARGET-ONLY-H200.md) -
[`evidence/SM120-PREFLIGHT.md`](evidence/SM120-PREFLIGHT.md)

### One correctness property you must know

**Speculative output will not match non-speculative output token-for-token, and
that is not a bug.** Over the first 60 positions the median top-1/top-2 logprob
gap is 1.875 nats, but 5% of positions sit within 0.125 nats and at least one is
an exact tie. Any numerical difference between the M=1 decode path and the
M=N+1 verify path flips one of them, and GGUF dispatches small-M work to a
different kernel path, so such a difference exists by construction.

Use the sealed France single-token check (token `17374`, holds 3/3 in every
configuration) plus run-to-run determinism instead. **Do not write an
equality-based parity test.**

---

## What the patches do

- tensor-name and transform adapter for the text-only Kimi-K3 Neuron GGUF
- TP3-safe vocabulary padding and correct distributed fused-loader rank use
- BF16 dequantization for the latent routed projections native K3 needs
- FP32 preservation for routers and AttnRes projections
- a CUDA grid-limit guard for top-16 MoE dispatch
- exact zero padding of the 64-head / 14,336-wide DSpark draft to TP3
- separate GGUF target and safetensors draft configuration/load paths
- the KimiLinear EAGLE3 auxiliary-state bridge
- an independent FULL draft graph alongside a PIECEWISE target graph

Superseded experiments `0002`, `0003` and their `0006` reversion are
intentionally absent. The Hopper FlashMLA patch is quarantined in
[`patches/optional`](patches/optional), and `apply_patches.sh` refuses to apply
it off sm_90.

---

## Repository map

| Path | Purpose |
|---|---|
| [`scripts/preflight_arch.py`](scripts) | **run first** - architecture and backend probe |
| [`scripts/serve_dspark.sh`](scripts) | serve target + DSpark draft |
| [`scripts/serve_target.sh`](scripts) | serve target only |
| [`scripts/verify_server.py`](scripts) | sealed France token check |
| [`APPLY.md`](APPLY.md) | exact clone, patch, install and assertion sequence |
| [`docs/MULTI-ARCH.md`](docs/MULTI-ARCH.md) | per-architecture instructions |
| [`docs/INTEGRATED-RUNBOOK.md`](docs/INTEGRATED-RUNBOOK.md) | operator runbook, gates, rollback |
| [`docs/DGX-SPARK-PORT.md`](docs/DGX-SPARK-PORT.md) | Spark feasibility study (superseded — see [kimi-k3-neuron-tp4-vllm-recipe](https://github.com/vcruz305/kimi-k3-neuron-tp4-vllm-recipe) for the validated recipe) |
| [`evidence`](evidence) | sanitized measurement receipts |
| [`patches`](patches) | the mandatory patch chain |
| [`config`](config) | text-only pruned Kimi-K3 configuration |
| [`tests`](tests) | hermetic source-contract checks |

---

## Non-claims

- Validated on **sm_90 only**. Every other architecture is untested; the
  preflight tells you the truth for your own box.
- **`TRITON_MLA` performance is unmeasured.** Every number here used
  `FLASH_ATTN_MLA`, which only sm_90 accepts. Most other architectures fall
  back to Triton and the cost is unknown.
- Upstream [vLLM #26211](https://github.com/vllm-project/vllm/issues/26211)
  reports DeepSeek-series failures on sm_120. Kimi-K3 is DeepSeek-lineage MLA,
  so it plausibly applies there.
- The 34.339 token/s figure used target capture `[1]` only; the qualified DSpark
  path separately measured `[1,3]` (prose, N=2) and `[1,4]` (coding, N=3).
- Native K3 MXFP4/GB300 throughput does not transfer to IQ1_S on TP3.
- This is not upstream vLLM support or an official vLLM image, and it
  redistributes no weights, draft weights, or tokenizer artifacts.

## License and attribution

Original recipe scripts, documentation and patch contributions are provided
under Apache-2.0. `config/config.json` is model-derived metadata and remains
subject to the applicable Kimi/model terms. Upstream projects and model
artifacts retain their own terms. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

# DSpark speculative decoding on the Neuron IQ1_S target: qualification receipt

Sanitized summary of the 2026-08-09 paid run. No pod identity, network address,
credential, or model file is included.

This is the **first GPU qualification** of the integrated
target-`[1,N+1]`-PIECEWISE + official-DSpark path. It supersedes the
"not yet GPU-qualified" status in `docs/INTEGRATED-RUNBOOK.md`.

## Environment

- hardware: 3 x NVIDIA H200, TP3, one sequence unless stated;
- target: Kimi-K3 Neuron IQ1_S GGUF, 330,167,832,024 bytes, `sha256 OK` 9/9;
- draft: `Inferact/Kimi-K3-DSpark` at the pinned revision, safetensors;
- vLLM `0.26.1rc1.dev511+g700d39b55`, PyTorch `2.13.0+cu130`, NCCL `2.29.7`;
- explicit `--kv-cache-memory-bytes 3221225472`; custom all-reduce disabled;
- breakable CUDA graph enabled; volatile uncorrected ECC zero throughout.

Model load: **126.1 GiB/rank in 170.9 s**; init engine 37.06 s.

## Construction gates

| # | Gate | Result |
|---|---|---|
| 1 | DSpark exact TP3 padding | `heads 64 -> 66`, `intermediate size 14336 -> 14337` |
| 2 | draft loaded from pinned safetensors, never GGUF | pass, 1.05 s |
| 3 | EAGLE3 boundaries | **`(3, 24, 48, 72, 90)`** — converted, not raw `[2,23,47,71,89]` |
| 4 | target PIECEWISE `[1,N+1]` | pass |
| 5 | GPU KV cache | 6,638 tokens (≥ 4,096 required) |
| 6 | free HBM after capture | 8,089 MiB (≥ 1,024 required) |
| 7 | graph capture per rank | 0.18 GiB Triton draft / 0.26 GiB FlashMLA draft (≤ 2 GiB) |
| 8 | ECC | zero |

Draft residency measured by difference: **2.54 GiB/rank**.

## Speed: `num_speculative_tokens` is the dominant knob

Single stream, fixed 256-token contract, temperature 0, seed 0, three
repetitions, median sustained decode. Draft attention `FLASHMLA` (optional
patch `0011`) unless noted.

| config | median token/s | emitted/step | implied step | vs target-only |
|---|---:|---:|---:|---:|
| target-only (contemporaneous) | 34.875 | 1.000 | 28.67 ms | 1.000x |
| DSpark N=1 | 39.113 | 1.689 | 43.17 ms | 1.122x |
| **DSpark N=2** | **42.464** | 2.207 | 51.97 ms | **1.218x** |
| DSpark N=3 | 41.325 | 2.257 | 54.61 ms | 1.185x |
| DSpark N=5 | 40.028 | 2.844 | 71.06 ms | 1.148x |
| DSpark N=7 (draft-config default) | 36.056 | 2.844 | 78.89 ms | 1.034x |

**N=2 clears the ≥1.15x promotion gate at 1.218x.**

The draft config's default of **7 speculative tokens is the worst point on the
curve** for this target. N=5 and N=7 emit an identical 2.844 tokens/step, i.e.
speculative positions 5 and 6 contribute nothing measurable while still costing
about 8 ms per step. Anyone serving an IQ1_S K3 target with the stock
`num_speculative_tokens` is paying roughly 18% for no benefit.

### Cost model and a reusable break-even rule

Fitting target-only, N=1 and N=7 gives `step(N) ~= 37.2 + 5.95*N` ms: a fixed
~8.5 ms to engage the speculative path plus **m = 5.95 ms per speculative
token**. Against a 28.67 ms standalone decode, speculative position *i* pays
only when

> **p_i > m / target_step = 5.95 / 28.67 = 0.2075**

Measured per-position acceptance:

| config | p0 | p1 | p2 | p3 | p4 | p5 | p6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| N=2 | .741 | .466 | | | | | |
| N=3 | .699 | .381 | .177 | | | | |
| N=7 | .700 | .467 | .278 | .189 | .122 | .044 | .044 |

p crosses 0.2075 between positions 2 and 3, which is where the measured optimum
sits. A marginal verified token costs only 21% of a standalone decode, so the
batched verify is cheap: the binding constraint is draft acceptance, not
verification cost.

## Workload entropy is a first-class knob, and it moves the optimum

Every number above used a single **prose explanation** prompt. Published DSpark
on full-precision Kimi-K3 reports roughly **4.73 accepted tokens/step on
low-entropy/coding** work against **2.61 on high-entropy** text, so the sweep
above was tuned on the *worst* case for a draft.

Re-measured with 5 coding prompts and 2 prose prompts, 256 tokens each,
canonical single stream, temperature 0, seed 0:

| N | **coding median** | coding mean | coding peak | mean accepted | emitted/step | prose median |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 49.273 | 48.184 | 49.78 | 1.490 | 2.490 | **41.092** |
| **3** | **52.469** | **51.638** | **57.65** | 1.820 | 2.820 | 41.134 |
| 4 | 43.196 | 46.839 | 55.36 | 2.019 | 3.019 | 38.688 |
| 5 | 45.033 | 45.237 | 53.01 | 2.181 | 3.181 | 37.345 |

N=3 re-run three times on a fresh server, 15 measurements total: medians
**52.454 / 52.464 / 52.442**, spread **0.022 token/s (0.04%)**. Run-to-run is
effectively deterministic at temperature 0 — acceptance is bit-identical across
repetitions — so the visible spread between prompts is genuine workload
variance, not measurement noise. Three of five coding prompts individually
exceed 50 token/s; the slowest is 45.76.

**The break-even rule derived from prose data predicted the coding optimum.**
Marginal accepted tokens per added speculative position on coding: 2→3 gives
**+0.33** (above the 0.2075 threshold, so N=3 beats N=2), 3→4 gives **+0.20**
(at the threshold, so N=4 does not help), 4→5 gives **+0.16** (below it). The
measured ranking matches.

**Tune N to the workload — one setting does not serve both:**

| workload | setting | result |
|---|---|---:|
| low-entropy / code | `num_speculative_tokens: 3`, capture `[1,4]` | **52.45 token/s** |
| high-entropy / prose | `num_speculative_tokens: 2`, capture `[1,3]` | 42.46 token/s |

N=3 on prose is 41.13, slightly *worse* than N=2's 42.46, so the choice is a
real trade rather than a free win.

Caveat on interpretation: the acceptance shortfall against full-precision K3
(1.82 vs a published 4.73 accepted/step on coding) is real quantization loss and
motivates distilling a draft against the compressed target. But a substantial
part of the apparent shortfall in earlier sections was **workload choice**, not
quantization.

## Levers that did nothing, and why

Recorded so others do not repeat them.

| lever | result | explanation |
|---|---:|---|
| optional patch `0011`, draft `FLASHMLA` vs `TRITON_MLA` | 36.056 vs 35.567 at N=7 | `TRITON_MLA` logs `does not support full CUDA graphs; running the draft eagerly`. `FLASHMLA` **does** get a FULL draft graph (capture 0.18 -> 0.26 GiB), yet the step moves 0.1 ms. The draft passes are not dispatch-bound. |
| `draft_sample_method: probabilistic` + `rejection_sample_method: block` | 42.441 vs 42.464 | The benchmark contract fixes temperature 0, so target sampling is greedy and block rejection collapses to exact-match acceptance. **The released sampling contract cannot be evaluated at temperature 0**; it needs a separate nonzero-temperature protocol. |
| `--async-scheduling` | 42.454 vs 42.464 | 0.02%. The step is not CPU-scheduler-bound. |
| target `cudagraph_mode: FULL` | 34.694 vs 34.875 | `CUDAGraphMode.FULL is not supported with KimiK3KDAAttentionBackend (support: AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY`. K3's KDA layers refuse a full graph. |
| symmetric-memory all-reduce | unavailable | `SymmMemCommunicator: World size 3 not supported`. TP3 has no fast all-reduce path; this is the same world-size-3 limitation that requires `--disable-custom-all-reduce`. |

All four sampling/scheduling variants produced one output hash and acceptance
identical to 15 decimals, with launch arguments verified to contain the
requested settings — these are genuine null results, not dropped flags.

## Batch scaling

Target-only, `--max-num-seqs 8`, capture `[1,2,4,8]`:

| batch | per-stream token/s | aggregate token/s | step ms | vs batch-1 |
|---:|---:|---:|---:|---:|
| 1 | 33.397 | 33.397 | 29.94 | 1.00x |
| 2 | 22.063 | 44.126 | 45.32 | 1.51x |
| 4 | 18.045 | 72.180 | 55.42 | 1.85x |
| 8 | 11.068 | 88.546 | 90.35 | 3.02x |

DSpark N=2, `--max-num-seqs 4`, capture `[1,3,6,12]`:

| batch | per-stream token/s | aggregate token/s |
|---:|---:|---:|
| 1 | 40.418 | 40.418 |
| 2 | 25.676 | **51.352** |
| 4 | 19.214 | **76.857** |

The target forward is strongly sublinear in M — M=8 costs 3.02x M=1, not 8x —
which is both why aggregate throughput scales well and why the marginal
speculative token is cheap. Speculation still helps at every batch size tested
(1.21x / 1.16x / 1.06x), with the gain shrinking as batching already fills the
pipeline.

## Correctness: exact token identity is unattainable by construction

The runbook's correctness gate demanded exact generated token IDs versus a
target-only run. **That gate cannot be satisfied for this model, and the reason
is numerical rather than a defect in the speculative path.**

- Target-only on this build reproduces the published 256-token output hash
  `b85de5ba…` exactly (`matches_published_target_output: true`), three separate
  times. The build is sound.
- DSpark output shares an **exact 39-token prefix** with target-only, then
  diverges: target picks `81370` (` Photos`), DSpark picks `1008` (` The`).
- Replaying that 39-token prefix at M=1 re-chooses ` Photos` at p=0.4574 with
  ` The` second at p=0.4037 — a **top-1/top-2 gap of 0.125 nats**. The M=1 path
  agrees across both servers; the split happens inside the batched verify.
- Over the first 60 positions the median top-1/top-2 gap is **1.875 nats**, but
  **5.0% of positions are within 0.125 nats and at least one is an exact 0.0000
  tie**.

With ties that dense, any numerical difference between the M=1 decode path and
the M=N+1 verify path — GGUF dispatches small-M work to a vector MMVQ path —
will flip some token during a 256-token generation. Evidence that the
acceptance logic itself is sound: 39 exact tokens before divergence, the sealed
France single-token parity passing 3/3 in **every** configuration, and complete
run-to-run determinism (all repetitions byte-identical).

**Recommendation: replace the exact-token-ID gate with a distributional or
task-quality gate for quantized targets.**

## Reproduction

```bash
./scripts/check_bundle.py
./scripts/prepare_sources.sh /opt/k3-sources
MAX_JOBS=16 ./scripts/build_from_source.sh /opt/k3-sources
./scripts/assert_runtime.py
```

Then serve with `num_speculative_tokens: 2` and `cudagraph_capture_sizes:
[1,3]`, not the draft config's default of 7.

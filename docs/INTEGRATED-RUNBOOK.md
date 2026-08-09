# Integrated Kimi-K3 Neuron GGUF + DSpark TP3 runbook

Status: target-only PIECEWISE graph inference is GPU-qualified on 3 x H200.
The combined DSpark path is also GPU-qualified on 3 x H200 as of 2026-08-09:
it constructs, serves, and clears the 1.15x promotion gate. Full receipt:
`evidence/DSPARK-TP3-H200.md`. Read "Choosing num_speculative_tokens" and
"Correctness, acceptance, and speed gates" below before picking a
`num_speculative_tokens` value or writing an equality-based correctness
check -- the draft config's default of 7 is measurably the worst point on
this target's curve, and exact target-token identity is not achievable on a
quantized target.

## Fastest credible path

Use all of the following together:

1. official vLLM Kimi-K3 target implementation;
2. patched official GGUF plugin for the pruned Neuron tensor layout;
3. TP3 with PyNCCL (`--disable-custom-all-reduce`);
4. breakable target PIECEWISE CUDA graphs;
5. official Inferact DSpark draft loaded separately from safetensors;
6. target graphs at M=1 plus a workload-tuned M=N+1 (3 for prose, 4 for
   coding) and a matching independent FULL draft graph;
7. TRITON_MLA for the first H200 correctness run.

Do not begin with eager inference or the optional Hopper FlashMLA change.
Eager target-only inference measured 6.632 token/s, while an M=1 PIECEWISE
graph reached 34.339 token/s over three serial 256-token runs.

## Exact versions

All immutable revisions live in `pins.env`:

- vLLM `75231eff2f3873e2bce7cc9558bb5227ea70b808`;
- vLLM GGUF plugin `d94067060884ea87766f12010c3a8b9c2d6715cc`;
- GGUF `fc23910006796671aecd5551d425b5e77b61d2f2`;
- tokenizer `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`;
- DSpark draft `cf6b8244620e7ea4b0651d214f28e89eac75bed6`.

The draft has five layers, 64 attention heads, width 14,336, and target block
IDs `[2,23,47,71,89]`. The TP3 patch pads heads 64 to 66 and width 14,336 to
14,337 with mathematical zeros. KimiLinear capture boundaries must be
`(3,24,48,72,90)` after the required `+1` conversion.

## Patch order

Run the authoritative script; do not apply a numeric glob:

```bash
./scripts/prepare_sources.sh /opt/k3-sources
```

It applies these plugin patches:

```text
patches/gguf-plugin/0001-feat-add-Kimi-K3-Neuron-GGUF-adapter.patch
patches/gguf-plugin/0005-fix-kimi-k3-dequantize-latent-projections-for-native.patch
patches/gguf-plugin/0007-fix-moe-guard-CUDA-vector-grid-z-limit.patch
patches/gguf-plugin/0009-fix-honor-distributed-TP-rank-in-GGUF-fused-loaders.patch
```

Then these vLLM patches:

```text
patches/vllm/0004-fix-kimi-k3-pad-vocabulary-for-odd-TP-sizes.patch
patches/vllm/0008-fix-kimi-k3-preserve-precision-sensitive-GGUF-weight.patch
patches/vllm/0010-kimi-k3-dspark-gguf-target-tp3.patch
patches/vllm/0012-dspark-draft-config-format-isolation.patch
patches/vllm/0013-kimi-linear-eagle3-target-bridge.patch
patches/vllm/0014-dflash-full-cg-with-piecewise-target.patch
```

The optional Hopper experiment is
`patches/optional/0011-hopper-flashmla-noncausal-dspark.patch`. It is never part
of the first paid run.

## Preflight

Before allocating GPUs:

```bash
python3 scripts/check_bundle.py
./scripts/prepare_sources.sh /opt/k3-sources
MAX_JOBS=16 ./scripts/build_from_source.sh /opt/k3-sources
python3 scripts/assert_runtime.py
```

Expected source-contract output:

- DSpark TP3 padding and loader tests: 3/3;
- independent draft config discovery: PASS;
- KimiLinear EAGLE3 bridge: 4/4;
- graph resolver: 6/6;
- FP32 router/AttnRes contract: PASS;
- `git diff --check` and Python compilation: PASS.

## Download and integrity

```bash
hf auth login
./scripts/download_assets.sh /models/k3-neuron /models/k3-tokenizer /models/k3-dspark
```

The script validates all nine GGUF shards, the 24,696-byte chat template, and
the four pinned tokenizer files. Never put an HF token in this repository or a
launch command.

## Establish the contemporaneous target baseline

```bash
export TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf
export TARGET_TOKENIZER=/models/k3-tokenizer
./scripts/serve_target.sh
```

In a second shell:

```bash
./scripts/verify_server.py --model k3-neuron
./scripts/benchmark_server.py \
  --model k3-neuron \
  --chat-template /models/k3-neuron/k3_chat_template.jinja \
  --tokens 256 \
  --repetitions 3
```

Record HBM after model load and graph capture, KV capacity, ECC counters, exact
build assertions, and the three individual speeds. The historical 34.339
token/s is a sanity reference, not a replacement for this baseline.

Stop on a failed sealed token, corrupted text, model-loader warning, graph
fallback, or less than 1 GiB free HBM per rank.

## Choosing num_speculative_tokens

Do not use the draft config's own default of 7. Measured on 3 x H200 (full
data: `evidence/DSPARK-TP3-H200.md`), single stream, 256-token contract,
temperature 0, against a 34.875 token/s contemporaneous target-only baseline:

| N | prose median | coding median |
|---:|---:|---:|
| 1 | 39.113 | - |
| 2 | **42.464** | 49.273 |
| 3 | 41.325 | **52.454** |
| 5 | 40.028 | 45.033 |
| 7 (draft-config default) | **36.056** | - |

N=7 gives only 1.034x target-only while N=2 gives 1.218x. N=5 and N=7 emit an
identical 2.844 tokens/step -- speculative positions 5 and 6 contribute
nothing measurable while still costing about 8 ms per step.

Use **workload-dependent N**, not a single fixed value:

| workload | `num_speculative_tokens` | `cudagraph_capture_sizes` |
|---|---:|---|
| low-entropy / coding | 3 | `[1,4]` |
| high-entropy / prose | 2 | `[1,3]` |

N=3 on prose (41.134) is slightly *worse* than N=2 (42.464), so this is a
genuine trade, not a free win -- pick the row that matches the workload you
are actually serving, and re-measure if your workload is neither.

**Why this shape: the break-even rule.** Fitted cost is
`step(N) ~= 37.2 + 5.95*N` ms against a 28.67 ms standalone decode step, so
speculative position *i* pays only when its acceptance probability exceeds

> **p_i > 5.95 / 28.67 = 0.2075**

Measured acceptance crosses that threshold between positions 2 and 3 on both
workloads, which is exactly where the optimum was measured. Use this rule to
retune N for any other quantization or draft rather than assuming these exact
values transfer.

Keep N=7 in mind only as a documented negative result: it is the draft
config's shipped default and the worst measured point on this target, not a
recommendation.

## First integrated DSpark launch

Stop the target-only server before launching DSpark. Then:

```bash
export TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf
export TARGET_TOKENIZER=/models/k3-tokenizer
export DRAFT_MODEL=/models/k3-dspark
export K3_ACK_DSPARK_UNVERIFIED=1
./scripts/serve_dspark.sh
```

The launcher fixes the first-run contract to a workload-tuned
`num_speculative_tokens` -- see "Choosing num_speculative_tokens" above.
There is no single correct value; the example below is the prose/general
default (`NUM_SPECULATIVE_TOKENS=2`, the qualified config the promotion gate
below was cleared against). Set `NUM_SPECULATIVE_TOKENS=3` and
`CUDAGRAPH_CAPTURE_SIZES=[1,4]` for a coding workload instead:

```json
{
  "method": "dspark",
  "num_speculative_tokens": 2,
  "attention_backend": "TRITON_MLA",
  "draft_sample_method": "greedy",
  "rejection_sample_method": "standard",
  "draft_load_config": {"load_format": "safetensors"}
}
```

**Never leave this at the draft config's own default of 7** -- on this IQ1_S
target N=7 is measurably the worst point on the curve.

It also uses:

- TP3 target and TP3 zero-padded draft;
- 3 GiB explicit KV cache per rank;
- target PIECEWISE captures `[1,N+1]` (`[1,3]` prose default, `[1,4]`
  coding);
- a matching independent FULL draft capture at M=N+1;
- maximum model length 4,096 and one sequence;
- loopback binding and the pinned local Kimi chat template.

## Construction and memory gates

Proceed only if every rank shows all of the following:

1. draft heads `64 -> 66` and width `14336 -> 14337`;
2. draft weights use the safetensors loader, never the GGUF plugin;
3. EAGLE3 boundaries are `(3,24,48,72,90)` with five auxiliary states;
4. target captures M=1 and the workload-tuned M=N+1 (3 prose, 4 coding) and
   the draft captures a matching FULL M=N+1 graph;
5. KV capacity is at least 4,096 tokens;
6. at least 1,024 MiB physical HBM remains after all captures;
7. total graph-capture memory delta is no more than 2 GiB per rank;
8. ECC errors remain zero.

Stop immediately on OOM, missing or unexpected weights, shape errors, draft
replication, raw zero-based EAGLE3 IDs, eager fallback, or capture failure.

## Correctness, acceptance, and speed gates

1. Run `verify_server.py --model k3-neuron-dspark`; require token ID 17374 in
   all 3 runs.
2. Compare target-only and speculative outputs on 16 fixed diverse prompts,
   128 output tokens each, at temperature zero and seed zero.
3. Do not gate on exact generated token IDs against target-only -- see below.
4. Run three fixed 256-token outputs and record drafted tokens, accepted tokens
   by position, mean accepted/emitted length, draft latency, M=N+1 target
   verifier latency, NCCL time, and end-to-end speed.
5. Compare with the contemporaneous target-only server using the same build,
   prompt, tokenizer, cache size, and seed.

**Exact token-ID identity between target-only and DSpark is not achievable on
this quantized target, and a mismatch alone is not evidence of a defect.**
GGUF dispatches small-M decode (M=1) to a vector MMVQ path that is
numerically different from the batched M=N+1 verify path. Over the first 60
positions of the sealed prompt the median top-1/top-2 gap is 1.875 nats, but
5.0% of positions sit within 0.125 nats of a tie (one is an exact 0.0000-nat
tie); target-only and DSpark share an exact 39-token prefix, then diverge at
position 40 accordingly. That density of near-ties guarantees some token
flips over a 256-token generation even when the speculative path is
implemented correctly -- it is a property of quantization, not a defect. Gate
correctness instead on:

- the sealed France prompt returning token ID 17374 in all 3 runs (item 1,
  unchanged -- this parity gate DOES hold in every configuration measured);
- byte-identical output across repeated runs at a fixed seed (run-to-run
  determinism -- also holds: all repetitions were byte-identical to 15
  decimals);
- a task-quality read of the 16-prompt suite (item 2) rather than per-token
  equality.

Full divergence analysis: the correctness section of
`evidence/DSPARK-TP3-H200.md`.

The repository includes a fixed-prompt A/B harness. Save the target-only
results before stopping that server, then repeat against DSpark:

```bash
mkdir -p results
./scripts/qualification_suite.py \
  --model k3-neuron --label target >results/target-qualification.jsonl
./scripts/benchmark_server.py \
  --model k3-neuron \
  --chat-template /models/k3-neuron/k3_chat_template.jinja \
  --tokens 256 --repetitions 3 >results/target-benchmark.json

# After launching the DSpark server:
./scripts/qualification_suite.py \
  --model k3-neuron-dspark --label dspark >results/dspark-qualification.jsonl
./scripts/benchmark_server.py \
  --model k3-neuron-dspark \
  --chat-template /models/k3-neuron/k3_chat_template.jinja \
  --tokens 256 --repetitions 3 >results/dspark-benchmark.json

./scripts/compare_qualification.py \
  results/target-qualification.jsonl results/dspark-qualification.jsonl
./scripts/analyze_qualification.py \
  results/target-qualification.jsonl \
  results/dspark-qualification.jsonl \
  results/target-benchmark.json \
  results/dspark-benchmark.json
```

Promote only if speculative median sustained decode is at least 1.15x the
target-only median and the 95% prompt-bootstrap lower bound is greater than
1.00x. Against the historical 34.339 token/s reference, the point threshold is
39.49 token/s; the 39-69 token/s interval was only a pre-GPU planning range.
The measured results are in "Choosing num_speculative_tokens" above and in
`evidence/DSPARK-TP3-H200.md` -- N=2 clears this gate at 1.218x.

## Probabilistic/block follow-up

Only after greedy/standard passes, restart and change both variables together:

```bash
export DRAFT_SAMPLE_METHOD=probabilistic
export REJECTION_SAMPLE_METHOD=block
```

Repeat all parity, acceptance, HBM, ECC, and throughput gates. Do not infer a
threefold speedup from GB300 results; those use native MXFP4, SM103 kernels, and
TP16 rather than IQ1_S on TP3 H200.

## Optional Hopper FlashMLA gate

Only after the complete TRITON_MLA run passes, prepare a fresh source tree with:

```bash
export K3_APPLY_OPTIONAL_HOPPER_FLASHMLA=1
./scripts/prepare_sources.sh /opt/k3-sources-flashmla
```

Build it and launch with `SPEC_BACKEND=FLASHMLA`. Keep the optional patch only
if all greedy target token IDs match the Triton run, acceptance does not
regress, there is no non-causal attention mismatch, and median end-to-end
decode improves by at least 3%.

## Profile only after a correct slow result

If DSpark is correct but misses the speed gate, profile one steady M=1 target
step and one M=N+1 verifier step. Attribute time to GGUF expert MMVQ, other GGUF
linears, KDA/MLA/AttnRes, NCCL, and uncovered CPU/launch gaps. Do not port
native MXFP4 kernels blindly: they cannot consume IQ1_S, and K3's native skinny
GEMM path is SM103-specific.

## Current unresolved items

The integrated target-`[1,N+1]`-PIECEWISE plus DSpark path is now GPU-qualified
(`evidence/DSPARK-TP3-H200.md`): capture/replay, acceptance distribution,
speed, and post-capture HBM/KV measurements are all measured there. Still
open:

- Hopper numerical oracle for the optional patch -- only a speed comparison
  against Triton (at N=7) has been run, not a token-match oracle;
- full upstream CUDA pytest beyond the hermetic source contracts.

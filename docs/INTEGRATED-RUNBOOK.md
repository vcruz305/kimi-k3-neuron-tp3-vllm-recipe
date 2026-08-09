# Integrated Kimi-K3 Neuron GGUF + DSpark TP3 runbook

Status: target-only PIECEWISE graph inference is GPU-qualified on 3 x H200.
The combined DSpark path is clean-apply and CPU-contract validated but has not
yet passed its real-GPU construction, parity, acceptance, or speed gates.

## Fastest credible path

Use all of the following together:

1. official vLLM Kimi-K3 target implementation;
2. patched official GGUF plugin for the pruned Neuron tensor layout;
3. TP3 with PyNCCL (`--disable-custom-all-reduce`);
4. breakable target PIECEWISE CUDA graphs;
5. official Inferact DSpark draft loaded separately from safetensors;
6. target graphs at M=1 and M=8 plus an independent FULL draft graph at M=7;
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

## First integrated DSpark launch

Stop the target-only server before launching DSpark. Then:

```bash
export TARGET_GGUF=/models/k3-neuron/k3-neuron-iq1s-00001-of-00009.gguf
export TARGET_TOKENIZER=/models/k3-tokenizer
export DRAFT_MODEL=/models/k3-dspark
export K3_ACK_DSPARK_UNVERIFIED=1
./scripts/serve_dspark.sh
```

The launcher fixes the first-run contract to:

```json
{
  "method": "dspark",
  "num_speculative_tokens": 7,
  "attention_backend": "TRITON_MLA",
  "draft_sample_method": "greedy",
  "rejection_sample_method": "standard",
  "draft_load_config": {"load_format": "safetensors"}
}
```

It also uses:

- TP3 target and TP3 zero-padded draft;
- 3 GiB explicit KV cache per rank;
- target PIECEWISE captures `[1,8]`;
- independent FULL M=7 draft capture;
- maximum model length 4,096 and one sequence;
- loopback binding and the pinned local Kimi chat template.

## Construction and memory gates

Proceed only if every rank shows all of the following:

1. draft heads `64 -> 66` and width `14336 -> 14337`;
2. draft weights use the safetensors loader, never the GGUF plugin;
3. EAGLE3 boundaries are `(3,24,48,72,90)` with five auxiliary states;
4. target captures M=1 and M=8 and the draft captures FULL M=7;
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
3. Require exact generated token IDs for every prompt under greedy/standard
   rejection.
4. Run three fixed 256-token outputs and record drafted tokens, accepted tokens
   by position, mean accepted/emitted length, draft latency, M=8 target verifier
   latency, NCCL time, and end-to-end speed.
5. Compare with the contemporaneous target-only server using the same build,
   prompt, tokenizer, cache size, and seed.

The repository includes an exact A/B harness. Save the target-only results
before stopping that server, then repeat against DSpark:

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
39.49 token/s. The 39-69 token/s interval is only a planning range.

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
step and one M=8 verifier step. Attribute time to GGUF expert MMVQ, other GGUF
linears, KDA/MLA/AttnRes, NCCL, and uncovered CPU/launch gaps. Do not port
native MXFP4 kernels blindly: they cannot consume IQ1_S, and K3's native skinny
GEMM path is SM103-specific.

## Current unresolved items

- integrated target `[1,8]` plus draft M=7 GPU capture/replay;
- real acceptance distribution and DSpark speed;
- post-capture HBM/KV measurements with the explicit 3 GiB cache;
- Hopper numerical oracle for the optional patch;
- full upstream CUDA pytest beyond the hermetic source contracts.

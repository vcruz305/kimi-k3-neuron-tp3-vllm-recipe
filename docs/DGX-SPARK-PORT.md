# Porting this recipe to a DGX Spark cluster

Feasibility study and phased plan for running the Kimi-K3 Neuron IQ1_S GGUF on
NVIDIA DGX Spark (GB10) nodes under vLLM, with the DSpark speculative draft.

Status: **Phase 0 complete (measured).** Phases 1-5 are planned, not executed.
Nothing in this document is a Spark measurement; the Spark numbers are estimates
and are labelled as such.

## Summary

| Question | Answer |
|---|---|
| Does it run on **3** Sparks? | **No, not at the current prune width.** Weights alone are 103.97 GiB/rank against ~110 GiB usable, before ~22 GiB/rank of measured runtime overhead. |
| Does it run on **4** Sparks? | **Yes.** 78.53 GiB/rank weights, ~31 GiB headroom. This is the recommended target. |
| Does 3 Sparks work at a narrower prune? | **Yes, at k=1024** (33.3% FFN retention): 76.45 GiB/rank. Requires re-running compression, and the quality cost is unmeasured. **k=1280 does not fit** (90.21 GiB/rank against an ~88 GiB budget). |
| Expected speed | **10-25 token/s** estimated, against 7.90 token/s measured on 4-Spark SparkInfer today. Not 50. |
| Biggest risk | The speculative half may not transfer at all - see "Regime shift". |

## Phase 0 - measured residency attribution

Method: pure GGUF header inventory, no GPU (`scripts/`-adjacent
`phase0_inventory.py`, receipt in `evidence/`). Totals are exact tensor bytes.

Total: **330,160,710,016 bytes = 307.49 GiB** across 9 shards.

| bucket | GiB | share | ggml types |
|---|---:|---:|---|
| routed expert FFN | **247.63** | **80.5%** | IQ1_S |
| attention / latent projections | 19.37 | 6.3% | F32, Q8_0 |
| other | 13.73 | 4.5% | F32, Q8_0 |
| shared expert FFN | 12.03 | 3.9% | Q8_0 |
| lm_head | 10.29 | 3.3% | F16, Q8_0 |
| router | 2.20 | 0.7% | F32 |
| token embedding | 2.19 | 0.7% | F16 |
| KDA + norms | 0.05 | 0.0% | F32 |

By quantization: **IQ1_S 247.63 GiB (80.5%)**, Q8_0 53.23 GiB (17.3%),
F16 4.38 GiB (1.4%), F32 2.25 GiB (0.7%).

Two facts that matter for the port:

1. **The model is 80.5% routed-expert FFN.** Everything else is rounding error.
   Size is therefore almost linear in the prune width k.
2. **Latent projections are already Q8_0/F32 on disk, not IQ1_S.** So patch
   `0005`, which dequantizes them to bf16, costs about **+19.37 GiB total =
   +6.46 GiB/rank at TP3** - not the +58 GiB it would cost if they were IQ1_S.
   That bounds how much reverting it could ever save.

### Per-rank residency

| | TP3 | TP4 |
|---|---:|---:|
| shardable / N | 101.76 | 76.32 |
| + replicated (router + norms) | 2.21 | 2.21 |
| **= weights per rank** | **103.97 GiB** | **78.53 GiB** |
| measured H200 residency | 126.10 GiB | - |
| **unexplained runtime overhead** | **22.13 GiB/rank** | - |

The 22.13 GiB gap is the number a Spark port has to fight. Identified
contributors, in descending confidence: patch `0005` bf16 dequantization
(+6.46), lm_head and token embedding if replicated rather than vocab-sharded
(+8.3 combined), patch `0004` vocab padding, allocator and activation
workspace, NCCL buffers. **Attributing the remainder empirically is the first
task of Phase 1** - revert `0005` and re-read vLLM's `Model loading took` line.

### DGX Spark fit

A Spark has 128 GB unified memory (119.2 GiB) shared with the OS and CPU;
assume **~110 GiB usable** by the serving process.

| config | weights/rank | + 22 GiB overhead | verdict |
|---|---:|---:|---|
| TP3 | 103.97 | 126.1 | **does not fit** |
| **TP4** | **78.53** | **100.7** | **fits, ~9 GiB spare** |

**Recommendation: 4 Sparks, not 3.** Two patches also become unnecessary at
even TP:

- `0004` (vocabulary padding for odd TP sizes) - not needed.
- `0010` (exact TP3 zero-padding of the draft, `heads 64 -> 66`,
  `intermediate 14336 -> 14337`) - **not needed**: 64 heads divide evenly by 4.
  That patch exists only because 3 does not divide 64.

### Prune width versus fit

**Only the routed experts scale with k.** An earlier revision of this table
scaled the whole 259.66 GiB expert bucket, which is wrong: that bucket is
routed 247.63 GiB **plus shared 12.03 GiB**, and the shared experts are
invariant by construction. The build constraint `k * n_expert_shared == 6144`
holds total shared width fixed at 6144, so `ffn_*_shexp` is `[7168, 6144]`
Q8_0 at every legal k — at k=1536 that is 4 shared experts, at k=1024 it is 6.
Verified directly against the shipped GGUF's tensor shapes.

So the correct split is **k-scaling 247.63 GiB, k-independent 59.85 GiB**
(shared experts 12.03 + attention 29.58 + other 13.87 + embed/output 4.38).
The 59.85 GiB is a floor no prune width can go below.

| k | FFN retention | total GiB | TP3 GiB/rank | TP4 GiB/rank |
|---:|---:|---:|---:|---:|
| 512 | 16.7% | 142.39 | 48.94 | 37.26 |
| 768 | 25.0% | 183.66 | 62.69 | 47.57 |
| **1024** | **33.3%** | **224.94** | **76.45** | **57.89** |
| 1280 | 41.7% | 266.21 | 90.21 | 68.21 |
| **1536 (current)** | **50.0%** | **307.49** | **103.97** | **78.53** |
| 1792 | 58.3% | 348.75 | 117.72 | 88.85 |
| 2048 | 66.7% | 390.02 | 131.48 | 99.16 |

Per-rank uses the Phase 0 replication model: `(total - 2.21) / N + 2.21`, where
2.21 GiB of router and norms is replicated on every rank. It reproduces the
measured k=1536 row exactly at both TP3 and TP4.

With a ~88 GiB weight budget per Spark (110 usable minus 22 overhead),
**k=1024 fits 3 Sparks at 76.45 GiB/rank**. Note `k=1024` satisfies the
documented constraint (`k * n_expert_shared == 6144` with `n_expert_shared = 6`,
and `k % 256 == 0`).

**k=1280 no longer fits 3 Sparks.** The old table put it at 89.54, described as
"right on the edge" of the 88 GiB budget; corrected, it is **90.21 GiB/rank** —
over. Anyone relying on the previous number should re-plan at k=1024.

Going the other way is not viable: k=2048 (66.7% retention) needs
**131.48 GiB/rank at TP3**, worse than today and beyond any Spark.

## Blocker 1 - sm_121 on aarch64

Tracked upstream as
[vllm-project/vllm#36821](https://github.com/vllm-project/vllm/issues/36821):
stock PyTorch ships CUDA kernels only through **sm_120**, so PyTorch, vLLM, and
the GGUF plugin must all be rebuilt for **sm_121** on **aarch64**.

The container shortcut used for the H200 run - overlaying patched Python onto
`vllm/vllm-openai:nightly`, which happens to match `pins.env` exactly - **has no
equivalent here.** There is no prebuilt sm_121 aarch64 image at this patch base.
This is a genuine source build.

The plugin's IQ1_S MMVQ and MoE kernels are hand-written CUDA and must be
numerically correct on a new architecture, not merely compile.
`diagnostics/k3_iq1s_tp3_micro_oracle.py` exists for exactly this and ports
unchanged. Patch `0007`'s CUDA grid-z guard also needs its limits re-checked
against sm_121.

## Blocker 2 - memory

Covered above. The task is reducing 22.13 GiB/rank of runtime overhead, of which
at most ~6.5 GiB is recoverable by reverting `0005` (and `0005` exists for a
reason - native K3 needs the dense latent path, so a Blackwell-compatible
alternative would be required).

## Not a blocker - interconnect

This corrects an earlier assumption in this project.

Each Spark has an onboard ConnectX-7 at **200 GbE**. Two nodes link with a single
200G QSFP56 DAC cable; **3 or more require a switched fabric.**

The per-layer all-reduce payload is `hidden 7168 x bf16 = 14,336 bytes` -
latency-bound, not bandwidth-bound. At roughly 30 microseconds per collective,
93 layers x 2 gives about **5.6 ms/token of network time**. Against an H200 step
of 28.67 ms that would be ~20% and serious. Against an estimated Spark step in
the hundreds of milliseconds it is **~4% - noise**.

The interconnect is the *least* of the three constraints, precisely because the
compute is slow enough to hide it. Note also that TP3 has **no fast all-reduce
path at all** - custom all-reduce does not support world size 3, and
`SymmMemCommunicator: World size 3 not supported` - so PyNCCL is already the
path on H200 and nothing is lost by moving to Ethernet. TP4 restores the
even-world-size options.

## What ports, what dies, what improves

| patch | Spark status |
|---|---|
| `0001` GGUF adapter, `0005`, `0008`, `0009` | port; re-validate numerics on sm_121 |
| `0007` MoE grid-z guard | port; re-check sm_121 grid limits |
| `0012`, `0013`, `0014` (Python only) | port unchanged |
| `0004`, `0010` | **unnecessary at TP4** |
| `0011` Hopper FlashMLA | **dead** - Hopper-only |

Dropping `0011` is an upgrade. Official `FLASHINFER_MLA` DSpark support targets
**Blackwell**, so a Spark deployment uses the *supported* attention path instead
of this repository's hand-rolled Hopper wrapper. Spark is the architecture the
upstream DSpark work actually targets.

## Speed estimate

**Bandwidth.** Spark unified LPDDR5X runs at **~273 GB/s** against H200 HBM3e at
4.8 TB/s - about 17x less.

**Cross-check.** If GB10 kernels run ~6x slower than H200's, the measured
28.67 ms target step becomes ~170 ms, i.e. **~5.9 token/s** - which lands almost
exactly on SparkInfer's measured **6.21 token/s** at TP3. Two independent
methods agreeing is weak but real evidence the estimate is not wild.

**The upside case.** vLLM's entire advantage on H200 was CUDA graphs:
**6.632 -> 34.339 token/s, a 5.2x pure launch-overhead removal.** Spark's Grace
ARM cores are *worse* at kernel launch than x86, so graph capture should matter
**more** there. If SparkInfer at 6.21-7.90 token/s is leaving a similar multiple
on the table, target-only could reach **10-20 token/s**.

Estimated outcome: **10-25 token/s**, versus 7.90 measured today. A real
multiple, but the 273 GB/s wall makes 50 token/s implausible.

## Regime shift - why the speculative half may not transfer

This is the most important prediction in this document, and it is falsifiable.

The break-even rule measured on H200 says speculative position *i* pays only when
its acceptance probability exceeds `m / target_step`, where `m` is the marginal
cost of one speculative token. On H200 that ratio is **0.2075**, because neither
model is bandwidth-bound and the draft is cheap for a structural reason: it has
**5 layers versus the target's 93**, so it is launch-bound and small.

On a 273 GB/s machine the regime flips to bandwidth-bound, and the relevant
comparison becomes bytes read per forward pass:

- draft is **dense**: 7.1 GB bf16, about **1.8 GB/rank at TP4**;
- target is **sparse MoE**: about 12.3 GB active per token, about
  **3.1 GB/rank at TP4**.

Ratio **1.8 / 3.1 = 0.58**. If Spark lands in the bandwidth-bound regime, each
draft pass costs ~58% of a target forward, and the break-even threshold rises
from 0.2075 to ~0.58. Only **p0 (0.744 measured on coding)** clears that.

Predicted consequences on Spark:

- optimal `num_speculative_tokens` collapses from **3 to 1**;
- on prose (p0 = 0.547) **speculation becomes net-negative** and should be off;
- the 1.5x speculative win measured on H200 largely **does not transfer**.

If true, the value of this port rests on CUDA graphs, not on DSpark. Phase 5
tests it directly by re-deriving `m / target_step` on Spark hardware.

## Phased plan with gates

**Phase 0 - residency attribution. COMPLETE.** Verdict: TP4, not TP3, at the
current prune width; k=1024 would open TP3.

**Phase 1 - empirical overhead attribution.** On any H200 box (~$20): revert
`0005`, re-read `Model loading took`, confirm the +6.46 GiB prediction; determine
whether lm_head and embeddings are replicated or sharded.
*Gate: is <=78 GiB/rank at TP4 reachable with the overhead included?*

**Phase 2 - sm_121 aarch64 build.** PyTorch + vLLM + GGUF plugin.
*Gate: `k3_iq1s_tp3_micro_oracle.py` passes on sm_121.*

**Phase 3 - two-Spark TP2 smoke on a small GGUF** over the direct QSFP56 DAC
cable. De-risks Ray, NCCL, and the fabric **without buying a switch**.
*Gate: clean load, generate, exit on 2 nodes.*

**Phase 4 - four-Spark TP4 bring-up, full model load.**
*Gate: loads with >=4 GiB free and >=4096 KV tokens per rank; ECC zero.*

**Phase 5 - graphs, then DSpark with `FLASHINFER_MLA`.** Measure target-only
with and without CUDA graphs first - that is where the 2-3x either appears or
does not. Then re-derive `m / target_step` and retune N per the break-even rule.
*Gate: does speculation still pay at any N?*

Phase 1 is the cheap kill-switch. Run it before buying fabric hardware.

## Prior art

Multi-Spark vLLM is established, though at far smaller model scale:

- [vLLM on the DGX Spark](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark) - official
- [mark-ramsey-ri/vllm-dgx-spark](https://github.com/mark-ramsey-ri/vllm-dgx-spark) - 1-to-N nodes, 3+ via switched fabric
- [sparkrun.dev multi-node tensor parallelism](https://sparkrun.dev/tutorials/multi-node/)
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
- [DeepSeek-V4-Flash across two Sparks](https://route179.dev/2026/07/28/deepseek-v4-flash-dual-dgx-spark-eks-hybrid/)
- [Known Ray + TP2 engine-init failure](https://forums.developer.nvidia.com/t/two-spark-cluster-tensor-parallel-size-2-causing-engine-initialization-failure-with-qwen3-vl-30b-ray-vllm/362518) - multi-node TP on Spark is not turnkey

To our knowledge no 2.78T-parameter MoE has been served across DGX Sparks under
vLLM, and none with a speculative draft. That, plus an sm_121 validation of
IQ1_S kernels, is where the novelty of this port would sit.

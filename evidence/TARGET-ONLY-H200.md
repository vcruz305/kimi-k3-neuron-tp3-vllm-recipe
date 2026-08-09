# Target-only H200 measurement contract

This is a sanitized summary of the target-only GPU receipt. It contains no pod
identity, network address, credential, or model file.

## Environment

- hardware: 3 x NVIDIA H200;
- target: Kimi-K3 Neuron IQ1_S GGUF, 330.2 GB / 307.49 GiB;
- tensor parallelism: TP3, one sequence;
- measured vLLM binary: `0.26.1rc1.dev511+g700d39b55`;
- PyTorch: `2.13.0+cu130`;
- NCCL after the plugin install: `2.29.7`;
- memory allocator: `--gpu-memory-utilization 0.95`;
- custom all-reduce disabled, PyNCCL selected;
- breakable CUDA graph enabled, target PIECEWISE capture `[1]`;
- model residency after capture: about 138,089 MiB per H200;
- free HBM after capture: about 5,068 MiB per H200;
- volatile uncorrected ECC: zero.

The complete CUDA toolkit and driver versions were not captured, so they are
not claimed here.

## Fixed-output benchmark

- shipped chat template SHA-256:
  `05bb501f8ac31fa6b0bf04803b5ada49abf9cdd51c3c90a4719b739df0000722`;
- template size: 24,696 bytes;
- user prompt: `Explain how photosynthesis works, including both the
  light-dependent reactions and the Calvin cycle.`;
- `thinking_effort=max`, generation prompt enabled;
- rendered prompt: 105 tokens;
- temperature 0, top-p 1, seed 0;
- streaming completions, three serial repetitions.

| Fixed output | Individual sustained decode rates | Median |
|---:|---|---:|
| 64 tokens | 33.856, 29.960, 30.092 token/s | 30.092 token/s |
| 256 tokens | 33.204, 34.339, 34.684 token/s | **34.339 token/s** |

All three 256-token outputs were byte-identical with SHA-256
`b85de5bae72080bed1eb3c28d68da4f19e80f2aa45bd7c0bf141f30cb30244b1`.
The sanitized source receipt for the three 256-token rows had SHA-256
`ed8f608f93423f31badb943c882cc07375ac857d0c1a274ce9857110935d5810`;
the raw receipt is intentionally not published because it contains local
runtime paths.

## Correctness gates

- sealed France prompt: token ID 17374 (` Paris`) in 3/3 runs;
- five-prompt cross-engine equality: 4/5;
- the fifth case was a close log-probability tie, so broad cross-engine
  byte-for-byte identity is not claimed.

These numbers apply to the measured target-only `[1]` graph contract. They do
not qualify the integrated target `[1,8]` + DSpark M=7 graph path.

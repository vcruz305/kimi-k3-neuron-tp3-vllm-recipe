# Exact source and install recipe

The repository is an overlay over two exact clean checkouts. Do not apply the
patches by numeric glob: DSpark patch `0010` and graph patch `0014` have a
required semantic order that is encoded in `scripts/apply_patches.sh`.

## Automated path

```bash
./scripts/check_bundle.py
./scripts/prepare_sources.sh /opt/k3-sources
MAX_JOBS=16 ./scripts/build_from_source.sh /opt/k3-sources
./scripts/assert_runtime.py
```

`prepare_sources.sh` does all of the following:

1. clones vLLM at `75231eff2f3873e2bce7cc9558bb5227ea70b808`;
2. clones vLLM GGUF plugin at
   `d94067060884ea87766f12010c3a8b9c2d6715cc`;
3. refuses dirty or incorrect source trees;
4. verifies `SHA256SUMS`;
5. applies only the mandatory chain below;
6. runs the hermetic CPU source-contract suite and Python compilation.

`build_from_source.sh` builds the exact vLLM source and installs the patched
plugin with `--no-deps`. It explicitly installs the plugin's pinned
`gguf==0.19.0` runtime dependency first; vLLM does not install that package.
The `--no-deps` boundary is important: a normal plugin dependency resolution
changed NCCL 2.30.7 to 2.29.7 in the measured container. The recipe does not
permit a plugin install to mutate the already selected vLLM/PyTorch/NCCL
stack.

## Mandatory patch order

### vLLM GGUF plugin

```text
patches/gguf-plugin/0001-feat-add-Kimi-K3-Neuron-GGUF-adapter.patch
patches/gguf-plugin/0005-fix-kimi-k3-dequantize-latent-projections-for-native.patch
patches/gguf-plugin/0007-fix-moe-guard-CUDA-vector-grid-z-limit.patch
patches/gguf-plugin/0009-fix-honor-distributed-TP-rank-in-GGUF-fused-loaders.patch
```

### vLLM

```text
patches/vllm/0004-fix-kimi-k3-pad-vocabulary-for-odd-TP-sizes.patch
patches/vllm/0008-fix-kimi-k3-preserve-precision-sensitive-GGUF-weight.patch
patches/vllm/0010-kimi-k3-dspark-gguf-target-tp3.patch
patches/vllm/0012-dspark-draft-config-format-isolation.patch
patches/vllm/0013-kimi-linear-eagle3-target-bridge.patch
patches/vllm/0014-dflash-full-cg-with-piecewise-target.patch
```

The optional Hopper FlashMLA patch is excluded. To source-test it only after a
successful TRITON_MLA qualification, set
`K3_APPLY_OPTIONAL_HOPPER_FLASHMLA=1` before preparing fresh sources.

## Runtime assertions

`scripts/assert_runtime.py` checks:

- installed vLLM and plugin import paths;
- source commit marker files emitted by the build;
- KimiLinear registration and the DSpark/EAGLE3 symbols;
- PyTorch CUDA and NCCL versions;
- three visible GPUs when run on the serving host.

Its JSON output is safe to attach to a bug report; it does not read tokens or
model files.

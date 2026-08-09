# Text-only Kimi-K3 Neuron configuration

`config.json` selects `KimiLinearForCausalLM` / `kimi_linear` for the text-only
Neuron GGUF. The released GGUF has no vision tower.

The pruned routed experts are 1,536 wide. The retained shared expert tensor is
6,144 wide, so the configuration uses four 1,536-wide shared experts. Native
MXFP4 quantization metadata and vision `auto_map` entries are intentionally
absent because they describe a different checkpoint layout.

Always pass this directory explicitly:

```bash
vllm serve "$TARGET_GGUF" \
  --hf-config-path ./config \
  --tokenizer "$TARGET_TOKENIZER" \
  --trust-remote-code \
  --chat-template "$TARGET_CHAT_TEMPLATE"
```

The remote tokenizer code is required only for the pinned Moonshot tokenizer
revision. The model architecture itself is registered in vLLM by the overlay.
This configuration does not replace the mandatory tensor adapter and precision
patches.

`config.json` is derived from Kimi-K3 model metadata. It is not relicensed as
original recipe code; use it under the Kimi-K3/model terms linked from the
repository's third-party notices.

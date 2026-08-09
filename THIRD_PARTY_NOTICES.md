# Third-party notices

This repository is an integration recipe. It does not redistribute model or
draft weights.

## vLLM

- Project: <https://github.com/vllm-project/vllm>
- Pinned commit: `75231eff2f3873e2bce7cc9558bb5227ea70b808`
- License: [Apache License 2.0](https://github.com/vllm-project/vllm/blob/75231eff2f3873e2bce7cc9558bb5227ea70b808/LICENSE)

Files under `patches/vllm` modify this project and remain subject to its
license and notices.

## vLLM GGUF plugin

- Project: <https://github.com/vllm-project/vllm-gguf-plugin>
- Pinned commit: `d94067060884ea87766f12010c3a8b9c2d6715cc`
- License: [Apache License 2.0](https://github.com/vllm-project/vllm-gguf-plugin/blob/d94067060884ea87766f12010c3a8b9c2d6715cc/LICENSE)

Files under `patches/gguf-plugin` modify this project and remain subject to its
license and notices.

## Kimi-K3 and tokenizer

- Model/tokenizer: <https://huggingface.co/moonshotai/Kimi-K3>
- Pinned tokenizer revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- License: [Kimi-K3 license](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/LICENSE)

## Kimi-K3 Neuron IQ1_S GGUF

- Model: <https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF>
- Pinned revision: `fc23910006796671aecd5551d425b5e77b61d2f2`
- Terms: [model card and license](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF/tree/fc23910006796671aecd5551d425b5e77b61d2f2)

## Inferact Kimi-K3 DSpark

- Draft model: <https://huggingface.co/Inferact/Kimi-K3-DSpark>
- Pinned revision: `cf6b8244620e7ea4b0651d214f28e89eac75bed6`
- License: [Kimi-K3 license shipped with the draft](https://huggingface.co/Inferact/Kimi-K3-DSpark/blob/cf6b8244620e7ea4b0651d214f28e89eac75bed6/LICENSE)

Users are responsible for satisfying all applicable model, tokenizer, draft,
upstream software, and hardware-vendor terms.

The repository's Apache-2.0 license covers original recipe scripts,
documentation, and patch contributions. `config/config.json` is derived from
Kimi-K3 model metadata and remains governed by the applicable Kimi/model terms.

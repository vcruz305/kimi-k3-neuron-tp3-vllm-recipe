#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

: "${TARGET_GGUF:?Set TARGET_GGUF to shard 00001-of-00009.gguf}"
: "${TARGET_TOKENIZER:?Set TARGET_TOKENIZER to the pinned local tokenizer directory}"

TARGET_CONFIG=${TARGET_CONFIG:-$RECIPE_ROOT/config}
TARGET_CHAT_TEMPLATE=${TARGET_CHAT_TEMPLATE:-$(dirname -- "$TARGET_GGUF")/k3_chat_template.jinja}
BIND_HOST=${BIND_HOST:-127.0.0.1}
PORT=${PORT:-8008}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-k3-neuron}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}

[[ -f "$TARGET_GGUF" ]] || { echo "missing GGUF: $TARGET_GGUF" >&2; exit 1; }
[[ -d "$TARGET_TOKENIZER" ]] || { echo "missing tokenizer: $TARGET_TOKENIZER" >&2; exit 1; }
[[ -f "$TARGET_CHAT_TEMPLATE" ]] || { echo "missing chat template: $TARGET_CHAT_TEMPLATE" >&2; exit 1; }

if [[ "$BIND_HOST" != 127.0.0.1 && "$BIND_HOST" != localhost \
      && -z ${VLLM_API_KEY:-} \
      && ${K3_ACK_UNAUTHENTICATED_NETWORK:-0} != 1 ]]; then
  echo "Refusing unauthenticated non-loopback bind. Set VLLM_API_KEY, or explicitly set K3_ACK_UNAUTHENTICATED_NETWORK=1." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

extra_args=()
if [[ -n ${VLLM_API_KEY:-} ]]; then
  echo "NOTE: --api-key can be visible to other users in the same host's process list." >&2
  extra_args+=(--api-key "$VLLM_API_KEY")
fi

exec python3 -m vllm.entrypoints.cli.main serve "$TARGET_GGUF" \
  --hf-config-path "$TARGET_CONFIG" \
  --tokenizer "$TARGET_TOKENIZER" \
  --trust-remote-code \
  --chat-template "$TARGET_CHAT_TEMPLATE" \
  --load-format gguf \
  --quantization gguf \
  --model-impl vllm \
  --tensor-parallel-size 3 \
  --disable-custom-all-reduce \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 64 \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --compilation-config '{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"cudagraph_num_of_warmups":1}' \
  --host "$BIND_HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${extra_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# shellcheck source=/dev/null
source "$RECIPE_ROOT/pins.env"

[[ ${K3_ACK_DSPARK_UNVERIFIED:-0} == 1 ]] || {
  echo "DSpark is not yet GPU-qualified. Set K3_ACK_DSPARK_UNVERIFIED=1 after reading the runbook." >&2
  exit 1
}

: "${TARGET_GGUF:?Set TARGET_GGUF to shard 00001-of-00009.gguf}"
: "${TARGET_TOKENIZER:?Set TARGET_TOKENIZER to the pinned local tokenizer directory}"

TARGET_CONFIG=${TARGET_CONFIG:-$RECIPE_ROOT/config}
TARGET_CHAT_TEMPLATE=${TARGET_CHAT_TEMPLATE:-$(dirname -- "$TARGET_GGUF")/k3_chat_template.jinja}
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the pinned, hash-verified local DSpark directory}"
SPEC_BACKEND=${SPEC_BACKEND:-TRITON_MLA}
DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-greedy}
REJECTION_SAMPLE_METHOD=${REJECTION_SAMPLE_METHOD:-standard}
BIND_HOST=${BIND_HOST:-127.0.0.1}
PORT=${PORT:-8008}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-k3-neuron-dspark}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-3221225472}

[[ -f "$TARGET_GGUF" ]] || { echo "missing GGUF: $TARGET_GGUF" >&2; exit 1; }
[[ -d "$TARGET_TOKENIZER" ]] || { echo "missing tokenizer: $TARGET_TOKENIZER" >&2; exit 1; }
[[ -f "$TARGET_CHAT_TEMPLATE" ]] || { echo "missing chat template: $TARGET_CHAT_TEMPLATE" >&2; exit 1; }
[[ -d "$DRAFT_MODEL" ]] || { echo "missing local draft: $DRAFT_MODEL" >&2; exit 1; }
(cd -- "$DRAFT_MODEL" && sha256sum -c "$RECIPE_ROOT/DSPARK-SHA256SUMS")

if [[ "$BIND_HOST" != 127.0.0.1 && "$BIND_HOST" != localhost \
      && -z ${VLLM_API_KEY:-} \
      && ${K3_ACK_UNAUTHENTICATED_NETWORK:-0} != 1 ]]; then
  echo "Refusing unauthenticated non-loopback bind. Set VLLM_API_KEY, or explicitly set K3_ACK_UNAUTHENTICATED_NETWORK=1." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_DFLASH_FULL_CUDAGRAPH_WITH_PIECEWISE_TARGET=1
export DRAFT_MODEL SPEC_BACKEND DRAFT_SAMPLE_METHOD
export REJECTION_SAMPLE_METHOD

SPEC_CONFIG=$(python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["DRAFT_MODEL"],
    "method": "dspark",
    "num_speculative_tokens": 7,
    "attention_backend": os.environ["SPEC_BACKEND"],
    "draft_sample_method": os.environ["DRAFT_SAMPLE_METHOD"],
    "rejection_sample_method": os.environ["REJECTION_SAMPLE_METHOD"],
    "draft_load_config": {"load_format": "safetensors"},
}, separators=(",", ":")))
PY
)

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
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --compilation-config '{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,8],"cudagraph_num_of_warmups":1}' \
  --speculative-config "$SPEC_CONFIG" \
  --host "$BIND_HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${extra_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_DIR TOKENIZER_DIR DSPARK_DIR" >&2
  exit 2
fi

MODEL_DIR=$1
TOKENIZER_DIR=$2
DSPARK_DIR=$3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# shellcheck source=/dev/null
source "$RECIPE_ROOT/pins.env"

command -v hf >/dev/null 2>&1 || {
  echo "The Hugging Face 'hf' CLI is required." >&2
  exit 1
}

mkdir -p -- "$MODEL_DIR" "$TOKENIZER_DIR" "$DSPARK_DIR"

hf download "$TARGET_MODEL_REPOSITORY" \
  --revision "$TARGET_MODEL_REVISION" \
  --local-dir "$MODEL_DIR" \
  --include '*.gguf' \
  --include 'k3_chat_template.jinja'

hf download "$TARGET_TOKENIZER_REPOSITORY" \
  --revision "$TARGET_TOKENIZER_REVISION" \
  --local-dir "$TOKENIZER_DIR" \
  --include 'encoding_k3.py' \
  --include 'tokenization_kimi.py' \
  --include 'tokenizer_config.json' \
  --include 'tiktoken.model'

hf download "$DSPARK_MODEL_REPOSITORY" \
  --revision "$DSPARK_MODEL_REVISION" \
  --local-dir "$DSPARK_DIR" \
  --include 'config.json' \
  --include 'model.safetensors'

(cd -- "$MODEL_DIR" && sha256sum -c "$RECIPE_ROOT/MODEL-SHA256SUMS")
(cd -- "$MODEL_DIR" && sha256sum -c "$RECIPE_ROOT/CHAT-TEMPLATE-SHA256SUMS")
(cd -- "$TOKENIZER_DIR" && sha256sum -c "$RECIPE_ROOT/TOKENIZER-SHA256SUMS")
(cd -- "$DSPARK_DIR" && sha256sum -c "$RECIPE_ROOT/DSPARK-SHA256SUMS")

echo "Pinned model, template, tokenizer, and draft assets are ready."

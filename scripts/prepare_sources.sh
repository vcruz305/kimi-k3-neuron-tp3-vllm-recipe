#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SOURCE_ROOT" >&2
  exit 2
fi

SOURCE_ROOT=$1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# shellcheck source=/dev/null
source "$RECIPE_ROOT/pins.env"

mkdir -p -- "$SOURCE_ROOT"

clone_exact() {
  local repository=$1
  local commit=$2
  local destination=$3

  if [[ -e "$destination" ]]; then
    [[ -d "$destination/.git" ]] || {
      echo "existing destination is not a Git checkout: $destination" >&2
      exit 1
    }
  else
    git clone --filter=blob:none --no-checkout "$repository" "$destination"
  fi

  git -C "$destination" fetch --depth=1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

clone_exact "$VLLM_REPOSITORY" "$VLLM_COMMIT" "$SOURCE_ROOT/vllm"
clone_exact "$GGUF_PLUGIN_REPOSITORY" "$GGUF_PLUGIN_COMMIT" \
  "$SOURCE_ROOT/vllm-gguf-plugin"

"$SCRIPT_DIR/apply_patches.sh" \
  "$SOURCE_ROOT/vllm" "$SOURCE_ROOT/vllm-gguf-plugin"

printf '%s\n' "$VLLM_COMMIT" >"$SOURCE_ROOT/VLLM_COMMIT"
printf '%s\n' "$GGUF_PLUGIN_COMMIT" >"$SOURCE_ROOT/GGUF_PLUGIN_COMMIT"

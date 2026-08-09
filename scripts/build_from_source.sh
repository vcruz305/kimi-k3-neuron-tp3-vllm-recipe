#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SOURCE_ROOT" >&2
  exit 2
fi

SOURCE_ROOT=$(cd -- "$1" && pwd)
VLLM_SOURCE="$SOURCE_ROOT/vllm"
PLUGIN_SOURCE="$SOURCE_ROOT/vllm-gguf-plugin"
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# shellcheck source=/dev/null
source "$RECIPE_ROOT/pins.env"

[[ -f "$SOURCE_ROOT/VLLM_COMMIT" ]] || {
  echo "Run scripts/prepare_sources.sh first." >&2
  exit 1
}

"$PYTHON_BIN" -m pip install -r "$VLLM_SOURCE/requirements/build/cuda.txt"
"$PYTHON_BIN" -m pip install -r "$VLLM_SOURCE/requirements/cuda.txt"

export VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-cuda}
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-4}

"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps "$VLLM_SOURCE"

# Do not let plugin dependency resolution silently replace torch/NCCL selected
# by the exact vLLM source build. Install the plugin's small runtime dependency
# explicitly at the recipe pin because vLLM itself does not provide it.
"$PYTHON_BIN" -m pip install "gguf==$GGUF_PYTHON_VERSION"
"$PYTHON_BIN" -m pip install --no-build-isolation --no-deps "$PLUGIN_SOURCE"

PYTHON_PREFIX=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
PIN_INSTALL_DIR=${K3_PIN_INSTALL_DIR:-$PYTHON_PREFIX/share/k3-neuron-vllm-pins}
install -d "$PIN_INSTALL_DIR"
install -m 0644 "$SOURCE_ROOT/VLLM_COMMIT" \
  "$PIN_INSTALL_DIR/VLLM_COMMIT"
install -m 0644 "$SOURCE_ROOT/GGUF_PLUGIN_COMMIT" \
  "$PIN_INSTALL_DIR/GGUF_PLUGIN_COMMIT"

echo "Pinned vLLM and GGUF plugin installed."

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

# CUDA architectures to build kernels for. Without this, the build only
# targets whatever the local GPU/toolchain defaults to, which is how this
# recipe ended up validated on sm_90 alone. Default covers three targets
# with two real compiles:
#
#   9.0    -> sm_90 (Hopper, e.g. H100/H200). This recipe's only
#             GPU-measured architecture; see evidence/DSPARK-TP3-H200.md.
#   12.0   -> sm_120 (Blackwell workstation, e.g. RTX PRO 6000). UNTESTED
#             by this recipe -- see README.md's compatibility matrix.
#   +PTX   -> also embeds compute_120 PTX alongside the native sm_120
#             cubin. sm_121 (NVIDIA DGX Spark, GB10) is the same Blackwell
#             major SM family as sm_120 (both major=12 -- vLLM's own
#             CMakeLists.txt lists 12.0 and 12.1 side by side in
#             CUDA_SUPPORTED_ARCHS). CUDA guarantees PTX compiled for one
#             capability JIT-compiles forward onto any device of equal or
#             greater capability (CUDA C++ Programming Guide, "PTX
#             Compatibility"), so the compute_120 PTX embedded here JITs
#             onto sm_121 the first time a kernel runs there. This is
#             confirmed necessary, not just convenient: an official
#             PyTorch cu130 wheel inspected while writing this recipe's
#             sm_121 support had native kernels through sm_120 and no
#             compute_* PTX at all, so stock wheels have no forward path
#             to sm_121 on their own (also tracked upstream as
#             https://github.com/vllm-project/vllm/issues/36821).
#
# Net effect: one build serves sm_90 natively, sm_120 natively, and sm_121
# via PTX JIT -- at the cost of a JIT-compile pause (can be several
# seconds per kernel) the first time each kernel runs on an sm_121 device.
# There is no native sm_121 cubin unless you also list "12.1" explicitly.
# sm_120 and sm_121 are both UNTESTED end-to-end by this recipe; run
# scripts/preflight_arch.py before trusting this in production.
#
# Blackwell (sm_120, sm_121, and datacenter sm_100) also needs
# VLLM_FLASH_ATTN_VERSION=2 at serve time: FlashAttention 3 (vLLM's
# default where available) has no Blackwell support in this vLLM pin,
# Hopper only. That variable is read by vllm's server at *serve* time
# (see scripts/serve_target.sh / scripts/serve_dspark.sh and
# scripts/preflight_arch.py's recommended_env output for the detected
# arch), not by this build script -- it is noted here because Blackwell
# builders need to know about it, and it is deliberately not exported
# with a forced default below, so sm_90 builds and serves are unaffected.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0;12.0+PTX"}

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

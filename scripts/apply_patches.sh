#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VLLM_SOURCE GGUF_PLUGIN_SOURCE" >&2
  exit 2
fi

VLLM_SOURCE=$(cd -- "$1" && pwd)
PLUGIN_SOURCE=$(cd -- "$2" && pwd)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

# shellcheck source=/dev/null
source "$RECIPE_ROOT/pins.env"

assert_clean_pin() {
  local source_dir=$1
  local expected=$2
  local actual
  actual=$(git -C "$source_dir" rev-parse HEAD)
  [[ "$actual" == "$expected" ]] || {
    echo "wrong commit in $source_dir: expected $expected, got $actual" >&2
    exit 1
  }
  [[ -z $(git -C "$source_dir" status --porcelain) ]] || {
    echo "source tree is not clean: $source_dir" >&2
    exit 1
  }
}

apply_one() {
  local source_dir=$1
  local patch_file=$2
  git -C "$source_dir" apply --check "$patch_file"
  git -C "$source_dir" apply "$patch_file"
}

(cd -- "$RECIPE_ROOT" && sha256sum -c SHA256SUMS)
assert_clean_pin "$PLUGIN_SOURCE" "$GGUF_PLUGIN_COMMIT"
assert_clean_pin "$VLLM_SOURCE" "$VLLM_COMMIT"

for patch_file in \
  "$RECIPE_ROOT/patches/gguf-plugin/0001-feat-add-Kimi-K3-Neuron-GGUF-adapter.patch" \
  "$RECIPE_ROOT/patches/gguf-plugin/0005-fix-kimi-k3-dequantize-latent-projections-for-native.patch" \
  "$RECIPE_ROOT/patches/gguf-plugin/0007-fix-moe-guard-CUDA-vector-grid-z-limit.patch" \
  "$RECIPE_ROOT/patches/gguf-plugin/0009-fix-honor-distributed-TP-rank-in-GGUF-fused-loaders.patch"
do
  apply_one "$PLUGIN_SOURCE" "$patch_file"
done

for patch_file in \
  "$RECIPE_ROOT/patches/vllm/0004-fix-kimi-k3-pad-vocabulary-for-odd-TP-sizes.patch" \
  "$RECIPE_ROOT/patches/vllm/0008-fix-kimi-k3-preserve-precision-sensitive-GGUF-weight.patch" \
  "$RECIPE_ROOT/patches/vllm/0010-kimi-k3-dspark-gguf-target-tp3.patch" \
  "$RECIPE_ROOT/patches/vllm/0012-dspark-draft-config-format-isolation.patch" \
  "$RECIPE_ROOT/patches/vllm/0013-kimi-linear-eagle3-target-bridge.patch" \
  "$RECIPE_ROOT/patches/vllm/0014-dflash-full-cg-with-piecewise-target.patch"
do
  apply_one "$VLLM_SOURCE" "$patch_file"
done

if [[ ${K3_APPLY_OPTIONAL_HOPPER_FLASHMLA:-0} == 1 ]]; then
  # Patch 0011 is a Hopper-only wrapper (adds non-causal support to
  # FlashMLA for the DSpark draft) and has only ever been validated on
  # sm_90 -- see README.md's compatibility matrix. Refuse to apply it
  # unless the visible GPU reports compute capability 9.0, so setting
  # K3_APPLY_OPTIONAL_HOPPER_FLASHMLA=1 on a Blackwell or Ampere box does
  # not silently carry an untested attention path into the source tree.
  #
  # This checks nvidia-smi directly rather than shelling out to
  # scripts/preflight_arch.py: this script runs before
  # build_from_source.sh, so torch/vLLM are not guaranteed to be
  # importable yet, and this also runs with no GPU at all inside
  # `docker build` (see the Dockerfile) and in CI's clean-apply job. Both
  # of those are legitimate reasons the arch cannot be verified here, not
  # proof the target is wrong -- hence the explicit override below rather
  # than a hard, unconditional refusal.
  if [[ ${K3_SKIP_HOPPER_ARCH_CHECK:-0} != 1 ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "K3_APPLY_OPTIONAL_HOPPER_FLASHMLA=1 but nvidia-smi is not on PATH, so the GPU architecture cannot be verified (for example, inside 'docker build', which has no GPU access). Refusing to apply patches/optional/0011-hopper-flashmla-noncausal-dspark.patch. If you are certain the build target is sm_90 (Hopper), set K3_SKIP_HOPPER_ARCH_CHECK=1 to apply it anyway." >&2
      exit 1
    fi
    hopper_check_caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | tr -d '\r' | sort -u | tr '\n' ',')
    hopper_check_caps=${hopper_check_caps%,}
    if [[ "$hopper_check_caps" != "9.0" ]]; then
      echo "K3_APPLY_OPTIONAL_HOPPER_FLASHMLA=1 but nvidia-smi reports compute capability '${hopper_check_caps:-none detected}', not the required 9.0 (sm_90 / Hopper). patches/optional/0011-hopper-flashmla-noncausal-dspark.patch is a Hopper-only wrapper and has never been validated elsewhere. Refusing to apply it. Run scripts/preflight_arch.py for the full picture, and set K3_SKIP_HOPPER_ARCH_CHECK=1 to override if you know what you are doing." >&2
      exit 1
    fi
    unset hopper_check_caps
  fi
  apply_one "$VLLM_SOURCE" \
    "$RECIPE_ROOT/patches/optional/0011-hopper-flashmla-noncausal-dspark.patch"
fi

git -C "$PLUGIN_SOURCE" diff --check
git -C "$VLLM_SOURCE" diff --check

python3 "$RECIPE_ROOT/tests/test_source_contract.py" "$VLLM_SOURCE"
python3 "$RECIPE_ROOT/tests/test_config_format_source_contract.py" "$VLLM_SOURCE"
python3 "$RECIPE_ROOT/tests/test_kimi_eagle3_source_contract.py" "$VLLM_SOURCE"
python3 "$RECIPE_ROOT/tests/validate_dflash_graph_patch.py" \
  "$VLLM_SOURCE/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
python3 "$RECIPE_ROOT/tests/verify_k3_fp32_contract.py" \
  "$VLLM_SOURCE/vllm/models/kimi_k3/nvidia/model.py"

if [[ ${K3_APPLY_OPTIONAL_HOPPER_FLASHMLA:-0} == 1 ]]; then
  python3 "$RECIPE_ROOT/tests/test_flashmla_source_contract.py" "$VLLM_SOURCE"
fi

while IFS= read -r file_name; do
  python3 -m py_compile "$PLUGIN_SOURCE/$file_name"
done < <(git -C "$PLUGIN_SOURCE" diff --name-only --diff-filter=AM -- '*.py')

while IFS= read -r file_name; do
  python3 -m py_compile "$VLLM_SOURCE/$file_name"
done < <(git -C "$VLLM_SOURCE" diff --name-only --diff-filter=AM -- '*.py')

echo "Mandatory overlay applied and source contracts passed."

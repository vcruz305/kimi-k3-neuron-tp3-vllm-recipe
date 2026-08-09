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

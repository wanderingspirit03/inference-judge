#!/usr/bin/env bash
# baseline-tpu -- the unoptimised reference point for a vllm-tpu target.
#
# Flags are the ones first brought up by hand on a v6e-4: Gemma 4 26B-A4B,
# weights and activations converted to float8_e4m3fn at load, KV cache fp8,
# tensor parallel over 4 chips.
#
# DO NOT TUNE THIS FILE.  It is the number every promotion is measured against;
# changing it silently invalidates the leaderboard.  New ideas go in a new
# experiments/<id>/ directory (copy this one, not TEMPLATE, on a TPU target).
#
# What differs from the GPU baselines:
#   * the image has two editable installs, /workspace/tpu_inference (the TPU
#     backend: kernels, runner, compile manager) and /workspace/vllm. run.sh
#     gives this launch private copies of your checkouts at
#     /src/tpu_inference and /src/vllm; step 1 lays them over the installs.
#     Nothing is re-installed and nothing is downloaded.
#   * XLA compiles every shape before it can run. The ruler mounts a private
#     copy of its compile cache at /root/.cache/vllm; a launch with the
#     incumbent's shapes and code reads it, anything else recompiles inside
#     the target's compile budget.
#   * the 8-bit conversion happens at load: the checkpoint on disk is BF16.
set -euo pipefail

: "${PORT:?run.sh sets PORT}"
: "${IJ_SERVED_NAME:?run.sh sets IJ_SERVED_NAME}"
: "${IJ_MODEL_DIR:?run.sh sets IJ_MODEL_DIR}"
echo "[launch] ${IJ_EXPERIMENT_ID:-unknown} on ${IJ_TARGET:-unknown} at $(date -u +%FT%TZ)"

# --- 1. lay the engine checkouts over the image's installs --------------------
# A merge, not a replace: a file you delete in the checkout is still in the
# image's copy, so empty a module rather than deleting it. .git stays behind.
overlay() {
  local src="$1" dst="$2"
  [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ] || { echo "[launch] no $src: image's own copy is used"; return 0; }
  [ -d "$dst" ] || { echo "[launch] $dst is not in this image" >&2; exit 1; }
  tar -C "$src" --exclude=.git -cf - . | tar -C "$dst" -xf -
  echo "[launch] $src laid over $dst"
}
overlay /src/tpu_inference /workspace/tpu_inference
overlay /src/vllm /workspace/vllm

# --- 2. environment -----------------------------------------------------------
# The target's engine.env arrives in the environment and wins; these are the
# values of the hand-measured run for anything it leaves unset.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-flax_nnx}"
export USE_BATCHED_RPA_KERNEL="${USE_BATCHED_RPA_KERNEL:-1}"
export MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-float8_e4m3fn}"
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-0}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

python -c 'import jax, vllm, tpu_inference; print("[launch] vllm", vllm.__version__, "jax", jax.__version__, "chips", len(jax.devices()))'

# --- 3. serve, in the foreground ----------------------------------------------
# 8-bit: every module's weights and activations go to float8_e4m3fn, except
# the MoE experts, which MOE_REQUANTIZE_WEIGHT_DTYPE converts.
QUANT='{"quantization": {"qwix": {"rules": [{"module_path": ".*experts.*"}, {"module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}'

ARGS=(
  "$IJ_MODEL_DIR"
  --served-model-name "$IJ_SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
  # 2 chips fail at load: the loader puts the BF16 weights on the chips first.
  --tensor-parallel-size 4
  # Inputs go up to 8192 tokens with 512 out (the target's size list). The
  # hand-measured run used 4096 for both; the first ruler baseline replaces
  # its numbers.
  --max-model-len 9216
  --max-num-batched-tokens 8192
  # NOT tuned for one stream on purpose: this file is the plain starting point.
  --max-num-seqs 8
  --gpu-memory-utilization 0.6
  --kv-cache-dtype fp8
  --async-scheduling
  --disable-chunked-mm-input
  --additional-config "$QUANT"
)

echo "[launch] vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"

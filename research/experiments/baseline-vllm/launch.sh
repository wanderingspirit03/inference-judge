#!/usr/bin/env bash
# baseline-vllm -- the unoptimised reference point for a vLLM target.
#
# A plain launcher: the model directory, the served name and the port come from
# the environment run.sh provides for the target, and vLLM's defaults do the
# rest.  A model that needs its own flags (context length, parsers, memory
# split) gets them in the target's baseline, not here.
#
# UNPROVEN ON REAL HARDWARE in this form: it has not been through the ruler on
# a GPU yet.
#
# DO NOT TUNE THIS FILE.  It is the number every promotion is measured against;
# changing it silently invalidates the leaderboard.  New ideas go in a new
# experiments/<id>/ directory.
set -euo pipefail

: "${PORT:?run.sh sets PORT}"
: "${IJ_SERVED_NAME:?run.sh sets IJ_SERVED_NAME}"
: "${IJ_MODEL_DIR:?run.sh sets IJ_MODEL_DIR}"
echo "[launch] ${IJ_EXPERIMENT_ID:-unknown} on ${IJ_TARGET:-unknown} at $(date -u +%FT%TZ)"
# The image built by setup/Dockerfile.base has `python`; an upstream engine image
# may have only `python3`.
PY=python; command -v python >/dev/null 2>&1 || PY=python3
"$PY" -c 'import vllm,torch;print("[launch] vllm",vllm.__version__,"torch",torch.__version__,"cuda",torch.version.cuda)'

ARGS=(
  "$IJ_MODEL_DIR"
  --served-model-name "$IJ_SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
)

echo "[launch] vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"

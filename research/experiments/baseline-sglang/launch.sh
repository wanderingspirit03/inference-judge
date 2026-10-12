#!/usr/bin/env bash
# baseline-sglang -- the unoptimised reference point for an SGLang target.
#
# A plain launcher: the model directory, the served name and the port come from
# the environment run.sh provides for the target, and SGLang's defaults do the
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
# inference-judge-base keeps SGLang in its own venv: its torch/flashinfer pins conflict
# with vLLM's and a shared environment is not reproducible.
PY=${SGLANG_PYTHON:-/opt/venv/sglang/bin/python}

[ -x "$PY" ] || { echo "[launch] no SGLang venv at $PY" >&2; exit 1; }
[ -f /opt/venv/sglang/.install-failed ] && {
  echo "[launch] the image's SGLang install failed at build time:" >&2
  cat /opt/venv/sglang/.install-failed >&2; exit 1; }

echo "[launch] ${IJ_EXPERIMENT_ID:-unknown} on ${IJ_TARGET:-unknown} at $(date -u +%FT%TZ)"
"$PY" -c 'import sglang,torch;print("[launch] sglang",sglang.__version__,"torch",torch.__version__,"cuda",torch.version.cuda)'

ARGS=(
  --model-path "$IJ_MODEL_DIR"
  --served-model-name "$IJ_SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
)

echo "[launch] sglang.launch_server ${ARGS[*]}"
exec "$PY" -m sglang.launch_server "${ARGS[@]}"

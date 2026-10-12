#!/usr/bin/env bash
# =============================================================================
# experiments/<id>/launch.sh -- the ONLY interface between you and the ruler.
#
# Copy this directory to experiments/<your-id>/, edit, then queue it:
#     queue/<your-id>.json = {"id": "<your-id>", "kind": "candidate"}    # scored
#     queue/<your-id>.json = {"id": "<your-id>", "kind": "instrument"}   # profile, unscored
# The driver's runner calls the ruler. You never call run.sh yourself.
#
# <your-id> must match ^[a-z0-9][a-z0-9-]{2,40}$ and must not already exist in
# /opt/bench/results/.  Name it after the hypothesis, not after a number:
# `async-sched-on`, `maxnumseqs-64`, `triton-moe-fused-silu`.
# =============================================================================
#
# THE TARGET
# ----------
# Model, accelerator, engine image, port, metric and budgets all come from the
# target file the operator chose (/opt/bench/TARGET names it; the file is
# /opt/bench/targets/<id>.yaml). Read it; do not hard-code what it says. This
# script sees the target only through the environment below.
#
# WHAT run.sh DOES AROUND THIS SCRIPT
# -----------------------------------
#   1. snapshots experiments/<id>/ and mounts it READ-ONLY at /ro/exp; your
#      engine trees likewise at /ro/src/<name>, one for each name in the
#      target's engine.sources (~/<name>-src; a target that lists none
#      has no /ro/src)
#   2. builds <expdir>/Dockerfile if you wrote one, else uses the target's
#      engine image; if you wrote deps.txt it is baked into a derived image
#      BEFORE the launch (bounded build, log in results/<id>/deps.log; a failed
#      install is BUILD_FAILED and this script is never reached)
#   3. starts the container SEALED: --network none, every mount read-only, no
#      /out (except --instrument), accelerator flags from the target only.
#      The entry point copies /ro/exp to /exp and /ro/src/<name> to
#      /src/<name>, writable and private to this launch, then runs
#      `bash /exp/launch.sh`. If the target has engine.compile_cache, its
#      mount point is the one writable mount: a copy of the ruler's cache
#      that your launches of one run share and that is thrown away after the
#      run (only what an incumbent launch compiled is ever added to the
#      ruler's cache)
#   4. polls GET /v1/models on $PORT until 200, within the target's
#      health.ready_timeout_s (compile_s + health_s when it names none)
#      -> else BUILD_FAILED; a server that exits is given up on at once
#   5. measures from a sidecar in the container's network namespace. A tps
#      target launches you several times with fresh nonce-prefixed prompts
#      every run: with metric.screen_window_s first alone for one short
#      window (a result more than metric.screen_tol under the incumbent's
#      stored value ends the run there),
#      then ABBA against the incumbent, which runs from the experiment AND
#      the engine trees it was promoted with. Every launch starts with an
#      unscored warm pass over each input size; quality is gated against the
#      fixed reference R on each of your ABBA launches. With a list of input
#      sizes the score is whole-answer speed (request sent to last token) and
#      a loss at any one size blocks the promotion
#   6. kills the container; the target's run_s is the hard timeout -> TIMEOUT
#
# YOUR CONTRACT
# -------------
#   * serve OpenAI-compatible HTTP on 0.0.0.0:$PORT, model name exactly
#     $IJ_SERVED_NAME, weights from $IJ_MODEL_DIR (read-only)
#   * GET /v1/models -> 200 once you are ready (the readiness signal)
#   * /v1/completions with echo + logprobs (the quality gate teacher-forces),
#     /v1/chat/completions with stream=true and `usage` in the final chunk
#   * run in the FOREGROUND: the last line must be `exec <server>`, so the
#     container dies when the server dies
#   * nothing can be downloaded (no network) and nothing written leaves the
#     container in a scored run. Profiles go to /out under --instrument only
#
# ENVIRONMENT
# -----------
#   /exp, /src/vllm, /src/sglang   writable copies, discarded after the launch
#   $IJ_MODEL_DIR    read-only weights (empty for the cpu-mock target)
#   $IJ_SERVED_NAME  the model name the ruler asks for
#   $IJ_TARGET       the target id, e.g. cpu-mock
#   $IJ_EXPERIMENT_ID, $PORT, plus the target's engine.env
#   /out                only under --instrument (results/<id>/artifacts)
#
# OPTIONAL FILES NEXT TO THIS SCRIPT
# ----------------------------------
#   Dockerfile   replaces the target's engine image (build log: build.log)
#   deps.txt     pip requirements, baked into the image before the launch
#                (into /opt/venv/vllm when the image has it, else system pip).
#                Wheels must be reachable at BUILD time: the launch has no
#                network.
#   patches/*.diff  run.sh does NOT apply these; apply them yourself, below
#   kernels/     your Triton/CUDA source, imported by your patched engine
#
# BUDGET
# ------
# compile_s + health_s from the target to become healthy. A python-level
# patch re-installed with --no-deps takes seconds; a CUDA rebuild belongs in a
# Dockerfile, where it happens before the clock starts.
# =============================================================================
set -euo pipefail

: "${PORT:?run.sh sets PORT}"
: "${IJ_SERVED_NAME:?run.sh sets IJ_SERVED_NAME}"
MODEL_DIR="${IJ_MODEL_DIR:-}"
echo "[launch] ${IJ_EXPERIMENT_ID:-unknown} on ${IJ_TARGET:-unknown} at $(date -u +%FT%TZ)"

# --- 1. optional: apply your patches to the engine source ---------------------
# /src is this launch's own copy; the host checkout is untouched.
#
#   for p in /exp/patches/*.diff; do
#     [ -e "$p" ] || continue
#     git -C /src/vllm apply -v "$p"
#   done
#
# Re-install editable, offline, and without letting the resolver touch torch:
#   VLLM_PRECOMPILED_WHEEL_LOCATION="$(cat /opt/vllm-wheel-url)" \
#     uv pip install --offline --python /opt/venv/vllm/bin/python -e /src/vllm --no-deps
# VLLM_PRECOMPILED_WHEEL_LOCATION must name a wheel already in the image: with
# no network a URL cannot be fetched. Prebuild in a Dockerfile otherwise.

# --- 2. optional: your kernels ------------------------------------------------
# Triton compiles lazily on first use, inside the health budget.
#   export TRITON_CACHE_DIR=/tmp/triton
#   export PYTHONPATH=/exp/kernels:${PYTHONPATH:-}

# --- 3. optional: profiling (--instrument runs only) --------------------------
# /out exists only under --instrument; a scored run has nowhere to write.
#   if [ -d /out ] && [ -w /out ]; then
#     exec nsys profile --delay 60 --duration 30 --kill none -o /out/profile \
#          --trace cuda,nvtx,osrt --force-overwrite true vllm serve ...
#   fi

# --- 4. required: serve, in the foreground ------------------------------------
# Change these flags. This copy is the plain default: measure your idea against
# the incumbent, not against a second baseline. A model that needs its own
# flags gets them from the target's baseline: start from that one.
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$IJ_SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT"

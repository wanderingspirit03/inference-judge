#!/usr/bin/env bash
# profile-baseline -- baseline-vllm, under nsys, so hour 0 has a profile.
#
#     queue/profile-baseline.json = {"id": "profile-baseline", "kind": "instrument"}
#
# CLAUDE.md tells the researcher to read "the latest
# profile", but nothing in bootstrap produces one -- the baselines serve, get
# measured, and are torn down.  This experiment exists so that getting the first
# profile costs one command instead of an hour of design.
#
# WHAT TO EXPECT
# --------------
# The verdict is INSTRUMENT: an instrument run is never scored.  nsys
# instrumentation costs a few percent, so this configuration is slower than
# the baseline it copies.  DO NOT put its numbers on the leaderboard and do
# not compare anything against it.  The artefacts you want are what
# run.sh collects out of /out:
#   /opt/bench/results/profile-baseline/artifacts/profile.nsys-rep      the trace
#   /opt/bench/results/profile-baseline/artifacts/profile-summary.txt   the tables
# Read the summary; `nsys stats` itself is not in the researcher's allowed
# commands, so this script runs it inside the container once the report exists.
#
# WHAT IS PROFILED
# ----------------
# The capture starts DELAY_S seconds AFTER the server first answers
# GET /v1/models (default 60 s: inside the ruler's traffic, never
# during weight load, torch.compile or CUDA-graph capture, which together take
# 3-4 minutes on this box) and lasts DURATION_S seconds (default 30: a few
# hundred MB).  A full-run trace is gigabytes and the write itself perturbs the
# numbers you came to look at.
#
# HOW
# ---------------------------------------------------------------------------
# The first version ran `exec nsys profile --delay 150 --duration 30 --kill none
# vllm serve`.  Two things were wrong with it on the real box: 150 s landed in
# the middle of CUDA-graph capture, and nsys was the container's PID 1, so when
# the collection window closed nsys exited (--kill none only spares the CHILD)
# and docker tore the container down: BUILD_FAILED at 208 s with a trace of the
# graph capture.  Now this shell stays PID 1, the server runs under
# `nsys launch` (instrumented, not collecting), and a helper turns collection
# on and off with `nsys start` / `nsys stop` once the server is healthy.
set -euo pipefail

: "${PORT:?run.sh sets PORT}"
: "${IJ_SERVED_NAME:?run.sh sets IJ_SERVED_NAME}"
: "${IJ_MODEL_DIR:?run.sh sets IJ_MODEL_DIR}"
DELAY_S=${IJ_PROFILE_DELAY_S:-60}     # after the server is healthy
DURATION_S=${IJ_PROFILE_DURATION_S:-30}

echo "[launch] $(date -u +%FT%TZ) profile-baseline, model $IJ_MODEL_DIR"

# The same flags as baseline-vllm, so this stays a copy of it and not a
# second, silently different baseline.
ARGS=(
  "$IJ_MODEL_DIR"
  --served-model-name "$IJ_SERVED_NAME"
  --host 0.0.0.0
  --port "$PORT"
)

healthy() {
  if command -v curl >/dev/null; then
    curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null
  else
    python3 - "$PORT" <<'PY'
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=3)
PY
  fi
}

if ! command -v nsys >/dev/null; then
  echo "[launch] WARNING: nsys is not in this image (see /opt/NSYS_MISSING)." >&2
  echo "[launch] Falling back to a coarse capture: nvidia-smi dmon into /out." >&2
  ( until healthy 2>/dev/null; do sleep 5; done; sleep "$DELAY_S"
    nvidia-smi dmon -c "$DURATION_S" > /out/gpu-dmon.txt 2>&1 || true ) &
  exec vllm serve "${ARGS[@]}"
fi

SESSION=inference-judge-profile
echo "[launch] nsys launch (instrumented, collection off) -> session $SESSION"
nsys launch --session-new="$SESSION" --trace cuda,nvtx,osrt vllm serve "${ARGS[@]}" &
NSYS_PID=$!

server_alive() {
  kill -0 "$NSYS_PID" 2>/dev/null || pgrep -f "vllm serve" >/dev/null 2>&1
}

# Collection helper: wait for health, wait DELAY_S more so the capture sits in
# real traffic, collect DURATION_S, then write the summary once the .nsys-rep
# is finalised (nsys stats on a half-written report fails).
(
  until healthy 2>/dev/null; do
    server_alive || { echo "[launch] server died before becoming healthy; no profile" >&2; exit 0; }
    sleep 5
  done
  echo "[launch] $(date -u +%FT%TZ) server healthy; capture in ${DELAY_S}s for ${DURATION_S}s"
  sleep "$DELAY_S"
  if nsys start --session="$SESSION" -o /out/profile --force-overwrite true \
       --sample none --cpuctxsw none; then
    sleep "$DURATION_S"
    nsys stop --session="$SESSION" || echo "[launch] nsys stop failed" >&2
  else
    echo "[launch] nsys start failed; the server keeps serving without a profile" >&2
    exit 0
  fi
  deadline=$(( SECONDS + 600 ))
  while [ "$SECONDS" -lt "$deadline" ]; do
    for r in /out/profile*.nsys-rep; do
      [ -e "$r" ] || continue
      # Stable size = finalised. Two reads 10 s apart is enough.
      s1=$(stat -c %s "$r" 2>/dev/null || echo 0)
      sleep 10
      s2=$(stat -c %s "$r" 2>/dev/null || echo 0)
      if [ "$s1" = "$s2" ] && [ "$s1" != 0 ]; then
        echo "[launch] nsys stats on $r"
        nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
          --format table "$r" > /out/profile-summary.txt 2>&1 \
          || echo "[launch] nsys stats failed; the .nsys-rep is still in /out" >&2
        exit 0
      fi
    done
    sleep 10
  done
  echo "[launch] no .nsys-rep appeared in /out; no summary written" >&2
) &

# This shell is PID 1.  run.sh stops the container with SIGTERM: pass it on to
# the server, then leave when the server is gone (whether nsys launch stayed in
# the foreground of its child or handed it off).
on_term() { pkill -TERM -f "vllm serve" 2>/dev/null || true; kill -TERM "$NSYS_PID" 2>/dev/null || true; }
trap on_term TERM INT
while server_alive; do sleep 5; done
echo "[launch] server exited"

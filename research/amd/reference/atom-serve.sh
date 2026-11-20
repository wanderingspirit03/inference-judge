#!/bin/bash
# The reference engine: ROCm ATOM (image rocm/atom-dev, pinned by digest: the 2026-11-16 latest) serving
# Qwen3.8-27B in bf16 on one GPU, no speculative decoding.
#   atom-serve.sh start    start the container (returns at once; poll /v1/models)
#   atom-serve.sh stop     stop and remove it
#   atom-serve.sh status   print whether it runs
# Settings come from the environment, with these defaults:
#   IJ_REF_PORT=8001  IJ_REF_MODEL_DIR=/srv/models/Qwen3.8-27B  IJ_REF_NAME=qwen3.8-27b
#   IJ_REF_IMAGE=ij-atom-ref:ad256a94-lp1 (the pinned digest rocm/atom-dev@sha256:ad256a94... plus
#   atom_token_logprobs.py; build-ref-image.sh makes it)  IJ_REF_MEM=0.5  IJ_REF_MAX_LEN=16384
#   IJ_REF_CONTAINER=ij-ref  IJ_REF_EXTRA="" (more server flags, split on spaces)
# The container carries the label ij.gpu.job=$IJ_GPU_JOB, so the GPU queue
# removes it when the job that started it is gone.
#
# A caller that cannot use Docker directly (builder and cand on a box set up
# with the Docker launcher) goes through controller/box/ij-docker's fixed
# reference profile instead: the image, the model and the flags come from the
# box's config, the caller picks only the name (ij-ref or ij-engine), the port
# and IJ_REF_MEM.  IJ_REF_EXTRA is refused there.
set -euo pipefail
NAME="${IJ_REF_CONTAINER:-ij-ref}"
PORT="${IJ_REF_PORT:-8001}"
MODEL_DIR="${IJ_REF_MODEL_DIR:-/srv/models/Qwen3.8-27B}"
SERVED="${IJ_REF_NAME:-qwen3.8-27b}"
IMAGE="${IJ_REF_IMAGE:-ij-atom-ref:ad256a94-lp1}"
MEM="${IJ_REF_MEM:-0.5}"
MAX_LEN="${IJ_REF_MAX_LEN:-16384}"
LAUNCHER="${IJ_DOCKER_LAUNCHER:-/opt/ij-amd/amd/controller/box/ij-docker}"
if [ ! -w /var/run/docker.sock ] && [ -x "$LAUNCHER" ]; then
  case "$NAME" in
    ij-ref) SHORT=ref ;;
    ij-engine) SHORT=engine ;;
    *) echo "atom-serve.sh: through the launcher the container is ij-ref or ij-engine, not $NAME" >&2; exit 2 ;;
  esac
  case "${1:-}" in
    start)
      if [ -n "${IJ_REF_EXTRA:-}" ]; then
        echo "atom-serve.sh: IJ_REF_EXTRA is not allowed through the launcher (its reference profile is fixed)" >&2; exit 2
      fi
      exec sudo -n "$LAUNCHER" ref start "$SHORT" "$PORT" "$MEM" ;;
    stop|status) exec sudo -n "$LAUNCHER" ref "$1" "$SHORT" ;;
    logs) exec sudo -n "$LAUNCHER" ref logs "$SHORT" "${2:-40}" ;;
    *) echo "usage: $0 start|stop|status|logs [n]" >&2; exit 2 ;;
  esac
fi
case "${1:-}" in
  start)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    RENDER_GID=$(getent group render | cut -d: -f3)
    VIDEO_GID=$(getent group video | cut -d: -f3)
    mkdir -p /var/cache/ij-ref
    # shellcheck disable=SC2086
    docker run -d --name "$NAME" --label "ij.gpu.job=${IJ_GPU_JOB:-none}" --network host \
      --device=/dev/kfd --device=/dev/dri --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
      --ipc=host --shm-size 16g \
      -e GPU_MAX_HW_QUEUES=1 -e AITER_LOG_LEVEL=WARNING \
      -v "$(dirname "$MODEL_DIR"):/models:ro" -v /var/cache/ij-ref:/root/.cache \
      "$IMAGE" python3 -m atom.entrypoints.openai_server \
      --model "/models/$(basename "$MODEL_DIR")" --served-model-name "$SERVED" \
      --kv_cache_dtype bf16 -tp 1 --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$MEM" --server-port "$PORT" ${IJ_REF_EXTRA:-} >/dev/null
    echo "started $NAME on port $PORT"
    ;;
  stop)
    if docker rm -f "$NAME" >/dev/null 2>&1; then echo "stopped $NAME"; else echo "$NAME was not running"; fi
    ;;
  status)
    docker ps --filter "name=^${NAME}\$" --format '{{.Names}} {{.Status}}'
    ;;
  logs)
    docker logs --tail "${2:-40}" "$NAME" 2>&1
    ;;
  *) echo "usage: $0 start|stop|status|logs [n]" >&2; exit 2 ;;
esac

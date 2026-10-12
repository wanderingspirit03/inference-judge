#!/usr/bin/env bash
# The local box, end to end, in one command (no cloud, no keys, no real claude):
#
#     bash setup/local_box/run_local_box.sh [--keep] [--clean] [--out DIR]
#
#   1. builds inference-judge-localbox:dev (ubuntu:24.04 + dockerd; setup/local_box/Dockerfile)
#   2. starts it --privileged with the repo mounted read-only at /src and the
#      box's own /var/lib/docker on the named volume inference-judge-localbox-docker
#   3. runs setup/local_box/scenario.py inside it: bootstrap for
#      targets/cpu-mock.yaml, then the driver, runner and heartbeat under a
#      systemd stand-in, with a stub `claude` playing the researcher
#   4. copies the box's logs to --out (default /tmp/inference-judge-localbox-<utc>) and
#      removes the container
#
#   --keep    leave the container running for inspection (docker exec -it <name> bash)
#   --clean   also remove the image and the docker volume afterwards (frees ~2 GB;
#             the next run rebuilds the image and re-pulls python:3.12-slim)
#
# Exit 0 iff every check passed.  Why Docker-in-Docker and not the host's socket:
# bench/run.sh bind-mounts box paths (/opt/bench, /home/researcher/...) into the
# engine container and talks to it on the box's 127.0.0.1.  With the host
# socket those paths would resolve on the Docker Desktop VM and the ports on
# the wrong loopback; with the box's own dockerd both are the box's, as on a
# rented machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMAGE="${LOCALBOX_IMAGE:-inference-judge-localbox:dev}"
VOLUME="${LOCALBOX_VOLUME:-inference-judge-localbox-docker}"
NAME="inference-judge-localbox-$$"
KEEP=0 CLEAN=0
OUT="/tmp/inference-judge-localbox-$(date -u +%Y%m%dT%H%M%SZ)"

while [ $# -gt 0 ]; do
  case "$1" in
    --keep) KEEP=1 ;;
    --clean) CLEAN=1 ;;
    --out) OUT="${2:?--out needs a directory}"; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

t0=$(date +%s)
docker info >/dev/null 2>&1 || { echo "docker is not running" >&2; exit 70; }
echo "== building $IMAGE"
docker build -q -t "$IMAGE" "$HERE" >/dev/null

# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  local rc=$?
  mkdir -p "$OUT"
  docker cp "$NAME:/var/log/localbox/." "$OUT/" >/dev/null 2>&1 || true
  docker cp "$NAME:/var/log/inference-judge/." "$OUT/inference-judge/" >/dev/null 2>&1 || true
  docker cp "$NAME:/home/driver/research-state/." "$OUT/research-state/" >/dev/null 2>&1 || true
  if [ "$KEEP" = 1 ]; then
    echo "container kept: docker exec -it $NAME bash   (remove: docker rm -f $NAME)"
  else
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  if [ "$CLEAN" = 1 ] && [ "$KEEP" != 1 ]; then
    docker volume rm "$VOLUME" >/dev/null 2>&1 || true
    docker image rm "$IMAGE" >/dev/null 2>&1 || true
    echo "removed $IMAGE and volume $VOLUME"
  fi
  echo "logs: $OUT   wall time: $(( $(date +%s) - t0 ))s"
  exit "$rc"
}
trap cleanup EXIT

echo "== starting $NAME"
docker run -d --privileged --name "$NAME" --hostname inference-judge-localbox \
  -v "$VOLUME:/var/lib/docker" -v "$REPO:/src:ro" "$IMAGE" >/dev/null
for _ in $(seq 1 60); do
  docker logs "$NAME" 2>&1 | grep -q 'box_init: ready' && break
  sleep 1
done
docker logs "$NAME" 2>&1 | grep -q 'box_init: ready' || { docker logs "$NAME" >&2; exit 70; }

echo "== scenario"
set +e
docker exec ${LOCALBOX_TIER:+-e LOCALBOX_TIER="$LOCALBOX_TIER"} "$NAME" python3 -I /src/setup/local_box/scenario.py
rc=$?
set -e
exit "$rc"

#!/usr/bin/env bash
# tools/tpu_seal_probe.sh -- which container seal do the TPU chips work under?
#
# Run on a TPU VM, as a user who can run docker, in the first minutes of a
# paid session and before anything is downloaded:
#     tools/tpu_seal_probe.sh [IMAGE] [CHIPS]
#
# It starts the engine image once per flag set, strongest seal first, and in
# each one runs only `import jax; len(jax.devices())`. The first set that sees
# all the chips is the one to put in the target's engine.docker_flags and
# engine.env. A node where even the last set (what worked by hand on
# 2026-10-11) fails is a faulty node: delete it and create a new one.
#
# What is known going in (2026-10-11): the chips are /dev/vfio/{vfio,0..3};
# libtpu reaches a control service on the host, over TCP by default
# (port 8353 on the host, so it needs the host network) or over the Unix socket
# /var/run/tpu-plugin/vbar.sock when VBAR_CONTROL_SERVICE_URL names it; a
# socket on a read-only bind mount still accepts connections. Nothing below
# has been tried on a TPU yet.
set -uo pipefail

IMAGE="${1:-vllm/vllm-tpu@sha256:8a56fb9620c878ca4d5a5654c4fcddfc65d263c1ecdb6c374f41f414cd07fc48}"
CHIPS="${2:-4}"
PER_TRY_S="${PER_TRY_S:-240}"          # a faulty node hangs for 60 s per chip session
SOCK_DIR=/var/run/tpu-plugin
VBAR="VBAR_CONTROL_SERVICE_URL=unix:$SOCK_DIR/vbar.sock"

devices=()
for d in /dev/vfio/vfio /dev/vfio/[0-9]*; do
  [ -e "$d" ] && devices+=(--device "$d")
done
echo "host: $(ls /dev/vfio 2>/dev/null | tr '\n' ' ')| socket: $(ls "$SOCK_DIR" 2>/dev/null | tr '\n' ' ')"
[ "${#devices[@]}" -gt 0 ] || echo "WARNING: no /dev/vfio devices on this host; only the privileged sets can work"

check="import jax; n = len(jax.devices()); print('CHIPS', n); raise SystemExit(0 if n == $CHIPS else 1)"

# name | seal it gives | docker flags
sets=(
  "sealed|no network, no privilege, devices and a read-only socket|--network none ${devices[*]} -v $SOCK_DIR:$SOCK_DIR:ro -e $VBAR"
  "sealed-memlock|as sealed, plus unlimited locked memory (vfio pins pages)|--network none ${devices[*]} -v $SOCK_DIR:$SOCK_DIR:ro -e $VBAR --ulimit memlock=-1:-1 --cap-add IPC_LOCK"
  "sealed-rawio|as sealed-memlock, plus SYS_RAWIO and SYS_ADMIN|--network none ${devices[*]} -v $SOCK_DIR:$SOCK_DIR:ro -e $VBAR --ulimit memlock=-1:-1 --cap-add IPC_LOCK --cap-add SYS_RAWIO --cap-add SYS_ADMIN"
  "privileged-nonet|privileged, but no network|--network none --privileged -v $SOCK_DIR:$SOCK_DIR:ro -e $VBAR"
  "privileged-host|none: privileged and host network (worked by hand)|--network host --privileged"
)

winner=""
for s in "${sets[@]}"; do
  IFS='|' read -r name seal flags <<<"$s"
  echo
  echo "== $name: $seal"
  echo "   docker run --rm $flags --shm-size 16g IMAGE python -c '<chip count>'"
  start=$(date +%s)
  # shellcheck disable=SC2086  # the flag sets are word lists on purpose
  out="$(timeout --signal=KILL "$PER_TRY_S" docker run --rm $flags --shm-size 16g \
           "$IMAGE" python -c "$check" 2>&1)"
  rc=$?
  took=$(( $(date +%s) - start ))
  if [ "$rc" -eq 0 ]; then
    echo "   PASS in ${took}s: $(printf '%s\n' "$out" | grep '^CHIPS' | tail -n 1)"
    [ -n "$winner" ] || winner="$name|$flags"
    # the strongest working set is the answer; the weaker ones add nothing
    break
  fi
  echo "   FAIL in ${took}s (exit $rc). Last lines:"
  printf '%s\n' "$out" | tail -n 6 | sed 's/^/     /'
done

echo
if [ -z "$winner" ]; then
  echo "RESULT: no flag set sees $CHIPS chips. This node is faulty: delete it and create a new one."
  exit 1
fi
echo "RESULT: strongest working seal is '${winner%%|*}'"
echo "  engine.docker_flags: ${winner#*|}"
case "${winner%%|*}" in
  privileged-*) echo "  NOTE: a privileged container can reach host disks. The ruler will record the seal as not proven." ;;
esac

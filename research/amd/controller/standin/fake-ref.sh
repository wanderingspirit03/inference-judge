#!/bin/bash
# fake-ref.sh start|stop|status|logs: the stand-in's reference engine, in the
# shape of reference/atom-serve.sh (same commands, same environment), as a
# process instead of a Docker container.
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PORT="${IJ_REF_PORT:-8001}"
STATE="${IJ_AMD_STATE:-/tmp}"
PIDF="$STATE/fake-ref.pid"
case "${1:-}" in
  start)
    "$0" stop >/dev/null 2>&1
    nohup python3 "$HERE/fake_ref.py" serve "$PORT" "${IJ_REF_NAME:-fake}" >"$STATE/fake-ref.log" 2>&1 &
    echo $! > "$PIDF"; echo "started fake reference on port $PORT" ;;
  stop)
    if [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null; then echo "stopped fake reference"; else echo "fake reference was not running"; fi
    rm -f "$PIDF" ;;
  status)
    [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && echo "fake-ref pid $(cat "$PIDF")" ;;
  logs) tail -n "${2:-30}" "$STATE/fake-ref.log" ;;
  *) echo "usage: fake-ref.sh start|stop|status|logs" >&2; exit 2 ;;
esac

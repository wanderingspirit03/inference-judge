#!/usr/bin/env bash
# One heartbeat (systemd: inference-judge-heartbeat.timer, every minute).  Writes
# $IJ_STATE_DIR/heartbeat.json (driver alive?, runner state, accelerator
# hours used, end of run, disk use, last verdict), charges accelerator time
# while the driver is down, and ALERTs once if the driver looks wedged.
set -euo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${IJ_PY:-python3}" -I "$LOOP_DIR/loopstate.py" heartbeat "$@"

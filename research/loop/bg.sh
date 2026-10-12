#!/usr/bin/env bash
# Start a long CPU job that outlives this agent session, and register it so
# the driver can account for it and reap it when the budget ends.
#
#     loop/bg.sh <name> <command> [args...]
#
# The job runs in its own session and process group, with stdin closed and
# output in $IJ_JOBS_SPOOL/<name>.<pid>.log (default ~/.inference-judge-jobs).  A
# record <pid>.json goes in the same spool; the driver checks it is really
# yours, a group leader and alive before it trusts it, and kills the whole
# group at the end of the run.  Jobs started any other way are found by the
# driver anyway in the two-user layout, but with no name and no log.
#
# Prints: "<pid> <log path>".  This is for CPU work only (reading, offline
# analysis, builds); the accelerators belong to the bench.
set -euo pipefail

[ $# -ge 2 ] || { echo "usage: loop/bg.sh <name> <command> [args...]" >&2; exit 2; }
NAME="$1"; shift
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { echo "bg.sh: name must match [A-Za-z0-9._-]{1,64}" >&2; exit 2; }
SPOOL="${IJ_JOBS_SPOOL:-$HOME/.inference-judge-jobs}"
mkdir -p "$SPOOL"

exec "${IJ_PY:-python3}" -I - "$SPOOL" "$NAME" "$@" <<'PY'
import json, os, shlex, subprocess, sys, time
spool, name, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
log_path = os.path.join(spool, f"{name}.{os.getpid()}.log")
log = open(log_path, "ab")
p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, close_fds=True)
final = os.path.join(spool, f"{name}.{p.pid}.log")
os.replace(log_path, final)
rec = {"pid": p.pid, "pgid": p.pid, "name": name, "cmd": shlex.join(argv)[:500],
       "start": time.time(), "log": final}
tmp = os.path.join(spool, f".{p.pid}.json.tmp")
with open(tmp, "w") as f:
    json.dump(rec, f)
os.replace(tmp, os.path.join(spool, f"{p.pid}.json"))
print(p.pid, final)
PY

#!/usr/bin/env bash
# Tier-1 stand-in for /opt/bench/run.sh in the local box (installed there, owned
# by bench, mode 0750, AFTER bootstrap).  It is a fake ruler: the verdict comes
# from the experiment's launch.sh (--behaviour fast -> PROMOTED, junk -> INVALID,
# setup/rehearse.sh's failing launch -> BUILD_FAILED and its wedged server ->
# TIMEOUT, anything else -> REJECTED), not from a measurement.  What it does exercise is
# everything around the ruler: the driver's sudoers route, the id check, bench
# reading the researcher's experiment through /home/researcher, a results dir the
# researcher and the driver can read, and a schema-2 results.json with a run_id.
#
# scenario.py plants a stub incumbent (leaderboard/cpu-mock.json) beside it:
# setup/rehearse.sh refuses a tps target without one.  Tier 2 keeps the real
# bench/run.sh and makes the incumbent with a real baseline run.
set -euo pipefail
ID_RE='^[a-z0-9][a-z0-9-]{2,40}$'
RESULTS=/opt/bench/results
kind=candidate
if [ "${1:-}" = --instrument ]; then kind=instrument; shift; fi
[ $# -eq 1 ] && [[ "$1" =~ $ID_RE ]] || { echo "run.sh: bad usage" >&2; exit 2; }
id="$1"
launch="/home/researcher/research/experiments/$id/launch.sh"
[ -r "$launch" ] || { echo "run.sh: bench cannot read $launch" >&2; exit 70; }
[ -e "$RESULTS/$id" ] && { echo "run.sh: $RESULTS/$id exists" >&2; exit 2; }
t0=$(date +%s.%N)
if grep -q 'localbox-slow' "$launch"; then sleep "${LOCALBOX_SLOW_S:-25}"; else sleep 3; fi
case "$(cat "$launch")" in
  *"--behaviour fast"*) verdict=PROMOTED delta=41.0 ;;
  *"--behaviour junk"*) verdict=INVALID delta=0 ;;
  *"rehearsal: failing on purpose"*) verdict=BUILD_FAILED delta=0 ;;
  *"never answers a request"*)       verdict=TIMEOUT delta=0 ;;
  *)                    verdict=REJECTED delta=0.4 ;;
esac
umask 022
mkdir -m 0755 "$RESULTS/$id"
dur=$(python3 -c "import time,sys;print(round(time.time()-float(sys.argv[1]),2))" "$t0")
python3 -I - "$RESULTS/$id/results.json" "$id" "$kind" "$verdict" "$delta" "$dur" <<'PY'
import json, os, sys, time
path, rid, kind, verdict, delta, dur = sys.argv[1:7]
doc = {"schema": 2, "id": rid, "run_id": f"{rid}-{time.time_ns()}", "kind": kind, "verdict": verdict,
       "duration_s": float(dur), "metric": {"kind": "tps", "headline": 1000 + float(delta) * 10},
       "stats": {"delta": float(delta) / 100, "mde": 0.03, "sigma_L": 0.01},
       "reasons": [] if verdict == "PROMOTED" else [f"localbox stub ruler: {verdict}"],
       "stub": True}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=1)
os.chmod(tmp, 0o644)
os.replace(tmp, path)
PY
echo "stub ruler: $id $verdict in ${dur}s"

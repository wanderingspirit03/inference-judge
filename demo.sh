#!/usr/bin/env bash
# The laptop demo: no accelerator, no keys, no model.
#
#     ./demo.sh [--quick] [--no-loop]
#
#   1. The cheat suite (about 3 minutes; --quick runs fewer simulations).
#      The real judge against mock servers: one that really is faster, and
#      a dozen that fake it. You get one row per cheat.
#   2. The loop rehearsal (about 3 minutes, needs Docker; --no-loop skips it).
#      A whole machine in a container: bootstrap, the three users, the queue
#      and the driver, with a scripted stand-in for the agent. The verdicts
#      in this part come from a stand-in judge too: it tests what is around
#      the judge, not the judge.
#
# Needs uv (https://docs.astral.sh/uv/). Results and logs go to a temporary
# directory; nothing in the checkout is rewritten. Exit 0 iff every row and
# every check passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/bench/.venv/bin/python"
QUICK=() LOOP=1

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=(--quick) ;;
    --no-loop) LOOP=0 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -x "$PY" ]; then
  command -v uv >/dev/null 2>&1 || { echo "demo.sh needs uv: https://docs.astral.sh/uv/" >&2; exit 2; }
  echo "== setting up bench/.venv (once)"
  make -C "$ROOT/bench" venv >/dev/null
fi

OUT="$(mktemp -d "${TMPDIR:-/tmp}/judge-demo.XXXXXX")"
rc=0

echo "== 1/2 the cheat suite: the real judge against servers that fake a speed-up"
"$PY" "$ROOT/scorecard/run_scorecard.py" ${QUICK[@]+"${QUICK[@]}"} --out "$OUT/cheat-suite.json" || rc=1

if [ "$LOOP" = 0 ]; then
  echo "== 2/2 the loop rehearsal: skipped (--no-loop)"
elif ! docker info >/dev/null 2>&1; then
  echo "== 2/2 the loop rehearsal: skipped, Docker is not running"
else
  echo "== 2/2 the loop rehearsal: a whole machine in a container (stand-in agent, stand-in judge)"
  bash "$ROOT/setup/local_box/run_local_box.sh" --out "$OUT/loop" || rc=1
fi

echo "== results and logs: $OUT"
exit "$rc"

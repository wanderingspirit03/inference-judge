#!/usr/bin/env bash
# Rehearse the funnel, end to end, before the units are enabled.
#
#     sudo bash setup/rehearse.sh [--target targets/<id>.yaml] [--quick] [--only CASE[,CASE]] [--no-clock] [--keep]
#
# Every case travels the path the unattended run uses.  The researcher
# writes experiments/<id>/ and queue/<id>.json; the installed runner
# (/opt/inference-judge/loop/runner.sh --once, as driver, with /etc/inference-judge/loop.env)
# pre-flights it and calls the ruler.  Nobody calls run.sh by hand, and the
# researcher has no right to.
#
#   driver     /opt/inference-judge/loop/run.sh --dry-run as driver: the agent (IJ_AGENT
#              in loop.env) would run as the researcher, through its adapter,
#              with --model primary --effort high, and the limit/crash
#              classifier passes its self-test.  Seconds; spends no quota.
#   preflight  launch.sh with a syntax error.  Expect PREFLIGHT_FAILED from the
#              runner in seconds, and no ruler slot spent (no results dir).
#   buildfail  launch.sh exits 1.  Expect BUILD_FAILED from the ruler.
#   hang       a server that turns ready (GET /v1/models 200) and then never
#              answers a request.  The ruler must end it by itself, as a failed
#              run: TIMEOUT or HUNG (its deadlines), or CANARY_FAILED, INVALID
#              or BUILD_FAILED if a phase gives up first.  Hitting this script's
#              outer bound is a FAIL ("the deadline is not enforced").  Costs
#              at most the target's budgets.run_s (a candidate run's limit).
#   same       the incumbent again under a new id: the leaderboard names it,
#              and its experiment dir is copied whole from the ruler's snapshot
#              (results/<its id>/snapshot/exp; the researcher's experiments/<its
#              id> if that is gone).  PROMOTED is a FAIL: the gate would promote
#              noise all day.  A WARN and a skip if neither has a launch.sh.
#
# Isolation.  The runner works in a scratch state dir with its own clock
# (loopstate start-clock) and no alert sink, so none of this reaches the
# researcher's tallies.  Afterwards the rehearsal's results dirs move out of
# /opt/bench/results (the driver's scan would count them) to
# /var/log/inference-judge/rehearse-<stamp>/, and the queue files and (unless --keep)
# the experiment dirs are removed.
#
# On PASS the real clock is restarted with the same budget, so the rehearsal is
# not charged to the run, unless a session has already started (--no-clock:
# never).
set -uo pipefail   # deliberately NOT -e: a failing case must not stop the run

# macOS ssh clients forward LC_* to the box, and sudo's audit plugin on this
# Ubuntu image aborts on a locale it cannot load, which would make every
# `sudo -u` here fail for a reason unrelated to the install.  Pin it.
export LC_ALL=C.UTF-8 LANG=C.UTF-8

usage() { sed -n '2,37p' "$0"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IJ_REPO="${IJ_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_ARG="${IJ_TARGET:-}"
QUICK=0; ONLY=""; NO_CLOCK=0; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --target) TARGET_ARG="${2:-}"; shift ;;
    --target=*) TARGET_ARG="${1#--target=}" ;;
    --quick) QUICK=1 ;;
    --only) ONLY="${2:-}"; shift ;;
    --no-clock) NO_CLOCK=1 ;;
    --keep) KEEP=1 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
for c in ${ONLY//,/ }; do
  case "$c" in driver|preflight|buildfail|hang|same) ;; *) echo "--only: a comma list of driver preflight buildfail hang same" >&2; exit 2 ;; esac
done

[ -z "$TARGET_ARG" ] && [ -f /etc/inference-judge/target.yaml ] && TARGET_ARG=/etc/inference-judge/target.yaml
# shellcheck source=setup/target.sh
. "$SCRIPT_DIR/target.sh"
target_load "$TARGET_ARG" || exit 2

LOOP_ENV=/etc/inference-judge/loop.env
LOOP=/opt/inference-judge/loop
# The agent that does the research and the researcher's user, as loop.env names them.
AGENT=$(sed -n 's/^IJ_AGENT=//p' "$LOOP_ENV" 2>/dev/null | tail -1); AGENT=${AGENT:-claude}
RESEARCHER_USER=$(sed -n 's/^IJ_RESEARCHER_USER=//p' "$LOOP_ENV" 2>/dev/null | tail -1); RESEARCHER_USER=${RESEARCHER_USER:-researcher}
# The model and the effort the driver asks the adapter for (research/loop/run.sh has the same defaults).
PRIMARY_MODEL=$(sed -n 's/^IJ_PRIMARY_MODEL=//p' "$LOOP_ENV" 2>/dev/null | tail -1 | tr -d '"'); PRIMARY_MODEL=${PRIMARY_MODEL:-primary}
EFFORT=$(sed -n 's/^IJ_EFFORT=//p' "$LOOP_ENV" 2>/dev/null | tail -1 | tr -d '"'); EFFORT=${EFFORT:-high}
ADAPTER="$(dirname "$LOOP")/adapters/$AGENT/session.sh"
STATE_DIR=/home/driver/research-state
RESEARCH_DIR=/home/researcher/research
RESULTS_DIR=/opt/bench/results
EXP_DIR="$RESEARCH_DIR/experiments"
STAMP=$(date -u +%H%M%S)
EVIDENCE="/var/log/inference-judge/rehearse-$(date -u +%Y%m%dT%H%M%SZ)"
BUILD_S=$(tget_opt budgets.build_s 1200)
HEALTH_S=$(tget_opt budgets.health_s 1200)
RUN_S=$(tget_opt budgets.run_s 2400)
PORT=$(tget_opt engine.port 8000)
SERVED=$(tget model.served_name)
RULER_BOUND=$(( ${BUILD_S%.*} + ${HEALTH_S%.*} + ${RUN_S%.*} + 600 ))

PASS=0; FAIL=0; WARNS=0
ok()   { printf '\033[32mPASS\033[0m  %-34s %s\n' "$1" "${2:-}"; PASS=$((PASS+1)); }
bad()  { printf '\033[31mFAIL\033[0m  %-34s %s\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }
warn() { printf '\033[33mWARN\033[0m  %-34s %s\n' "$1" "${2:-}"; WARNS=$((WARNS+1)); }
note() { printf '      %s\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root (it acts as researcher and as driver)" >&2; exit 1; }
for p in "$LOOP_ENV" "$LOOP/runner.sh" "$LOOP/loopstate.py" "$EXP_DIR"; do
  [ -e "$p" ] || { echo "$p missing: run setup/bootstrap.sh first" >&2; exit 1; }
done
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
# A live runner would take the rehearsal's queue files into the real run.
if command -v systemctl >/dev/null && systemctl is-active --quiet inference-judge-runner inference-judge-driver 2>/dev/null; then
  echo "inference-judge-runner/inference-judge-driver are running: stop them first (systemctl stop inference-judge-runner inference-judge-driver)" >&2; exit 1
fi
if [ -e "$STATE_DIR/runner.lock" ] && ! flock -n "$STATE_DIR/runner.lock" true; then
  echo "a runner holds $STATE_DIR/runner.lock: stop it first" >&2; exit 1
fi
target_summary
# A candidate is measured against the incumbent (ABBA/ICI), so nothing but
# the operator's baseline can be judged before there is one.
if [ ! -s "/opt/bench/leaderboard/$T_ID.json" ]; then
  echo "no incumbent for $T_ID (/opt/bench/leaderboard/$T_ID.json): run the operator baseline first" >&2
  echo "  (docs/RUNBOOK.md, Reference and calibration); every case would end INCONCLUSIVE" >&2
  exit 1
fi

# ------------------------------------------------------------------- seams
SCRATCH="/home/driver/rehearse-$STAMP"      # the runner's state for this rehearsal
SCRATCH_HYP="$SCRATCH/HYPOTHESES.jsonl"
CREATED=(); QUEUED=(); RAN=()

# AS_DRIVER <cmd...>: as the driver's user, with the units' environment.
# shellcheck disable=SC2016  # expanded by the inner shell
AS_DRIVER=(sudo -u driver -H bash -c 'set -a; . "$1"; set +a; shift; exec "$@"' _ "$LOOP_ENV")
# AS_RUNNER <cmd...>: the same, pointed at the scratch state.
AS_RUNNER=("${AS_DRIVER[@]}" env IJ_STATE_DIR="$SCRATCH" IJ_HYPOTHESES="$SCRATCH_HYP"
           IJ_ALERT= IJ_LEGACY_STATE=/nonexistent)
as_researcher() { sudo -u researcher -H bash -c "$1"; }

# make_experiment <id> <prediction>  (launch.sh on stdin), as the researcher
make_experiment() {
  local id="$1" pred="$2"
  as_researcher "mkdir -p '$EXP_DIR/$id' && cat > '$EXP_DIR/$id/launch.sh' && chmod 0755 '$EXP_DIR/$id/launch.sh'"
  CREATED+=("$EXP_DIR/$id")
  jq -nc --arg id "$id" --argjson p "$pred" \
    '{experiment: $id, hypothesis: "rehearsal", prediction: $p, unit: "pct"}' >> "$SCRATCH_HYP"
}
# copy_experiment <src-dir> <id> <prediction> <marker>: the whole directory, as
# the researcher, with the marker as launch.sh's second line
copy_experiment() {
  local src="$1" id="$2" pred="$3" marker="$4"
  tar -C "$src" -cf - . | as_researcher "mkdir -p '$EXP_DIR/$id' && tar -C '$EXP_DIR/$id' -xf -"
  { head -n 1 "$src/launch.sh"; echo "# REHEARSAL: $marker"; tail -n +2 "$src/launch.sh"; } \
    | make_experiment "$id" "$pred"
}
queue() {   # queue <id>, as the researcher: tmp + rename, like any client
  as_researcher "mkdir -p '$RESEARCH_DIR/queue' && printf '{\"id\": \"%s\", \"kind\": \"candidate\"}' '$1' > '$RESEARCH_DIR/queue/.$1.tmp' && mv '$RESEARCH_DIR/queue/.$1.tmp' '$RESEARCH_DIR/queue/$1.json'"
  QUEUED+=("$RESEARCH_DIR/queue/$1.json")
}
event_verdict() {   # the runner's verdict event for <id>, from the scratch log
  jq -rs --arg id "$1" '[.[] | select(.type == "verdict" and .id == $id)] | last | .verdict // "MISSING"' \
    "$SCRATCH/events.jsonl" 2>/dev/null || echo MISSING
}

# run_case <label> <id> <bound_s> <want-regex> <forbid-regex> [ruler|preflight]
run_case() {
  local label="$1" id="$2" bound="$3" want="$4" forbid="$5" path="${6:-ruler}"
  local log="$SCRATCH/$id.runner.log" t0 dur rc ev f="$RESULTS_DIR/$id/results.json" rv
  printf '\n\033[1m== %s (%s)\033[0m  outer bound %ss\n' "$label" "$id" "$bound"
  queue "$id"
  t0=$(date -u +%s)
  timeout -k 30 "$bound" "${AS_RUNNER[@]}" "$LOOP/runner.sh" --once > "$log" 2>&1
  rc=$?; dur=$(( $(date -u +%s) - t0 ))
  [ -d "$RESULTS_DIR/$id" ] && RAN+=("$id")
  note "runner: $(grep -m1 'runner:' "$log" | sed 's/^.*runner: //')"
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    bad "$label" "the runner had not returned after ${bound}s: the deadline is not enforced"
    note "this is the failure that costs the day: the queue would sit behind it. log: $log"
    return 1
  fi
  ev=$(event_verdict "$id")
  if [ "$path" = ruler ]; then
    if [ ! -f "$f" ]; then
      bad "$label" "no results.json from the ruler after ${dur}s (runner verdict $ev)"
      note "every outcome must be a verdict on disk. runner log: $log; bench log: $SCRATCH/runs/$id.log"
      return 1
    fi
    rv=$(jq -r '.verdict // "MISSING"' "$f")
    [ "$rv" = "$ev" ] || { bad "$label" "results.json says $rv, the runner logged $ev"; return 1; }
  else
    rv="$ev"
  fi
  if [[ "$rv" =~ $forbid ]]; then
    bad "$label" "verdict $rv after ${dur}s: this verdict must never happen here"
    jq -r '.reasons // [] | .[]' "$f" 2>/dev/null | head -3 | sed 's/^/      /'
    return 1
  fi
  if [[ "$rv" =~ $want ]]; then ok "$label" "verdict $rv after ${dur}s"; return 0; fi
  bad "$label" "verdict $rv after ${dur}s, wanted /$want/"
  jq -r '.reasons // [] | .[]' "$f" 2>/dev/null | head -3 | sed 's/^/      /'
  return 1
}

wants() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }

# The scratch state and its clock (run_min_s needs time left; 4 h is plenty).
sudo -u driver mkdir -m 0750 "$SCRATCH" || { echo "cannot create $SCRATCH as driver" >&2; exit 1; }
: > "$SCRATCH_HYP"; chown driver:driver "$SCRATCH_HYP"
"${AS_RUNNER[@]}" python3 -I "$LOOP/loopstate.py" start-clock --accel-hours 4 --wall-hours 4 >/dev/null \
  || { echo "loopstate start-clock failed in $SCRATCH" >&2; exit 1; }

# ------------------------------------------------------------------- driver
if wants driver; then
  printf '\n\033[1m== driver dry run (as driver)\033[0m\n'
  # exactly the driver's route: IJ_RESEARCHER_SUDO, then env IJ_AGENT_BIN=...
  # from loop.env, then the adapter.  run.sh does not look for the agent's own
  # command, so this is where a missing one shows.
  # shellcheck disable=SC2016  # expanded by the inner shell
  ver=$("${AS_DRIVER[@]}" bash -c 'exec $IJ_RESEARCHER_SUDO env ${IJ_AGENT_BIN:+"IJ_AGENT_BIN=$IJ_AGENT_BIN"} "$1" version' _ "$ADAPTER" 2>&1); vrc=$?
  if [ "$vrc" -eq 0 ] && [ -n "$ver" ]; then ok "$AGENT version as $RESEARCHER_USER" "$ver"
  else bad "$AGENT version as $RESEARCHER_USER" "${ver:-no output} ($ADAPTER version)"; fi
  # shellcheck disable=SC2016
  dry=$("${AS_RUNNER[@]}" bash -c 'cd "$IJ_RESEARCH_DIR" && exec "$1" --dry-run' _ "$LOOP/run.sh" 2>&1); drc=$?
  if [ "$drc" -eq 0 ]; then ok "run.sh --dry-run" "exit 0"
  else bad "run.sh --dry-run" "exit $drc"; printf '%s\n' "$dry" | tail -5 | sed 's/^/      /'; fi
  # The session's command line: the adapter, and the prompt on its stdin.
  cmdline=$(printf '%s\n' "$dry" | grep -F -- "adapters/$AGENT/session.sh run" | head -1)
  if [ -n "$cmdline" ]; then ok "driver runs adapters/$AGENT/session.sh"
  else bad "driver runs adapters/$AGENT/session.sh" "run.sh --dry-run printed no such command"; fi
  for flag in "-u $RESEARCHER_USER" "--model $PRIMARY_MODEL" "--effort $EFFORT"; do
    case "$cmdline" in
      *" $flag"*) ok "driver passes $flag" ;;
      *)          bad "driver passes $flag" "command was: ${cmdline:-<not printed>}" ;;
    esac
  done
  case "$dry" in
    *WRONG*|*"SELF-TEST FAILED"*) bad "limit/crash classifier" "see run.sh --dry-run" ;;
    *)                            ok  "limit/crash classifier" ;;
  esac
fi

# ---------------------------------------------------------------- preflight
if wants preflight; then
  id="rehearsal-preflight-$STAMP"
  make_experiment "$id" 5 <<'SH'
#!/usr/bin/env bash
# Rehearsal: a syntax error.  Pre-flight must refuse it before the ruler sees it.
if then
SH
  run_case "preflight failure" "$id" 600 '^PREFLIGHT_FAILED$' '^(PROMOTED|BASELINE)$' preflight
  starts=$(jq -s --arg id "$id" '[.[] | select(.type == "run_start" and .id == $id)] | length' "$SCRATCH/events.jsonl" 2>/dev/null)
  if [ -e "$RESULTS_DIR/$id" ] || [ "${starts:-0}" != 0 ]; then
    bad "preflight spent no ruler slot" "a results dir or run_start exists for $id"
  else ok "preflight spent no ruler slot"; fi
fi

# ---------------------------------------------------------------- buildfail
if wants buildfail; then
  id="rehearsal-buildfail-$STAMP"
  make_experiment "$id" 5 <<'SH'
#!/usr/bin/env bash
# Rehearsal: fail at once, the way a bad flag or a missing dependency fails.
set -euo pipefail
echo "[launch] rehearsal: failing on purpose" >&2
exit 1
SH
  run_case "build failure" "$id" "$RULER_BOUND" '^BUILD_FAILED$' '^(PROMOTED|BASELINE)$'
fi

# --------------------------------------------------------------------- hang
if wants hang; then
  id="rehearsal-hang-$STAMP"
  make_experiment "$id" 5 <<SH
#!/usr/bin/env bash
# Rehearsal: ready on :$PORT, then never answers a request -- a server wedged
# after start-up.  Only the ruler's own deadline can end this run.
set -euo pipefail
exec python3 - <<'PY'
import http.server, json, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({"object": "list", "data": [{"id": "$SERVED", "object": "model"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        time.sleep(10 ** 7)
http.server.ThreadingHTTPServer(("0.0.0.0", $PORT), H).serve_forever()
PY
SH
  note "budget: the ruler's own deadline (budgets.run_s ${RUN_S}s); at most ~$((${RUN_S%.*} / 60)) min"
  # Any failed-run verdict passes: what matters is that the ruler ended it and
  # wrote a verdict.  Which deadline or phase did it is in the PASS line.
  run_case "hang is ended by the ruler" "$id" "$RULER_BOUND" \
    '^(TIMEOUT|HUNG|CANARY_FAILED|INVALID|BUILD_FAILED)$' '^(PROMOTED|INCONCLUSIVE|BASELINE)$'
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "$id"; then
    bad "hang: container cleaned up" "a container for $id is still running"
  else ok "hang: container cleaned up" "none left"; fi
fi

# --------------------------------------------------------------------- same
if wants same && [ "$QUICK" != 1 ]; then
  id="rehearsal-same-$STAMP"
  # The incumbent again: the leaderboard names it, and the ruler's own snapshot
  # of its experiment dir (patches and deps.txt included) is what was measured.
  # The researcher's copy is the fallback.
  src=""
  board="/opt/bench/leaderboard/$T_ID.json"
  inc=$(jq -r '.best.id // empty' "$board" 2>/dev/null)
  for d in "$RESULTS_DIR/$inc/snapshot/exp" "$EXP_DIR/$inc"; do
    [ -n "$inc" ] && [ -f "$d/launch.sh" ] && { src="$d"; break; }
  done
  if [ -n "$src" ]; then
    copy_experiment "$src" "$id" 0 "a copy of the incumbent $inc ($src). It must NOT be promoted."
  fi
  if [ -z "$src" ]; then
    warn "same as reference" "skipped: no launch.sh for the incumbent '${inc:-none}' of $T_ID in its results snapshot or in $EXP_DIR"
  else
    note "same as $src"
    run_case "same as reference" "$id" "$RULER_BOUND" \
      '^(REJECTED|REJECTED_SCREEN|INCONCLUSIVE)$' '^PROMOTED$'
    if [ "$(jq -r '.verdict // ""' "$RESULTS_DIR/$id/results.json" 2>/dev/null)" = PROMOTED ]; then
      note "PROMOTED here means the gate promotes noise. Do not start the loop."
    fi
  fi
fi

# ---------------------------------------------------------------- afterwards
echo
for q in ${QUEUED[@]+"${QUEUED[@]}"}; do rm -f "$q"; done
if [ "${#RAN[@]}" -gt 0 ]; then
  # The driver's scan counts every results dir; the evidence moves out of its way.
  mkdir -p "$EVIDENCE/results"
  for id in "${RAN[@]}"; do mv "$RESULTS_DIR/$id" "$EVIDENCE/results/"; done
fi
mkdir -p "$EVIDENCE/state"
cp -a "$SCRATCH/." "$EVIDENCE/state/" 2>/dev/null
rm -rf "$SCRATCH"
note "evidence (results, runner logs, scratch events): $EVIDENCE"
if [ "$KEEP" = 1 ]; then
  note "--keep: leaving ${#CREATED[@]} rehearsal experiment dir(s) in $EXP_DIR"
else
  for d in ${CREATED[@]+"${CREATED[@]}"}; do rm -rf "$d"; done
fi

printf '\n%d passed, %d warnings, %d failed\n' "$PASS" "$WARNS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
  echo
  echo "DO NOT START THE LOOP. A funnel that cannot end a broken experiment will"
  echo "spend the day on one, and one that promotes a copy of the reference will"
  echo "fill the leaderboard with noise. Fix it and rehearse again."
  exit 1
fi

# ------------------------------------------------------------- the real clock
if [ "$NO_CLOCK" = 1 ]; then
  note "--no-clock: the run clock is left as it is"
elif [ -s "$STATE_DIR/events.jsonl" ] && [ "$(jq -s '[.[] | select(.type == "session_start")] | length' "$STATE_DIR/events.jsonl")" -gt 0 ]; then
  note "a session has already run: the clock is left as it is"
else
  # Restart it with the budget bootstrap chose (the last run_config wins in replay).
  read -r accel_h wall_h < <(jq -rs '[.[] | select(.type == "run_config")] | last //
      {accel_budget_s: 86400, clock_start: 0, wall_deadline: 86400}
      | "\(.accel_budget_s / 3600) \((.wall_deadline - .clock_start) / 3600)"' "$STATE_DIR/events.jsonl" 2>/dev/null \
      || echo "24 24")
  if "${AS_DRIVER[@]}" env IJ_LEGACY_STATE=/nonexistent python3 -I "$LOOP/loopstate.py" start-clock \
       --accel-hours "${accel_h:-24}" --wall-hours "${wall_h:-24}"; then
    note "the rehearsal is not charged to the run."
  else bad "restart the run clock" "loopstate start-clock failed"; exit 1; fi
fi
echo
echo "Next:  systemctl enable --now inference-judge-runner inference-judge-driver inference-judge-heartbeat.timer"

#!/usr/bin/env bash
# The driver.  It owns the clock, the ledgers and the recovery policy; the agent
# owns the science; the runner (loop/runner.sh, a separate process) owns the
# ruler queue.  Under systemd it is inference-judge-driver.service (setup/systemd/);
# by hand:
#
#     ./loop/run.sh
#
# Each pass asks loop/loopstate.py what to do next.  loopstate replays the
# append-only event log in the state dir (default ~/research-state/), scans
# the ruler's results, and answers one of
#     session <model> <timeout_s> <prompt_file> <reason>   (it has just written BRIEF.md)
#     sleep <seconds> <reason>
#     finish <reason>
# Sessions are triggered by events: a verdict landed, the queue fell below its
# target, a timer, the wrap-up before the end.  Measurement never waits for a
# session: the runner keeps draining the queue while the agent thinks.
#
# This script only runs the agent's adapter (adapters/$IJ_AGENT/session.sh,
# one session per call), classifies how the session ended (ok, timeout, quota
# limit, auth failure, crash) and reports that back.  It never
# exits on crashes: crashes back off, then move to the fallback model, then
# pause with an ALERT while the runner keeps measuring.  It exits 10 when the
# run's budget is spent (systemd: RestartPreventExitStatus=10).
set -uo pipefail

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="${IJ_RESEARCH_DIR:-$(dirname "$LOOP_DIR")}"
STATE_DIR="${IJ_STATE_DIR:-$HOME/research-state}"
LOG="${IJ_DRIVER_LOG:-$STATE_DIR/driver.log}"
# Which coding agent does the research.  Its adapter turns one session into one
# JSON result (adapters/README.md); adapters/ sits beside loop/, in the repo
# and in the root-owned install.
AGENT="${IJ_AGENT:-claude}"
ADAPTERS_DIR="$(dirname "$LOOP_DIR")/adapters"
ADAPTER="$ADAPTERS_DIR/$AGENT/session.sh"
# The agent's command, when it is not on the researcher's PATH.  sudo resets
# the environment, so it is handed to the adapter after the sudo prefix.
AGENT_ENV=()
[ -n "${IJ_AGENT_BIN:-}" ] && AGENT_ENV=(env "IJ_AGENT_BIN=$IJ_AGENT_BIN")
PY="${IJ_PY:-python3}"
# Every model runs at high reasoning effort; the adapter passes it on.
EFFORT="${IJ_EFFORT:-high}"
KILL_GRACE="${IJ_KILL_GRACE:-60}"           # SIGKILL this long after SIGTERM
# Quota back-off: first sleep, doubled on every consecutive
# limit, capped.  A parsed reset time wins over the back-off but is capped too.
RATE_LIMIT_SLEEP="${IJ_RATE_LIMIT_SLEEP:-600}"          # 10 min, then 20, 40
RATE_LIMIT_SLEEP_MAX="${IJ_RATE_LIMIT_SLEEP_MAX:-3600}" # never sleep past 1 h
# When the driver is not the researcher's Unix user (the hardened layout, see
# setup/harden_research.sh), sessions and the research-repo commit run through
# this prefix, e.g. "sudo -n -u researcher -H".  Empty = same user.
read -r -a RESEARCHER_SUDO <<< "${IJ_RESEARCHER_SUDO:-}"
GIT_COMMIT="${IJ_GIT_COMMIT:-1}"
EXIT_FINISHED=10
DRY_RUN=0
ONCE=0
MAX_LOOPS=0
export IJ_STATE_DIR="$STATE_DIR" IJ_RESEARCH_DIR="$RESEARCH_DIR"

usage() {
  cat <<'USAGE'
Usage: ./loop/run.sh [--dry-run] [--once] [--max-loops N] [--install-cron] [--help]
       ./loop/run.sh --start-clock [--accel-hours H] [--wall-hours H | --deadline EPOCH]
       ./loop/run.sh --resume

  --dry-run       on a copy of the state dir: print the next action, BRIEF.md
                  and the prompt it would send, and self-test the classifier.
                  Starts no agent and writes nothing real.
  --once          stop after one session.
  --max-loops N   stop after N decisions (tests).
  --start-clock   start the run clock now.  The budget is accelerator-hours
                  (charged while the box is up) plus a wall deadline; the run
                  ends at whichever comes first.  Defaults: 24 and 24.  Without
                  this, the clock is the one bootstrap started.
  --resume        clear a pause (or: touch RESUME in the state dir).
  --install-cron  add the daily literature pass to this user's crontab.

Environment: IJ_STATE_DIR, IJ_RESEARCH_DIR, IJ_RESULTS_DIR, IJ_AGENT (the
adapter to use: claude, codex, pi or your own; default claude), IJ_AGENT_BIN,
IJ_RESEARCHER_SUDO, IJ_EFFORT, IJ_ALERT (an http(s) URL for an ntfy-style
POST, or a file), IJ_PRIMARY_MODEL, IJ_FALLBACK_MODEL (default: the words
"primary" and "fallback", which the adapter maps to models),
IJ_MAX_CRASHES, IJ_FALLBACK_SESSIONS,
IJ_BACKOFF_BASE_S, IJ_ITER_TIMEOUT_S, IJ_QUEUE_TARGET, IJ_TIMER_S,
IJ_POLL_S, IJ_RATE_LIMIT_SLEEP, IJ_RATE_LIMIT_SLEEP_MAX, IJ_PY.

State dir: events.jsonl (the log), runs.jsonl, process.jsonl, BRIEF.md,
prompt.txt, heartbeat.json, ALERTS.md, driver.log, iterations/<n>.json (the
adapter's result for session n), runs/<id>.log (the bench output).
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --once) ONCE=1 ;;
    --max-loops) MAX_LOOPS="${2:?}"; shift ;;
    --install-cron) INSTALL_CRON=1 ;;
    --start-clock) shift; exec "$PY" -I "$LOOP_DIR/loopstate.py" start-clock "$@" ;;
    --resume) exec "$PY" -I "$LOOP_DIR/loopstate.py" resume ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ ! -x "$ADAPTER" ]; then
  echo "IJ_AGENT=$AGENT: no adapter at $ADAPTER" >&2
  echo "adapters here: $(cd "$ADAPTERS_DIR" 2>/dev/null && ls -d -- */ 2>/dev/null | tr -d / | grep -v '^TEMPLATE$' | tr '\n' ' ')" >&2
  exit 1
fi

now()   { date -u +%s; }
iso()   { date -u +%Y-%m-%dT%H:%M:%SZ; }
say()   { printf '%s %s\n' "$(iso)" "$*" | tee -a "$LOG"; }
logline() { printf '%s %s\n' "$(iso)" "$*" >> "$LOG"; }
lstate() { "$PY" -I "$LOOP_DIR/loopstate.py" "$@"; }

# A session is capped so it can never run past the end of the run.  timeout(1)
# on the box, gtimeout on a Mac with coreutils, else a small Python wrapper
# with the same contract: SIGTERM the process group at the cap, SIGKILL after
# the grace, exit 124.
TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
TIMEOUT_KILL=()
if [ -n "$TIMEOUT_BIN" ] && "$TIMEOUT_BIN" -k 1 1 true >/dev/null 2>&1; then
  TIMEOUT_KILL=(-k "$KILL_GRACE")
fi
CAP_PY='
import os, signal, subprocess, sys
secs, grace = float(sys.argv[1]), float(sys.argv[2])
p = subprocess.Popen(sys.argv[3:], start_new_session=True)
try:
    sys.exit(p.wait(timeout=secs))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait()
    sys.exit(124)
'
run_capped() {  # run_capped <seconds> <command...>
  local secs="$1"; shift
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" ${TIMEOUT_KILL[@]+"${TIMEOUT_KILL[@]}"} "$secs" "$@"
  else
    "$PY" -I -c "$CAP_PY" "$secs" "$KILL_GRACE" "$@"
  fi
}

# --------------------------------------------------------------- cron install
if [ "${INSTALL_CRON:-0}" = 1 ]; then
  line="0 * * * * [ \$(date -u +\\%H) = 00 ] && $LOOP_DIR/literature.sh >> $STATE_DIR/literature.log 2>&1"
  ( crontab -l 2>/dev/null | grep -v 'loop/literature.sh'; echo "$line" ) | crontab -
  echo "installed: $line"
  crontab -l
  exit 0
fi

# ------------------------------------------------------ outcome classification
# Two failures look alike from outside and must be treated in opposite ways:
#
#   a quota limit  is normal, expected several times a day, and must
#                  NEVER count as a crash -- three of them would otherwise stop
#                  the driver for good and leave the rented machine idle.
#   a crash        is a defect; crashes back off, then fall back to another
#                  model, then pause with an ALERT (loopstate.py).
#   an auth error  (expired login, bad key, no credit) is neither: retrying
#                  cannot fix it, so the loop pauses and alerts at once.  It is
#                  checked BEFORE the limit pattern, because auth messages often
#                  say "try again", which LIMIT_RE matches.
#
# Two rules learned the hard way:
#
#  * Free prose from a SUCCESSFUL iteration is never
#    consulted, so an iteration that merely discusses provider rate limiting is
#    not thrown away.  `.result` is read only when the envelope says it is an
#    error.
#  * The classification does not depend on the exit status.
#    Claude Code 2.1.x reports quota exhaustion both as a non-zero exit and as
#    an error envelope with exit 0, and the message wording varies ("You've hit
#    your usage limit", session/weekly limits, overloaded_error, 429, 529).
#    The old detector missed all but four spellings and missed `.errors[]`
#    entirely.
LIMIT_RE="you've hit your|you have hit your|hit your limit|usage limit|session limit|weekly limit|monthly limit|5-hour limit|rate.?limit|rate_limit|quota exceeded|quota_exceeded|insufficient quota|out of credit|overloaded|over capacity|at capacity|capacity constraint|service_unavailable|(^|[^0-9])(429|529)([^0-9]|\$)|resets at|resets in|reset at|try again"

# True when the JSON result object is an error envelope of any shape.
is_error_envelope() {   # <stdout-file>
  jq -e '(.is_error? == true) or ((.errors? // [] | length) > 0) or (.error? != null)' \
    "$1" >/dev/null 2>&1
}

# All the error-bearing text of a result object, lowercased.  `.result` is
# included only for an error envelope (the first rule above).
error_text_of() {   # <stdout-file>
  jq -r '
    def s: if . == null then "" else tostring end;
    ((.is_error? == true) or ((.errors? // [] | length) > 0) or (.error? != null)) as $e
    | [ (.error? | s),
        ((.errors? // []) | map(tostring) | join(" ")),
        (.subtype? | s),
        (.message? | s),
        (.permission_denials? | s),
        (if $e then (.result? | s) else "" end) ] | join(" ")' "$1" 2>/dev/null \
    | tr '[:upper:]' '[:lower:]'
}

# "not logged in" is also the prefix an adapter puts in front of a failed
# session's text when the agent's own login check fails (adapters/README.md).
AUTH_RE="authentication_error|authentication failed|unauthorized|invalid api key|invalid_api_key|incorrect api key|invalid x-api-key|invalid bearer|no api key|missing api key|api key (is )?(missing|not set|not found)|failed to authenticate|oauth (token|session) (has )?expired|token has expired|token expired|please run /login|run /login|not logged in|login required|credit balance is too low|(^|[^0-9])401([^0-9]|\$)"

# classify <stdout-file> <stderr-file> <rc> -> limit | auth | crash | ok
classify() {
  local out="$1" err="$2" rc="$3" txt
  if jq -e . "$out" >/dev/null 2>&1; then
    if is_error_envelope "$out"; then
      txt=$(error_text_of "$out")
      if [[ "$txt" =~ $AUTH_RE ]]; then printf 'auth\n'; return 0; fi
      if [[ "$txt" =~ $LIMIT_RE ]]; then printf 'limit\n'; return 0; fi
    fi
    if [ "$rc" -ne 0 ]; then
      txt=$(tail -c 4000 "$err" 2>/dev/null | tr '[:upper:]' '[:lower:]')
      if [[ "$txt" =~ $AUTH_RE ]]; then printf 'auth\n'; return 0; fi
      if [[ "$txt" =~ $LIMIT_RE ]]; then printf 'limit\n'; return 0; fi
      printf 'crash\n'; return 0
    fi
    printf 'ok\n'; return 0
  fi
  # No result object: a hard stop before it was written, or nothing at all.
  if [ "$rc" -ne 0 ]; then
    txt=$( { tail -c 4000 "$err"; tail -c 4000 "$out"; } 2>/dev/null \
           | tr '[:upper:]' '[:lower:]')
    if [[ "$txt" =~ $AUTH_RE ]]; then printf 'auth\n'; return 0; fi
    if [[ "$txt" =~ $LIMIT_RE ]]; then printf 'limit\n'; return 0; fi
    printf 'crash\n'; return 0
  fi
  printf 'ok\n'
}

# The reset moment a limit message advertises, as an epoch, or nothing.  Three
# shapes are understood: an ISO timestamp, a unix timestamp next to the word
# "reset", and the "resets at 3pm" / "resets at 15:00" wording Claude Code
# prints.  A clock time with no zone is read in the box's local time, which is
# UTC on this image; a wrong guess costs at most RATE_LIMIT_SLEEP_MAX because
# every sleep is capped.
parse_reset_epoch() {   # <text> -> epoch on stdout
  local txt="$1" m e hh mm ap
  m=$(printf '%s' "$txt" | grep -oiE '[0-9]{4}-[0-9]{2}-[0-9]{2}[t ][0-9]{2}:[0-9]{2}(:[0-9]{2})?z?' | head -1)
  if [ -n "$m" ]; then
    e=$(date -u -d "$m" +%s 2>/dev/null || true)
    if [ -n "$e" ]; then printf '%s\n' "$e"; return 0; fi
  fi
  m=$(printf '%s' "$txt" | grep -oiE 'reset[a-z_]*"?[:= ]+"?1[0-9]{9}' | grep -oE '1[0-9]{9}$' | head -1)
  if [ -n "$m" ]; then printf '%s\n' "$m"; return 0; fi
  m=$(printf '%s' "$txt" | grep -oiE 'reset[a-z]*( at| by)? [0-9]{1,2}(:[0-9]{2})? ?([ap]m)?' | head -1 | tr '[:upper:]' '[:lower:]')
  if [ -n "$m" ]; then
    # [^0-9:] and not [^0-9]: leftmost-longest matching would otherwise let
    # ".*" eat "15" out of "15:30" and read the MINUTES as the hour.
    hh=$(printf '%s' "$m" | sed -nE 's/.*[^0-9:]([0-9]{1,2})(:[0-9]{2})?( ?[ap]m)?$/\1/p')
    mm=$(printf '%s' "$m" | sed -nE 's/.*[0-9]{1,2}:([0-9]{2})( ?[ap]m)?$/\1/p')
    ap=$(printf '%s' "$m" | sed -nE 's/.*([ap])m$/\1/p')
    case "$hh" in ''|*[!0-9]*) return 1 ;; esac
    mm=${mm:-00}
    [ "$ap" = p ] && [ "$hh" -lt 12 ] && hh=$((hh + 12))
    [ "$ap" = a ] && [ "$hh" -eq 12 ] && hh=0
    [ "$hh" -gt 23 ] && return 1
    hh=$(printf '%02d' "$hh")
    # GNU date on the box; the BSD form keeps this testable on a Mac.
    e=$(date -d "today $hh:$mm" +%s 2>/dev/null \
        || date -j -f '%Y-%m-%d %H:%M:%S' "$(date +%Y-%m-%d) $hh:$mm:00" +%s 2>/dev/null \
        || true)
    if [ -n "$e" ]; then
      [ "$e" -le "$(now)" ] && e=$((e + 86400))
      printf '%s\n' "$e"; return 0
    fi
  fi
  return 1
}

# How long to wait after a limit: until the advertised reset if there is one,
# otherwise 10, 20, 40 minutes as the streak grows.  Always capped.
limit_sleep_for() {   # <streak> <stdout-file> <stderr-file>
  local streak="$1" out="$2" err="$3" txt reset s i
  txt="$( { error_text_of "$out"; tail -c 4000 "$err" 2>/dev/null; } | tr '\n' ' ')"
  reset=$(parse_reset_epoch "$txt" || true)
  if [ -n "${reset:-}" ] && [ "$reset" -gt "$(now)" ]; then
    s=$(( reset - $(now) + 60 ))
  else
    s=$RATE_LIMIT_SLEEP
    i=1
    while [ "$i" -lt "$streak" ] && [ "$s" -lt "$RATE_LIMIT_SLEEP_MAX" ]; do
      s=$(( s * 2 )); i=$(( i + 1 ))
    done
  fi
  [ "$s" -gt "$RATE_LIMIT_SLEEP_MAX" ] && s=$RATE_LIMIT_SLEEP_MAX
  [ "$s" -lt 60 ] && s=60
  printf '%s\n' "$s"
}

# ------------------------------------------------------------------ session
# One session of the agent, as the researcher, in the research repo.  The
# prompt goes to the adapter on stdin; its result goes to iterations/<n>.json
# in the state dir: the only verbatim record of what the agent said it did.
run_session() {   # <model> <timeout_s> <prompt_file> <reason>
  local model="$1" cap="$2" prompt_file="$3" reason="$4" n out err rc t0 dur cls streak sleep_s
  n=$(lstate get sessions)
  mkdir -p "$STATE_DIR/iterations"
  out="$STATE_DIR/iterations/$n.json"; err="$STATE_DIR/iterations/$n.stderr"
  logline "session=$n start model=$model cap=${cap}s reason=$reason"
  t0=$(now)
  ( cd "$RESEARCH_DIR" && run_capped "$cap" ${RESEARCHER_SUDO[@]+"${RESEARCHER_SUDO[@]}"} \
      ${AGENT_ENV[@]+"${AGENT_ENV[@]}"} "$ADAPTER" run --model "$model" --effort "$EFFORT" ) \
    < "$prompt_file" > "$out" 2> "$err"
  rc=$?
  dur=$(( $(now) - t0 ))
  [ -s "$err" ] || rm -f "$err"

  cls=$(classify "$out" "$err" "$rc")
  if [ "$cls" = crash ] && { [ "$rc" = 124 ] || [ "$rc" = 137 ]; }; then
    cls=timeout     # the cap, not a defect: never counted as a crash
  fi
  if [ "$cls" = limit ]; then
    streak=$(( $(lstate get limit_streak) + 1 ))
    sleep_s=$(limit_sleep_for "$streak" "$out" "$err")
    lstate session-end --class limit --rc "$rc" --dur "$dur" --sleep "$sleep_s" >/dev/null
    say "session=$n quota limit (streak $streak); sleeping $((sleep_s/60)) min, not counted"
  else
    lstate session-end --class "$cls" --rc "$rc" --dur "$dur" --usage-file "$out" >/dev/null
    say "session=$n $cls rc=$rc in ${dur}s (model $model, trigger $reason)"
  fi
  if [ "$GIT_COMMIT" = 1 ]; then
    ${RESEARCHER_SUDO[@]+"${RESEARCHER_SUDO[@]}"} git -C "$RESEARCH_DIR" add -A >/dev/null 2>&1 || true
    ${RESEARCHER_SUDO[@]+"${RESEARCHER_SUDO[@]}"} git -C "$RESEARCH_DIR" diff --cached --quiet 2>/dev/null \
      || ${RESEARCHER_SUDO[@]+"${RESEARCHER_SUDO[@]}"} git -C "$RESEARCH_DIR" commit -qm \
           "driver: session $n ($cls, $reason)" >/dev/null 2>&1 || true
  fi
}

# ------------------------------------------------------------------- dry run
if [ "$DRY_RUN" = 1 ]; then
  tmp=$(mktemp -d)
  [ -f "$STATE_DIR/events.jsonl" ] && cp "$STATE_DIR/events.jsonl" "$tmp/"
  # A copy of the log, no alert sink, no job reaping: nothing real is touched.
  IJ_STATE_DIR="$tmp"; IJ_ALERT=""; IJ_DRY_RUN=1
  export IJ_STATE_DIR IJ_ALERT IJ_DRY_RUN
  echo "--- next action (state dir copied to $tmp)"
  action=$(lstate next); echo "$action"
  if [ -f "$tmp/BRIEF.md" ]; then echo; echo "--- BRIEF.md"; cat "$tmp/BRIEF.md"; fi
  if [ -f "$tmp/prompt.txt" ]; then
    echo "--- prompt that would be sent"; cat "$tmp/prompt.txt"; echo
    echo; echo "--- command that would run (cwd $RESEARCH_DIR)"
    echo "${RESEARCHER_SUDO[*]:-} ${AGENT_ENV[*]:-} $ADAPTER run --model $(echo "$action" | cut -d' ' -f2) --effort $EFFORT < <prompt above>" | tr -s ' ' | sed 's/^ //'
  fi
  echo
  echo "--- classifier (a limit is never a crash; an auth failure is never a crash)"
  rl_out=$(mktemp); rl_err=$(mktemp)
  dry_fail=0
  cls() {  # cls <want> <rc> <label>
    local want="$1" rc="$2" label="$3" got
    got=$(classify "$rl_out" "$rl_err" "$rc")
    if [ "$got" = "$want" ]; then printf '  %-44s %s\n' "$label" "ok ($got)"
    else printf '  %-44s WRONG: wanted %s, got %s\n' "$label" "$want" "$got"; dry_fail=1; fi
  }
  : > "$rl_err"
  jq -n '{type:"result",is_error:false,result:"I checked whether the provider applies a rate limit of 429 requests; it does."}' > "$rl_out"
  cls ok    0 "success whose prose mentions 429"
  jq -n '{type:"result",is_error:false,result:"The 401 handler and the login page were unauthorized."}' > "$rl_out"
  cls ok    0 "success whose prose mentions 401"
  jq -n '{type:"result",is_error:true,subtype:"error_during_execution",result:"Claude usage limit reached. Try again later."}' > "$rl_out"
  cls limit 1 "is_error, 'usage limit'"
  jq -n '{type:"result",is_error:true,subtype:"error","result":"You'"'"'ve hit your session limit. Resets at 3pm."}' > "$rl_out"
  cls limit 0 "session limit, exit 0"
  jq -n '{type:"result",is_error:false,errors:[{type:"overloaded_error",message:"Overloaded"}]}' > "$rl_out"
  cls limit 0 "errors[] overloaded_error, exit 0"
  jq -n '{type:"result",error:{type:"rate_limit_error",message:"429 too many requests"}}' > "$rl_out"
  cls limit 0 ".error rate_limit_error"
  jq -n '{type:"result",is_error:true,result:"OAuth token has expired. Please run /login. Try again after logging in."}' > "$rl_out"
  cls auth  1 "expired login, says 'try again' (auth first)"
  jq -n '{type:"result",is_error:true,subtype:"success",result:"Failed to authenticate: OAuth session expired and could not be refreshed"}' > "$rl_out"
  cls auth  1 "expired login that could not be refreshed"
  jq -n '{type:"result",error:{type:"authentication_error",message:"invalid x-api-key"}}' > "$rl_out"
  cls auth  0 ".error authentication_error"
  jq -n '{type:"result",is_error:true,subtype:"error_during_execution",result:"Edit tool failed: file not found"}' > "$rl_out"
  cls crash 1 "a genuine error, non-zero exit"
  jq -n '{type:"result",agent:"codex",is_error:true,result:"not logged in: stream disconnected before completion"}' > "$rl_out"
  cls auth  1 "another agent, logged out (adapter's prefix)"
  jq -n '{type:"result",agent:"pi",is_error:true,result:"No API key found for provider openrouter"}' > "$rl_out"
  cls auth  1 "another agent, no API key"
  jq -n '{type:"result",agent:"codex",is_error:true,result:"You'"'"'ve hit your usage limit. Try again in 3 hours 12 minutes."}' > "$rl_out"
  cls limit 1 "another agent, usage limit"
  jq -n '{type:"result",agent:"pi",is_error:true,result:"the agent printed no result"}' > "$rl_out"
  cls crash 1 "an adapter's 'no result'"
  printf 'not json\n' > "$rl_out"; printf 'Error: 429 rate limit exceeded\n' > "$rl_err"
  cls limit 1 "hard stop, no JSON, stderr says 429"
  printf 'Invalid API key · Please run /login\n' > "$rl_err"
  cls auth  1 "hard stop, no JSON, stderr says /login"
  printf 'Segmentation fault\n' > "$rl_err"
  cls crash 1 "hard stop, no JSON, real crash"
  rm -f "$rl_out" "$rl_err"
  echo "  reset-time parse 'resets at 3pm'         : $(parse_reset_epoch "you've hit your limit. resets at 3pm" >/dev/null 2>&1 && echo "ok" || echo "not parsed (back-off used)")"
  echo "  back-off 1/2/3/9 (s)                     : $(limit_sleep_for 1 /dev/null /dev/null) $(limit_sleep_for 2 /dev/null /dev/null) $(limit_sleep_for 3 /dev/null /dev/null) $(limit_sleep_for 9 /dev/null /dev/null)"
  [ "$dry_fail" = 0 ] || echo "  CLASSIFIER SELF-TEST FAILED"
  rm -rf "$tmp"
  echo
  echo "(dry run: nothing was written to the real state dir and no agent was started)"
  [ "$dry_fail" = 0 ] || exit 1
  exit 0
fi

# ---------------------------------------------------------------- main loop
mkdir -p "$STATE_DIR" || exit 1
chmod 0755 "$STATE_DIR" 2>/dev/null || true
printf '%s\n' "$$" > "$STATE_DIR/driver.pid"
lstate start || { echo "loopstate.py start failed" >&2; exit 1; }
say "driver: start pid=$$ state=$STATE_DIR research=$RESEARCH_DIR"

loops=0
while true; do
  if ! action=$(lstate next); then
    say "driver: loopstate next failed; retrying in 30 s"
    sleep 30; continue
  fi
  read -r kind a b c d <<< "$action"
  case "$kind" in
    session) run_session "$a" "$b" "$c" "$d" ;;
    sleep)   sleep "$a" ;;
    finish)  say "driver: finished -- $a ${b:-} ${c:-} ${d:-}"; exit "$EXIT_FINISHED" ;;
    *)       say "driver: unknown action '$action'"; sleep 30 ;;
  esac
  loops=$((loops + 1))
  [ "$ONCE" = 1 ] && [ "$kind" = session ] && { say "driver: --once done"; exit 0; }
  [ "$MAX_LOOPS" -gt 0 ] && [ "$loops" -ge "$MAX_LOOPS" ] && exit 0
done

#!/usr/bin/env bash
# run.sh - the ruler.
#
# Claude writes an experiment in research/experiments/<id>/ and then asks this
# script to measure it. This script is owned by a different Unix user, so
# Claude cannot change how it measures. That separation is the whole reason
# anyone should believe the numbers.
#
# Everything about the hardware, the engine and the model comes from one
# target file (targets/<id>.yaml, DESIGN 2.1), named by $BENCH_ROOT/TARGET.
#
# A verdict is drawn from whole launches (DESIGN 2.2-2.6), in ABBA order
# (I C C I) or ICI (I C I): the incumbent is started from its stored
# snapshot, and each launch is health within the compile budget, one unscored
# warm sweep, then metric.blocks scored sweeps. C1 also runs the canary. The
# screen (stats.screen against the sentinel) runs on C1, or on a short launch
# S of its own when the target sets metric.screen_window_s. Every C runs
# quality2 against the fixed reference R; the verdict is stats.abba or
# stats.ici.
#
# In order:
#   1  check the id, refuse to overwrite an old result
#   1b refuse to run when a ruler file changed since RULER_LOCK (clock start)
#   2  take the exclusive bench lock (one experiment at a time)
#   3  copy the experiment and the engine sources somewhere private
#   4  snapshot what won: the experiment dir, the engine diffs (staged and
#      unstaged), the untracked file list and the HEADs - collected by a git
#      that runs inside a throwaway container, never as bench
#   5  build the image if the experiment brought a Dockerfile, and bake its
#      deps.txt into a derived image (the build has network, the launch never)
#   6  the scored launch is sealed: --network none, no published port, every
#      mount read-only (container_entry copies /ro/* into the container's own
#      filesystem). A probe container with the same flags records what the
#      seal holds in results.json `isolation`. The ruler measures from a
#      sidecar container in the server's network namespace.
#   7  the launches, the screen, the quality gate, the verdict
#   8  save everything, kill the containers, print the verdict
#
# The whole body runs under a bench-owned watchdog: whatever phase is stuck,
# the run is killed and written up as TIMEOUT when the budget is gone. A TPS
# sweep that outlives its own bound is HUNG.
#
# Every verdict is a result, so the exit code is 0 for PROMOTED, REJECTED,
# INCONCLUSIVE, INVALID and friends. A non-zero exit means the measurement
# itself broke (2 = bad usage, 70 = infrastructure, a moved ruler included).
#
# Modes accepted through sudo (the researcher's and the driver's entry point):
#   <id>                a scored run
#   --instrument <id>   unscored: one launch, /out mounted for profiles,
#                       verdict INSTRUMENT, never on the leaderboard
#   --lock-ruler        write RULER_LOCK for the live ruler; refuses when one
#                       already exists, so it locks once and only once
# Operator and test modes (refused through sudo):
#   --dry-run           print every command this run would execute, run nothing
#   --local             run launch.sh as a host process (no Docker, no
#                       seal) - the fast end-to-end tests use it
#   --calibrate <id>    TPS Step 0: CALIBRATE_LAUNCHES launches of the
#                       incumbent -> calibration/<target>.json
#   --confirm <id>      TPS Step 3: ICI of the promoted <id> against the
#                       previous best on a disjoint prompt pool; the
#                       leaderboard keeps the confirmed effect
set -euo pipefail

# --------------------------------------------------------- 0. isolation ----
# The caller (unix user `researcher`) controls their own environment and their own
# working directory. Neither may reach into this script: a researcher-owned
# json.py or pathlib.py next to the caller's cwd must never be importable by a
# python that runs as bench. So: scrub the loader environment, move to the
# directory this script actually lives in, and pin the interpreter.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONEXECUTABLE \
      LD_PRELOAD LD_LIBRARY_PATH DOCKER_HOST || true

SELF="$0"
_resolved="$(readlink -f "$0" 2>/dev/null || true)"
[[ -n "$_resolved" ]] && SELF="$_resolved"
SELF_DIR="$(cd -- "$(dirname -- "$SELF")" && pwd -P)"
cd "$SELF_DIR"

# Invoked through sudo? Then this is the production entry point and nothing the
# caller can set is trusted: no flags, no environment overrides.
UNDER_SUDO=0
[[ -n "${SUDO_USER:-}" ]] && UNDER_SUDO=1

# cfg NAME DEFAULT - read an environment override, but only when we are not
# running as the privileged entry point.
cfg() {
  local name="$1" def="$2" val
  if (( UNDER_SUDO )); then printf '%s' "$def"; return 0; fi
  val="${!name:-}"
  printf '%s' "${val:-$def}"
}

# ---------------------------------------------------------------- settings --
BENCH_ROOT="$(cfg BENCH_ROOT "$SELF_DIR")"
RESEARCH_ROOT="$(cfg RESEARCH_ROOT /home/researcher/research)"
SRC_ROOT="$(cfg IJ_SRC_ROOT /home/researcher)"        # nothing is copied from outside this
MODELS_DIR="$(cfg MODELS_DIR /opt/models)"
CONTAINER_VENV="$(cfg CONTAINER_VENV /opt/venv/vllm)"  # where deps.txt is installed
# The hash written at clock start (DESIGN 2.6). Tests point it elsewhere;
# through sudo it is always $BENCH_ROOT/RULER_LOCK.
RULER_LOCK_PATH="$(cfg BENCH_RULER_LOCK "$BENCH_ROOT/RULER_LOCK")"
# The image the measurement sidecar and the isolation probe run in: python and
# the bench dependencies, nothing else (targets/cpu-mock.Dockerfile).
RULER_IMAGE="$(cfg IJ_RULER_IMAGE inference-judge-ruler:latest)"

# Target-derived settings: an override here wins, otherwise the target says.
# They are resolved in load_target_settings, after the arguments.
BASE_IMAGE_OVR="$(cfg BASE_IMAGE "")"
MODEL_NAME_OVR="$(cfg MODEL_NAME "")"
HEALTH_TIMEOUT_OVR="$(cfg HEALTH_TIMEOUT_S "")"
BUILD_TIMEOUT_OVR="$(cfg BUILD_TIMEOUT_S "")"
HARD_TIMEOUT_OVR="$(cfg HARD_TIMEOUT_S "")"
BUDGET_OVR="$(cfg IJ_BUDGET_S "")"

PORT="$(cfg PORT 8000)"                              # host port: --local only
DEPS_TIMEOUT_S="$(cfg DEPS_TIMEOUT_S 600)"           # 10 minutes for deps.txt
CANARY_MIN_MATCH="$(cfg CANARY_MIN_MATCH 8)"         # of 10
CANARY_TIMEOUT_OVR="$(cfg CANARY_TIMEOUT_S "")"
CANARY_TIMEOUT_S="${CANARY_TIMEOUT_OVR:-120}"        # one canary request (a sealed launch adds the compile budget)
CANARY_MAX_TIMEOUTS=""                               # empty: the harness's own (two); a sealed launch sets 1
LOCK_MODE_PREF="$(cfg IJ_LOCK_MODE auto)"         # auto | flock | mkdir
KEEP_IMAGES="$(cfg IJ_KEEP_IMAGES 5)"             # inference-judge-exp:* images to keep
GIT_IMAGE="$(cfg IJ_GIT_IMAGE alpine/git:latest)" # sandbox for snapshot git
CLEANUP_IMAGE="$(cfg IJ_CLEANUP_IMAGE alpine:3)"  # sandbox for rm -rf as root
# Step 0 needs about ten launches (DESIGN 2.5); tests shrink it.
CALIBRATE_LAUNCHES="$(cfg CALIBRATE_LAUNCHES 10)"

# The interpreter. Fixed path, never inherited from the caller's PATH games,
# and always run with -I (isolated: no cwd on sys.path, no PYTHON* env, no user
# site-packages). -I also drops the script's own directory from sys.path, so
# bench modules are put back on it explicitly by PY_BOOT below - that is the
# only directory our python is ever allowed to import from.
if [[ -x "$SELF_DIR/.venv/bin/python" ]]; then
  PY_DEFAULT="$SELF_DIR/.venv/bin/python"
elif [[ -x /opt/bench/.venv/bin/python ]]; then
  PY_DEFAULT=/opt/bench/.venv/bin/python
else
  PY_DEFAULT="$(command -v python3 || echo /usr/bin/python3)"
fi
PY="$(cfg BENCH_PY "$PY_DEFAULT")"
PY_BOOT='import sys, runpy
sys.path.insert(0, sys.argv[1])
_script = sys.argv[2]
sys.argv = [_script] + sys.argv[3:]
runpy.run_path(_script, run_name="__main__")'

ID_RE='^[a-z0-9][a-z0-9-]{2,40}$'
DRY_RUN=0
LOCAL=0
MODE="score"            # score | instrument | calibrate | confirm | lock-ruler
ID=""
START_EPOCH="$(date -u +%s)"
STARTED_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID=""
CONTAINER=""            # the name prefix; each launch adds its label
CURRENT_CONTAINER=""
SUT_PID=""
TMPROOT=""
RESULTS_DIR=""
ARTIFACTS_DIR=""
EXP_DIR=""
VERDICT="INFRASTRUCTURE_FAILURE"
REASONS=""              # newline-separated, for results.json `reasons`
IMAGE=""
IMAGE_DIGEST=""
IS_BASELINE=0
CLEANUP_OK=true
DEPS_PRESENT=false
DEPS_INSTALLED=false
LOCK_MODE=""
LOCK_HELD=0
RULER_LOCK_STATE="absent"
MEAS_TIMEOUT=""         # when set, py/meas calls are bounded by it (HUNG)
LAUNCHER=""             # docker | local, from the target and the flags

usage() {
  cat <<'USAGE'
run.sh - measure one experiment and return a verdict.

usage: run.sh <experiment-id>                 (production, also under sudo)
       run.sh --instrument <experiment-id>    (unscored profiling run, sudo ok)
       run.sh --lock-ruler                    (clock start: lock the ruler)
       run.sh [--dry-run] [--local] <id>                     (operator/test)
       run.sh [--dry-run] [--local] --calibrate <run-id>     (operator)
       run.sh [--dry-run] [--local] --confirm <promoted-id>  (operator)

  <experiment-id>   directory name under research/experiments/,
                    matching ^[a-z0-9][a-z0-9-]{2,40}$
  --dry-run         print the commands, execute nothing
  --local           run launch.sh on the host, unsealed (tests)
  --instrument      one launch with /out mounted; verdict INSTRUMENT
  --calibrate       Step 0: launches of the incumbent -> sigma_L, MDE
  --confirm         Step 3: held-out ICI of a promoted run
  --lock-ruler      write RULER_LOCK (refuses if one exists)
  -h, --help        this text

phases: canary, warm sweep, blocks, screen, quality2, verdict
(ABBA or ICI against the incumbent, judged by stats.py).
ids starting with `baseline-` are reserved for the operator and need the
flag file $BENCH_ROOT/.allow-baseline.

The target is $BENCH_ROOT/TARGET (one line: a target id under targets/).
A run refuses to start when RULER_LOCK exists and the live ruler differs.

Flags are refused when the script is invoked through sudo, except
--instrument <id> and --lock-ruler. Environment overrides (BENCH_ROOT, PORT,
HARD_TIMEOUT_S, IJ_BUDGET_S, BENCH_TARGET, ...) are ignored there too.

exit codes: 0 for every verdict, 2 for bad usage, 70 for infrastructure.
USAGE
}

# `|| true`: a caller that went away takes its pipe with it, and a log line
# that cannot be written must not end the run under set -e.
log()  { printf '[run.sh %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2 || true; }
die()  { log "FATAL: $*"; exit 70; }
usage_error() { printf 'run.sh: %s\n' "$*" >&2; exit 2; }

# Every external command goes through this, so --dry-run can print instead of
# running, and so the log shows exactly what was executed.
run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: %s\n' "$*"
    return 0
  fi
  log "+ $*"
  "$@"
}

# python, isolated, importing only from BENCH_ROOT. MEAS_TIMEOUT, when set,
# bounds it (rc 124: the caller writes HUNG).
py() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: %s -I %s\n' "$PY" "$*"
    return 0
  fi
  log "+ python -I $*"
  if [[ -n "$MEAS_TIMEOUT" ]]; then
    with_timeout "$MEAS_TIMEOUT" "$PY" -I -c "$PY_BOOT" "$BENCH_ROOT" "$@"
    return $?
  fi
  "$PY" -I -c "$PY_BOOT" "$BENCH_ROOT" "$@"
}
py_capture() { "$PY" -I -c "$PY_BOOT" "$BENCH_ROOT" "$@"; }

# json_get FILE dotted.key - empty string when anything is missing.
json_get() {
  "$PY" -I -c 'import json, sys
try:
    value = json.load(open(sys.argv[1]))
except Exception:
    value = {}
for key in sys.argv[2].split("."):
    value = (value or {}).get(key) if isinstance(value, dict) else None
print("" if value is None else value)' "$1" "$2" 2>/dev/null || printf ''
}

# How long anything gets between SIGTERM and SIGKILL.
KILL_GRACE_S="$(cfg IJ_KILL_GRACE_S 10)"
TIMEOUT_BIN=""            # resolved once, on first use
TIMEOUT_FOREGROUND=""
resolve_timeout() {
  [[ -n "$TIMEOUT_BIN" ]] && return 0
  if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN=timeout
  elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN=gtimeout
  else TIMEOUT_BIN=none; return 0
  fi
  # GNU timeout runs the command in a *new* process group unless --foreground,
  # and a command in its own group survives the watchdog's group-wide kill and
  # keeps the lock descriptor it inherited. Where the flag exists, we use it.
  if "$TIMEOUT_BIN" --foreground 1 true >/dev/null 2>&1; then
    TIMEOUT_FOREGROUND="--foreground"
  fi
  return 0
}

# with_timeout SECONDS cmd... - GNU timeout if we have it, gtimeout on a Mac
# with coreutils, otherwise a plain bash timer. Returns the command's code,
# or 124 when it had to be killed.
#
# SIGTERM, then SIGKILL $KILL_GRACE_S later: a docker
# client that ignores TERM used to be waited on for ever, and the watchdog
# runs these calls itself after a TIMEOUT.
with_timeout() {
  local secs="$1"; shift
  resolve_timeout
  if [[ "$TIMEOUT_BIN" != "none" ]]; then
    # shellcheck disable=SC2086
    "$TIMEOUT_BIN" $TIMEOUT_FOREGROUND -k "$KILL_GRACE_S" "$secs" "$@"
  else
    local cmd_pid timer_pid rc=0 fired="${TMPDIR:-/tmp}/inference-judge-timeout.$$.$RANDOM"
    rm -f "$fired" 2>/dev/null || true
    "$@" & cmd_pid=$!
    # The timer takes its sleep with it when it is cancelled, and holds none
    # of the caller's pipes: an orphaned `sleep 900` kept the transcript pipe
    # open, and whoever was reading run.sh's output waited for it.
    ( sp=""
      trap '[[ -n "$sp" ]] && kill "$sp" 2>/dev/null; exit 0' TERM
      sleep "$secs" & sp=$!
      wait "$sp"
      : > "$fired" 2>/dev/null || true
      kill -TERM "$cmd_pid" 2>/dev/null || true
      sleep "$KILL_GRACE_S" & sp=$!
      wait "$sp"
      kill -KILL "$cmd_pid" 2>/dev/null || true ) > /dev/null 2>&1 & timer_pid=$!
    wait "$cmd_pid" || rc=$?
    kill "$timer_pid" 2>/dev/null || true
    wait "$timer_pid" 2>/dev/null || true
    if [[ -e "$fired" ]]; then rc=124; rm -f "$fired" 2>/dev/null || true; fi
    return "$rc"
  fi
}

# bounded SECONDS shell-function args... - what with_timeout is for external
# commands, for a shell function: `timeout` is a binary and cannot run one.
# The watchdog's own cleanup runs outside the body it monitors, so without this
# a wedged docker daemon could hang the watchdog after it had already killed
# the run, and the lock would never be released.
bounded() {
  local secs="$1"; shift
  local pid waited=0
  # traps reset: the parent ignores TERM on purpose, the cleanup must not.
  ( trap - INT TERM HUP PIPE; "$@" ) & pid=$!
  while (( waited < secs )) && kill -0 "$pid" 2>/dev/null; do
    sleep 1
    waited=$(( waited + 1 ))
  done
  if kill -0 "$pid" 2>/dev/null; then
    log "bounded: $1 still running after ${secs}s, terminating it"
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep "$KILL_GRACE_S"
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  return 0
}

check_deadline() {
  local now elapsed
  now="$(date -u +%s)"
  elapsed=$(( now - START_EPOCH ))
  if (( elapsed > BUDGET_S )); then
    log "budget of ${BUDGET_S}s gone after ${elapsed}s"
    end_run TIMEOUT
  fi
}

remaining_s() {
  local left=$(( BUDGET_S - ( $(date -u +%s) - START_EPOCH ) ))
  (( left < 1 )) && left=1
  printf '%s' "$left"
}

# ------------------------------------------------------------ 1. arguments --
parse_args() {
  if (( UNDER_SUDO )); then
    # The privileged entry point: a bare id, `--instrument <id>`, or
    # `--lock-ruler`. Nothing else, and no environment.
    case "$#:${1:-}" in
      1:--lock-ruler) MODE="lock-ruler"; return 0 ;;
      2:--instrument) MODE="instrument"; ID="$2" ;;
      1:*) ID="$1" ;;
      *) usage_error "expected exactly one experiment id (or --instrument <id>), got $# arguments" ;;
    esac
    [[ "$ID" =~ $ID_RE ]] || usage_error "bad id '$ID' (want $ID_RE); flags are not accepted under sudo"
  else
    while (( $# )); do
      case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --local) LOCAL=1; shift ;;
        --instrument|--profile) MODE="instrument"; shift ;;
        --calibrate) MODE="calibrate"; shift ;;
        --confirm) MODE="confirm"; shift ;;
        --lock-ruler) MODE="lock-ruler"; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) usage_error "unknown flag $1" ;;
        *) ID="$1"; shift ;;
      esac
    done
    [[ "$MODE" == "lock-ruler" ]] && return 0
    [[ -n "$ID" ]] || { usage >&2; usage_error "no experiment id given"; }
    [[ "$ID" =~ $ID_RE ]] || usage_error "bad id '$ID' (want $ID_RE)"
    # A baseline is the reference, never a promotion: nothing to confirm.
    if [[ "$MODE" == "confirm" && "$ID" == baseline-* ]]; then
      usage_error "--confirm $ID: a baseline-* run is the reference, not a promoted run; there is nothing to confirm"
    fi
  fi

  RESULTS_DIR="$BENCH_ROOT/results/$ID"
  # A confirmation belongs to the run it confirms: results/<id>/confirm/.
  [[ "$MODE" == "confirm" ]] && RESULTS_DIR="$BENCH_ROOT/results/$ID/confirm"
  ARTIFACTS_DIR="$RESULTS_DIR/artifacts"
  CONTAINER="inference-judge-exp-$ID"
  if [[ -e "$RESULTS_DIR" && "$DRY_RUN" != "1" ]]; then
    usage_error "$RESULTS_DIR already exists; pick a new id"
  fi
  if [[ "$MODE" == "confirm" && ! -d "$BENCH_ROOT/results/$ID" && "$DRY_RUN" != "1" ]]; then
    usage_error "--confirm $ID: there is no finished run $ID to confirm"
  fi
  EXP_DIR="$RESEARCH_ROOT/experiments/$ID"
  if [[ "$MODE" != "calibrate" && "$MODE" != "confirm" && ! -d "$EXP_DIR" \
        && "$DRY_RUN" != "1" ]]; then
    die "no experiment at $EXP_DIR"
  fi
  [[ -f "$EXP_DIR/deps.txt" ]] && DEPS_PRESENT=true || DEPS_PRESENT=false
  RUN_ID="$ID-$(date -u +%Y%m%dT%H%M%SZ)-$$"
}

# ------------------------------------------------------------- the target --
# $BENCH_ROOT/TARGET names the target (one line, an id under targets/). Tests
# may name one with BENCH_TARGET instead; through sudo only the file counts,
# and without it the run is refused rather than guessed.
TARGET_FILE=""
resolve_target() {
  local want dir
  want="$(cfg BENCH_TARGET "")"
  if [[ -z "$want" && -f "$BENCH_ROOT/TARGET" ]]; then
    want="$(head -n 1 "$BENCH_ROOT/TARGET" | tr -d '[:space:]')"
  fi
  if [[ -z "$want" ]]; then
    (( UNDER_SUDO )) && die "no target: $BENCH_ROOT/TARGET is missing (bootstrap writes it)"
    want="cpu-mock"
  fi
  if [[ "$want" == */* ]]; then
    (( UNDER_SUDO )) && die "TARGET must be a target id, not a path"
    TARGET_FILE="$want"
  else
    [[ "$want" =~ ^[a-z0-9][a-z0-9-]{1,40}$ ]] || die "bad target id '$want'"
    for dir in "$BENCH_ROOT/targets" "$SELF_DIR/../targets"; do
      if [[ -f "$dir/$want.yaml" ]]; then TARGET_FILE="$dir/$want.yaml"; break; fi
    done
  fi
  [[ -n "$TARGET_FILE" && -f "$TARGET_FILE" ]] || die "target '$want' not found under $BENCH_ROOT/targets"
}

# budgets.run_s is what one candidate run gets: the design's full launches
# (abba four, ici three) and the short screen launch. A calibration is
# CALIBRATE_LAUNCHES full launches under the same watchdog, so it gets the
# same share for each of them, and never less than one run's. A baseline is
# one full launch (B) under a run's budget, and for a real engine the other
# cold launches of the reference R after it (reference_finish): each of those
# is a ready wait and one pass over the quality prompts, and adds the limits
# of both.
run_budget_s() {
  local per=4 s="$T_RUN_S"
  [[ "${T_DESIGN:-}" == "ici" ]] && per=3
  if [[ "$MODE" == "calibrate" ]] && (( CALIBRATE_LAUNCHES > per )); then
    s=$(( (T_RUN_S * CALIBRATE_LAUNCHES + per - 1) / per ))
  fi
  if (( IS_BASELINE )) && [[ -n "$T_REF_LAUNCHES" ]]; then
    s=$(( T_RUN_S + (T_REF_LAUNCHES - 1) * (HEALTH_TIMEOUT_S + $(quality_limit_s)) ))
  fi
  printf '%s' "$s"
}

# All of one pass over the quality prompts (a gate, or one launch of R): 900 s
# and the compile budget, for the shapes the warm sweep did not compile.
quality_limit_s() {
  printf '%s' "${QUALITY_TIMEOUT_S:-$(( 900 + T_COMPILE_S ))}"
}

# The watchdog's limit for this run. Asked again once the id is known to be
# the operator's baseline (reserved_id_check).
set_budget() {
  HARD_TIMEOUT_S="${HARD_TIMEOUT_OVR:-$(run_budget_s)}"
  BUDGET_S="${BUDGET_OVR:-$HARD_TIMEOUT_S}"
}

# The target as shell variables (T_*), through target.py's validated loader.
load_target_settings() {
  local vars
  vars="$("$PY" -I -c "$PY_BOOT" "$BENCH_ROOT" "$BENCH_ROOT/target.py" shell \
            "$TARGET_FILE" "$BENCH_ROOT" 2>&1)" || die "target $TARGET_FILE: $vars"
  eval "$vars"
  (( T_RUNNABLE )) || die "target $T_ID is a sketch (runnable: false); nothing can be measured on it"
  MODEL_NAME="${MODEL_NAME_OVR:-$T_SERVED_NAME}"
  BASE_IMAGE="${BASE_IMAGE_OVR:-$T_IMAGE}"
  IMAGE="$BASE_IMAGE"
  HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_OVR:-$T_READY_S}"   # target.py: health.ready_timeout_s
  BUILD_TIMEOUT_S="${BUILD_TIMEOUT_OVR:-$T_BUILD_S}"
  set_budget
  LEADERBOARD_TPS="$BENCH_ROOT/leaderboard/$T_ID.json"
  CALIBRATION="$BENCH_ROOT/calibration/$T_ID.json"
  CANARY_REF="$BENCH_ROOT/canary/$T_ID.json"
  if (( LOCAL )) && [[ "$T_ENGINE" != "mock" ]]; then
    usage_error "--local runs launch.sh on this host; only the mock engine may"
  fi
  LAUNCHER="docker"
  (( LOCAL )) && LAUNCHER="local"
  return 0
}

# --------------------------------------------------------- 1b. ruler lock --
# DESIGN 2.6: the driver locks the ruler at clock start. A run whose live
# ruler (the bench files and the target) differs from the lock is refused,
# in --dry-run too, before anything is measured or written.
ruler_lock_check() {
  if [[ ! -e "$RULER_LOCK_PATH" ]]; then
    RULER_LOCK_STATE="absent"
    return 0
  fi
  local msg
  if msg="$("$PY" -I -c "$PY_BOOT" "$BENCH_ROOT" "$BENCH_ROOT/target.py" check-lock \
              "$BENCH_ROOT" "$TARGET_FILE" "$RULER_LOCK_PATH" 2>&1)"; then
    RULER_LOCK_STATE="matched"
    log "ruler lock: $msg"
    return 0
  fi
  die "refusing to run: $msg"
}

lock_ruler() {
  local msg
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: %s -I target.py lock %s %s %s\n' "$PY" "$BENCH_ROOT" "$TARGET_FILE" "$RULER_LOCK_PATH"
    exit 0
  fi
  msg="$("$PY" -I -c "$PY_BOOT" "$BENCH_ROOT" "$BENCH_ROOT/target.py" lock \
           "$BENCH_ROOT" "$TARGET_FILE" "$RULER_LOCK_PATH" 2>&1)" \
    || usage_error "$msg"
  printf 'RULER_LOCK %s %s\n' "$msg" "$RULER_LOCK_PATH"
  exit 0
}

# baseline-* is the operator's, not the researcher's. The operator drops
# .allow-baseline next to run.sh before measuring the baseline and removes it
# afterwards (docs/RUNBOOK.md); without that file the id is simply refused.
reserved_id_check() {
  [[ "$ID" == baseline-* ]] || return 0
  [[ "$MODE" == "calibrate" ]] && return 0
  if [[ -e "$BENCH_ROOT/.allow-baseline" ]]; then
    IS_BASELINE=1
    set_budget
    log "reserved id $ID accepted: $BENCH_ROOT/.allow-baseline is present"
    return 0
  fi
  log "reserved id $ID refused: no $BENCH_ROOT/.allow-baseline"
  add_reason "baseline-* ids are reserved for the operator"
  VERDICT="REJECTED_RESERVED_ID"
  finish
  announce
  exit 0
}

prepare_dirs() {
  run_cmd mkdir -p "$RESULTS_DIR" "$ARTIFACTS_DIR" "$RESULTS_DIR/snapshot" \
                   "$BENCH_ROOT/leaderboard" "$BENCH_ROOT/canary"
  if [[ "$DRY_RUN" != "1" ]]; then
    chmod 755 "$RESULTS_DIR" 2>/dev/null || true
    chmod 777 "$ARTIFACTS_DIR" 2>/dev/null || true
  fi
}

add_reason() {
  REASONS="${REASONS:+$REASONS
}$*"
}

# --------------------------------------------------------------- 2. lock ----
# One experiment at a time: one GPU, one port, one leaderboard. flock where it
# exists (Linux), a mkdir lock where it does not (macOS ships no flock).
acquire_lock() {
  local lockfile="$BENCH_ROOT/.lock" lockdir="$BENCH_ROOT/.lock.d" mode="$LOCK_MODE_PREF"
  if [[ "$mode" == "auto" ]]; then
    # quoted so shellcheck does not read `mode=mkdir` as a missing $(...)
    if command -v flock >/dev/null 2>&1; then mode="flock"; else mode="mkdir"; fi
  fi
  # DESIGN 2.1: the lock's path is published in a target-independent file,
  # so the driver can tell a busy bench without knowing how it locks.
  local published="$lockfile"
  [[ "$mode" == "mkdir" ]] && published="$lockdir"
  if [[ "$(cat "$BENCH_ROOT/LOCK_PATH" 2>/dev/null || true)" != "$published" ]]; then
    printf '%s\n' "$published" > "$BENCH_ROOT/LOCK_PATH.tmp.$$" 2>/dev/null \
      && mv "$BENCH_ROOT/LOCK_PATH.tmp.$$" "$BENCH_ROOT/LOCK_PATH" 2>/dev/null || true
  fi
  if [[ "$mode" == "flock" ]]; then
    exec 9>"$lockfile"
    flock -n 9 || return 1
  else
    if ! mkdir "$lockdir" 2>/dev/null; then
      # a lock whose owner is gone is not a lock
      local owner=""
      [[ -f "$lockdir/pid" ]] && owner="$(cat "$lockdir/pid" 2>/dev/null || true)"
      if [[ -n "$owner" ]] && ! kill -0 "$owner" 2>/dev/null; then
        log "removing stale lock from pid $owner"
        rm -rf "$lockdir"
        mkdir "$lockdir" 2>/dev/null || return 1
      else
        return 1
      fi
    fi
    printf '%s\n' "$$" > "$lockdir/pid" 2>/dev/null || true
  fi
  LOCK_MODE="$mode"
  LOCK_HELD=1
  return 0
}

release_lock() {
  (( LOCK_HELD )) || return 0
  [[ "$LOCK_MODE" == "mkdir" ]] && rm -rf "$BENCH_ROOT/.lock.d" 2>/dev/null || true
  LOCK_HELD=0
  return 0
}


# --------------------------------------------------- 3-6. the Docker parts --
# resolve_source PATH - echo the real path, but only if it is under $SRC_ROOT.
# A researcher who replaces ~/vllm-src with a symlink to /opt/bench must not
# get the bench's own files copied into their container.
resolve_source() {
  local p="$1" rp
  rp="$(readlink -f "$p" 2>/dev/null || true)"
  [[ -n "$rp" ]] || { log "cannot resolve $p"; return 1; }
  case "$rp" in
    "$SRC_ROOT"|"$SRC_ROOT"/*) printf '%s' "$rp"; return 0 ;;
    *) log "refusing $p: it resolves to $rp, outside $SRC_ROOT"; return 1 ;;
  esac
}

# safe_rsync SRC/ DST/ - never follow a link out of the tree, never copy
# devices, sockets or fifos.
safe_rsync() {
  run_cmd rsync -a --delete --safe-links --no-specials --no-devices "$1" "$2"
}

# source_dir NAME - the researcher's checkout of one of the target's engine
# trees (engine.sources): <src root>/NAME-src.
source_dir() { printf '%s' "$SRC_ROOT/$1-src"; }

copy_experiment() {
  if [[ "$DRY_RUN" == "1" ]]; then
    local name trees=""
    TMPROOT="<private-tmp>"
    for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do trees+=", $(source_dir "$name")"; done
    printf 'WOULD RUN: check realpath of %s%s all lie under %s, refuse otherwise\n' \
      "$EXP_DIR" "$trees" "$SRC_ROOT"
    printf 'WOULD RUN: rsync -a --delete --safe-links --no-specials --no-devices %s/ %s/exp/\n' \
      "$EXP_DIR" "$TMPROOT"
    copy_engines
    return 0
  fi
  # normally created by main() before the body forks, so that the watchdog can
  # still clean it up after a TIMEOUT
  [[ -n "$TMPROOT" && -d "$TMPROOT" ]] || TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/inference-judge-$ID-XXXXXX")"
  local exp
  exp="$(resolve_source "$EXP_DIR")"     || die "experiment directory is not under $SRC_ROOT"
  run_cmd mkdir -p "$TMPROOT/exp"
  safe_rsync "$exp/" "$TMPROOT/exp/"
  copy_engines
}

# The engine trees mounted (read-only) into every launch. A missing tree is an
# empty one for the launchers that never read it (the mock engine, --local).
copy_engines() {
  local name src rp
  if [[ "$DRY_RUN" == "1" ]]; then
    TMPROOT="${TMPROOT:-<private-tmp>}"
    for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
      printf 'WOULD RUN: rsync -a --delete --safe-links --no-specials --no-devices %s/ %s/src/%s/\n' \
        "$(source_dir "$name")" "$TMPROOT" "$name"
    done
    return 0
  fi
  [[ -n "$TMPROOT" && -d "$TMPROOT" ]] || TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/inference-judge-$ID-XXXXXX")"
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    src="$(source_dir "$name")"
    run_cmd mkdir -p "$TMPROOT/src/$name"
    if [[ ! -e "$src" && ( "$T_ENGINE" == "mock" || "$LAUNCHER" != "docker" ) ]]; then
      log "no $name source at $src: mounting an empty tree"
      continue
    fi
    rp="$(resolve_source "$src")" || die "$name source is not under $SRC_ROOT"
    safe_rsync "$rp/" "$TMPROOT/src/$name/"
  done
}

# A promoted run keeps the engine trees it was measured with (without .git),
# next to its experiment snapshot. The next comparison starts the incumbent
# from them: with today's checkouts on both sides an engine edit would be
# compared with itself, and a reverted checkout would slow the incumbent.
# Called before the workspace is cleaned up.
keep_engine_trees() {
  [[ "$VERDICT" == "PROMOTED" || "$VERDICT" == "BASELINE" ]] || return 0
  local name keep="$RESULTS_DIR/snapshot/src"
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    [[ "$DRY_RUN" == "1" || -d "$TMPROOT/src/$name" ]] || continue
    run_cmd mkdir -p "$keep/$name"
    run_cmd rsync -a --delete --safe-links --no-specials --no-devices --exclude .git \
      "$TMPROOT/src/$name/" "$keep/$name/"
  done
}

# src_stage SUT_EXP - the engine trees that go with an experiment copy: the
# entry's own for <which>-exp when board_entry found them, today's otherwise.
src_stage() {
  local own="${1%-exp}-src"
  if [[ "$1" == *-exp && ( "$DRY_RUN" == "1" || -d "$own" ) ]]; then
    printf '%s' "$own"
  else
    printf '%s' "$TMPROOT/src"
  fi
}

# The /src copies are deleted at the end of the run (they are huge). What is
# kept is the experiment directory and the engine diffs: enough to rebuild the
# winner, small enough to keep forever.
snapshot_step() {
  local snap="$RESULTS_DIR/snapshot" name src
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: rsync -a --safe-links %s/ %s/exp/\n' "$EXP_DIR" "$snap"
    for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
      src="$(source_dir "$name")"
      printf 'WOULD RUN: docker run --rm --network none -u <owner of %s> -v %s:/r:ro %s ... diff HEAD > %s/%s.patch ; ... rev-parse HEAD > %s/%s.head ; ... ls-files --others > %s/%s.untracked\n' \
        "$src" "$src" "$GIT_IMAGE" "$snap" "$name" "$snap" "$name" "$snap" "$name"
    done
    return 0
  fi
  local exp
  if exp="$(resolve_source "$EXP_DIR")" && [[ -d "$exp" ]]; then
    run_cmd mkdir -p "$snap/exp"
    safe_rsync "$exp/" "$snap/exp/"
  fi
  run_cmd mkdir -p "$snap"
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    snapshot_repo "$(source_dir "$name")" "$name"
  done
}

# owner_of PATH - "uid:gid" of PATH. GNU stat on the box, BSD stat on a Mac.
owner_of() {
  stat -c '%u:%g' "$1" 2>/dev/null || stat -f '%u:%g' "$1" 2>/dev/null || true
}

# sandboxed_git REPO ARGS... - run git against a researcher-owned checkout
# without ever running it as bench.
#
# git reads the checkout's own config and attributes, and several of those
# knobs run arbitrary commands: core.fsmonitor, diff.external, *.textconv,
# clean/smudge filters. The checkout belongs to `researcher`, and bench is in the
# docker group, so `git -C <the researcher's repo> diff` as bench is a direct path from
# a researcher-written file to root on the host. So git runs in a throwaway
# container instead: no network, the checkout mounted read-only, and as the
# checkout's own owner uid, so the worst a hostile config can reach is a
# container that is deleted a second later. --no-ext-diff, --no-textconv and
# core.fsmonitor=false are belt and braces on top of that.
#
# A read-only mount is enough: `git diff` only refreshes the index
# opportunistically and gives up quietly when it cannot take the index lock
# (verified against alpine/git with staged, unstaged and untracked changes).
sandboxed_git() {
  local rp="$1"; shift
  with_timeout 120 docker run --rm --network none \
    -u "$(owner_of "$rp")" -v "$rp:/r:ro" -w /r \
    -e HOME=/tmp -e GIT_CONFIG_NOSYSTEM=1 \
    --entrypoint git "$GIT_IMAGE" \
    -c safe.directory='*' -c core.fsmonitor=false --no-pager "$@"
}

# What a snapshot is worth depends on it being complete, so we take staged and
# unstaged changes (`diff HEAD`) plus the list of untracked files.
# Snapshotting is bookkeeping, never a gate: every failure here is written into
# the .head file and the run carries on.
snapshot_repo() {
  local src="$1" name="$2" snap="$RESULTS_DIR/snapshot" rp
  : > "$snap/$name.patch"
  : > "$snap/$name.untracked"
  if ! rp="$(resolve_source "$src")"; then
    printf 'not snapshotted: %s did not resolve to a path under %s\n' \
      "$src" "$SRC_ROOT" > "$snap/$name.head"
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1 \
     || ! with_timeout 60 docker image inspect "$GIT_IMAGE" >/dev/null 2>&1; then
    printf 'not snapshotted: alpine/git image missing (%s)\n' "$GIT_IMAGE" \
      > "$snap/$name.head"
    log "snapshot $name: skipped, $GIT_IMAGE is not present on this host"
    return 0
  fi
  if ! sandboxed_git "$rp" rev-parse --git-dir >/dev/null 2>&1; then
    printf 'not a git repository: %s\n' "$rp" > "$snap/$name.head"
    return 0
  fi
  sandboxed_git "$rp" rev-parse HEAD > "$snap/$name.head" 2>/dev/null \
    || printf 'unknown\n' > "$snap/$name.head"
  sandboxed_git "$rp" diff --no-ext-diff --no-textconv HEAD \
    > "$snap/$name.patch" 2>/dev/null || true
  sandboxed_git "$rp" ls-files --others --exclude-standard \
    > "$snap/$name.untracked" 2>/dev/null || true
  log "snapshot $name: $(head -c 40 "$snap/$name.head" | tr -d '\n'), $(wc -c < "$snap/$name.patch" | tr -d ' ') bytes of diff, $(wc -l < "$snap/$name.untracked" | tr -d ' ') untracked files"
}


build_image() {
  if [[ ! -f "$EXP_DIR/Dockerfile" && "$DRY_RUN" != "1" ]]; then
    IMAGE="$BASE_IMAGE"
    log "experiment brought no Dockerfile, using $IMAGE"
  else
    IMAGE="inference-judge-exp:$ID"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'WOULD RUN: timeout %ss docker build --label inference-judge.experiment=%s -t %s -f %s/exp/Dockerfile %s/exp  (log to %s/build.log)\n' \
        "$BUILD_TIMEOUT_S" "$ID" "$IMAGE" "$TMPROOT" "$TMPROOT" "$RESULTS_DIR"
    else
      log "+ docker build -t $IMAGE (log: $RESULTS_DIR/build.log, limit ${BUILD_TIMEOUT_S}s)"
      if ! with_timeout "$BUILD_TIMEOUT_S" \
           docker build --label "inference-judge.experiment=$ID" -t "$IMAGE" \
           -f "$TMPROOT/exp/Dockerfile" "$TMPROOT/exp" \
           > "$RESULTS_DIR/build.log" 2>&1; then
        log "image build failed or timed out, see build.log"
        add_reason "docker build failed or ran over ${BUILD_TIMEOUT_S}s (build.log)"
        end_run BUILD_FAILED
      fi
    fi
  fi
  deps_image
  record_image_digest
}

# deps.txt is baked into a derived image at build time. The build has the
# network; the scored launch never does, so nothing can be fetched at launch.
deps_dockerfile() {
  cat <<DOCKERFILE
FROM $IMAGE
COPY deps.txt /tmp/inference-judge-deps.txt
RUN if command -v uv >/dev/null 2>&1 && [ -x "$CONTAINER_VENV/bin/python" ]; then \\
      uv pip install --python "$CONTAINER_VENV/bin/python" -r /tmp/inference-judge-deps.txt; \\
    elif command -v uv >/dev/null 2>&1; then \\
      uv pip install --system -r /tmp/inference-judge-deps.txt; \\
    else \\
      pip install -r /tmp/inference-judge-deps.txt; \\
    fi
DOCKERFILE
}

deps_image() {
  [[ "$DEPS_PRESENT" == "true" || "$DRY_RUN" == "1" ]] || return 0
  local tag="inference-judge-exp:$ID-deps" ctx="$TMPROOT/deps-build"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: if %s/deps.txt exists: timeout %ss docker build -t %s from %s with deps.txt installed into %s (log to %s/deps.log)\n' \
      "$EXP_DIR" "$DEPS_TIMEOUT_S" "$tag" "$IMAGE" "$CONTAINER_VENV" "$RESULTS_DIR"
    DEPS_INSTALLED="$DEPS_PRESENT"
    return 0
  fi
  # --local runs launch.sh on this host: there is no image to bake deps.txt into
  [[ "$LAUNCHER" == "docker" ]] || return 0
  mkdir -p "$ctx"
  cp "$TMPROOT/exp/deps.txt" "$ctx/deps.txt"
  deps_dockerfile > "$ctx/Dockerfile"
  log "+ docker build -t $tag (deps.txt, log: $RESULTS_DIR/deps.log, limit ${DEPS_TIMEOUT_S}s)"
  if ! with_timeout "$DEPS_TIMEOUT_S" \
       docker build --label "inference-judge.experiment=$ID" -t "$tag" "$ctx" \
       > "$RESULTS_DIR/deps.log" 2>&1; then
    log "deps.txt install failed or timed out, see deps.log"
    add_reason "deps.txt did not install within ${DEPS_TIMEOUT_S}s (deps.log)"
    end_run BUILD_FAILED
  fi
  IMAGE="$tag"
  DEPS_INSTALLED=true
}

record_image_digest() {
  [[ "$DRY_RUN" == "1" ]] && { IMAGE_DIGEST="sha256:dry-run"; return 0; }
  [[ "$LAUNCHER" == "docker" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
  [[ -n "$IMAGE_DIGEST" ]] && log "image digest $IMAGE_DIGEST"
  return 0
}

# The container entry point. Every mount is read-only, so the experiment and
# the engine sources are copied into the container's own filesystem first:
# launch.sh still finds a writable /exp and /src/<engine>, and nothing it
# writes there can reach the host.
container_entry() {
  cat <<'ENTRY'
set -eu
mkdir -p /exp /src
cp -a /ro/exp/. /exp/
for d in /ro/src/*/; do
  if [ -d "$d" ]; then e="$(basename "$d")"; mkdir -p "/src/$e" && cp -a "$d." "/src/$e/"; fi
done
cd /exp
exec bash /exp/launch.sh
ENTRY
}

# The engine's compile cache (engine.compile_cache). The seed belongs to the
# ruler; a launch gets a copy, writable, and nothing a candidate writes there
# reaches the seed or an incumbent's launch. An incumbent's launch has its own
# copy. The candidate has one for its launches of one run (KEPT_STAGE: S, C1
# and C2), so it compiles its shapes once and not three times; because it can
# carry anything else across in there too, no prompt is sent to it twice
# (retire_prompts). The copy goes with the run's workspace.
CACHE_STAGE=""
KEPT_STAGE=""
KEPT_STAGED=0
cache_stage() {
  CACHE_STAGE=""
  [[ -n "$T_CACHE_MOUNT" && "$LAUNCHER" == "docker" ]] || return 0
  CACHE_STAGE="$TMPROOT/cache/$1"
  if [[ "$1" == "$KEPT_STAGE" ]]; then
    (( KEPT_STAGED )) && return 0
    KEPT_STAGED=1
  fi
  run_cmd mkdir -p "$CACHE_STAGE"
  if [[ "$DRY_RUN" == "1" || -d "$T_CACHE_SEED" ]]; then
    run_cmd rsync -a "$T_CACHE_SEED/" "$CACHE_STAGE/"
  else
    log "no compile cache seed at $T_CACHE_SEED: the launch starts with an empty cache"
  fi
}

# cache_settle LABEL ROLE [STAGE] - after the launch has stopped. What an
# incumbent (or the operator's baseline) compiled is added to the seed, never
# replacing a file that is there; a candidate's cache is never added, and is
# thrown away here unless it is the one kept for its next launch.
cache_settle() {
  local label="$1" role="$2" name="${3:-$1}"
  local stage="$TMPROOT/cache/$name"
  [[ -n "$T_CACHE_MOUNT" && "$LAUNCHER" == "docker" ]] || return 0
  [[ "$name" == "$KEPT_STAGE" ]] && return 0
  if [[ "$role" == "incumbent" || "$label" == "B" ]]; then
    if [[ "$DRY_RUN" == "1" \
          || "$(json_get "$RESULTS_DIR/launches/$label/launch.json" status)" == "ok" ]]; then
      run_cmd mkdir -p "$T_CACHE_SEED" || true
      run_cmd rsync -a --ignore-existing "$stage/" "$T_CACHE_SEED/" \
        || log "could not add launch $label's compiled shapes to $T_CACHE_SEED"
    fi
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "WOULD RUN: docker run --rm --network none -v %s/cache:/s alpine rm -rf /s/%s\n" "$TMPROOT" "$name"
    return 0
  fi
  with_timeout 180 docker run --rm --network none -v "$TMPROOT/cache:/s" "$CLEANUP_IMAGE" \
    rm -rf "/s/$name" >/dev/null 2>&1 || true
  CACHE_STAGE=""
}

# The seal of a launch (DESIGN 2.6, "The seal"): no network, no
# published port, no writable mount but the private compile cache. GPUs only
# when the target has them; devices and accelerator flags only from the
# target's own docker_flags. --instrument alone gets /out.
SEAL_ARGS=()
seal_args() {
  local sut_exp="$1" f has_net=0 name trees
  trees="$(src_stage "$sut_exp")"
  SEAL_ARGS=()
  for f in ${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"}; do
    case "$f" in --network*|--net=*|--net) has_net=1 ;; esac
  done
  (( has_net )) || SEAL_ARGS+=(--network none)
  [[ "$T_ACCEL" == "gpu" ]] && SEAL_ARGS+=(--gpus all)
  [[ -n "$T_SHM" ]] && SEAL_ARGS+=(--shm-size "$T_SHM")
  SEAL_ARGS+=(${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"})
  [[ -n "$T_MODEL_DIR" ]] && SEAL_ARGS+=(-v "$MODELS_DIR:/opt/models:ro")
  SEAL_ARGS+=(-v "$sut_exp:/ro/exp:ro")
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    SEAL_ARGS+=(-v "$trees/$name:/ro/src/$name:ro")
  done
  [[ -n "$CACHE_STAGE" ]] && SEAL_ARGS+=(-v "$CACHE_STAGE:$T_CACHE_MOUNT:rw")
  [[ "$T_ENGINE" == "mock" ]] && SEAL_ARGS+=(-v "$BENCH_ROOT/mock_server.py:/opt/inference-judge/mock_server.py:ro")
  [[ "$MODE" == "instrument" ]] && SEAL_ARGS+=(-v "$ARTIFACTS_DIR:/out:rw")
  SEAL_ARGS+=(-e "IJ_EXPERIMENT_ID=$ID" -e "IJ_TARGET=$T_ID"
              -e "IJ_SERVED_NAME=$MODEL_NAME" -e "IJ_MODEL_DIR=$T_MODEL_DIR"
              -e "PORT=$T_PORT")
  for f in ${T_ENV[@]+"${T_ENV[@]}"}; do SEAL_ARGS+=(-e "$f"); done
  return 0
}

# start_container LABEL SUT_EXP IMAGE [STAGE] - one sealed launch, on its own
# copy of the compile cache unless STAGE names another.
start_container() {
  local label="$1" sut_exp="$2" image="$3"
  CURRENT_CONTAINER="$CONTAINER${label:+-$label}"
  cache_stage "${4:-${label:-main}}"
  seal_args "$sut_exp"
  if [[ "$DRY_RUN" == "1" ]]; then
    # one line, flags before the image, every value without spaces: the plan
    # is what the scorecard replays to prove the seal (scenario 5)
    printf 'WOULD RUN: docker run -d --name %s --label inference-judge.experiment=%s' "$CURRENT_CONTAINER" "$ID"
    printf ' %q' "${SEAL_ARGS[@]}"
    printf ' %s bash -c <container_entry: copy /ro/exp and /ro/src/* into the container, exec bash /exp/launch.sh>\n' "$image"
    return 0
  fi
  if [[ "$LAUNCHER" == "local" ]]; then
    start_local "$label" "$sut_exp"
    return 0
  fi
  run_cmd docker run -d --name "$CURRENT_CONTAINER" \
    --label "inference-judge.experiment=$ID" \
    "${SEAL_ARGS[@]}" \
    "$image" bash -c "$(container_entry)" >/dev/null
}

# --local: the test launcher. launch.sh runs on this host as a child process,
# unsealed; refused through sudo with every other flag.
start_local() {
  local label="$1" sut_exp="$2" dir="$RESULTS_DIR/launches/${1:-main}"
  mkdir -p "$dir"
  ( cd "$sut_exp" && exec env PORT="$PORT" IJ_HOST=127.0.0.1 \
      IJ_EXPERIMENT_ID="$ID" IJ_TARGET="$T_ID" IJ_SERVED_NAME="$MODEL_NAME" \
      IJ_MOCK_SERVER="$BENCH_ROOT/mock_server.py" IJ_PY="$PY" \
      bash "$sut_exp/launch.sh" ) > "$dir/server.log" 2>&1 &
  SUT_PID=$!
  printf '%s\n' "$SUT_PID" > "$RESULTS_DIR/.sut.pid"
  log "local launch $label: pid $SUT_PID on port $PORT"
}

# The isolation probe: a throwaway container with exactly the launch's seal,
# running python from the ruler image. It tries the network and every mount,
# the target's own among them; what it managed goes into results.json
# `isolation`. --privileged, a host network or a host device from the target
# is a weaker seal, and so is any mount it could write that is not the
# private cache copy, and so is a unix socket under a mount the target brings
# (named in host_sockets: read-only does not stop a connect(), so it is a
# line to a host service). `seal` is "full" only when none of these is there.
# A privileged container, a device or a socket reaches the host whatever the
# probe saw, so with any of them a clean probe is "not_proven", never
# "blocked".
ISOLATION_JSON=""
isolation_probe() {
  local sut_exp="$1" out="$RESULTS_DIR/isolation.json" line notes="$T_SEAL_NOTES"
  if [[ "$LAUNCHER" != "docker" ]]; then
    ISOLATION_JSON="{\"network\": \"host\", \"writable_mounts\": \"n/a\", \"exfil_probe\": \"not_sealed\", \"seal\": \"none: $LAUNCHER launcher (test only)\"}"
    [[ "$DRY_RUN" == "1" ]] || printf '%s\n' "$ISOLATION_JSON" > "$out"
    return 0
  fi
  local f host mount prev="" sockets=() tmounts=()
  for f in ${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"}; do
    if [[ "$prev" == "-v" ]]; then
      host="${f%%:*}"; mount="${f#*:}"; mount="${mount%%:*}"
      tmounts+=("$mount")
      if [[ "$DRY_RUN" != "1" && -e "$host" ]]; then
        while IFS= read -r line; do
          [[ -n "$line" ]] && sockets+=("$mount${line#"$host"}")
        done < <(find "$host" -maxdepth 3 -type s 2>/dev/null | sort)
      fi
    fi
    prev="$f"
  done
  cache_stage probe
  seal_args "$sut_exp"
  local probe='import json, os, socket, sys
res = {}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close(); res["net"] = "ok"
except Exception as exc:
    res["net"] = "blocked"; res["net_err"] = type(exc).__name__
res["sockets"] = {}
for path in [x for x in os.environ.get("IJ_SOCKETS", "").splitlines() if x]:
    try:
        s = socket.socket(socket.AF_UNIX); s.settimeout(3); s.connect(path); s.close()
        res["sockets"][path] = "connected"
    except Exception as exc:
        res["sockets"][path] = type(exc).__name__
res["writable"] = []
for d in sys.argv[1:]:
    try:
        if os.path.isfile(d):       # a file mount: appending to it is the leak
            open(d, "a").close()
        elif d == os.environ.get("IJ_PRIVATE"):   # left there: run.sh looks for it
            open(os.path.join(d, ".inference-judge-private-probe"), "w").write("x")
        else:
            p = os.path.join(d, ".inference-judge-exfil-probe"); open(p, "w").write("x"); os.unlink(p)
        res["writable"].append(d)
    except Exception:
        pass
print("EXFIL " + json.dumps(res))'
  local targets=(/ro/exp) name reached=""
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do targets+=("/ro/src/$name"); done
  targets+=(${tmounts[@]+"${tmounts[@]}"})
  [[ -n "$CACHE_STAGE" ]] && targets+=("$T_CACHE_MOUNT")
  [[ -n "$T_MODEL_DIR" ]] && targets+=(/opt/models)
  # a file mount: /opt/inference-judge itself is the container's own layer, not the host
  [[ "$T_ENGINE" == "mock" ]] && targets+=(/opt/inference-judge/mock_server.py)
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: docker run --rm <the same seal flags> %s python3 -c <exfil probe: connect 1.1.1.1:53, write into %s> -> %s\n' \
      "$RULER_IMAGE" "${targets[*]}" "$out"
    return 0
  fi
  line="$(with_timeout 120 docker run --rm --label "inference-judge.experiment=$ID" \
            -e "IJ_PRIVATE=$T_CACHE_MOUNT" \
            -e "IJ_SOCKETS=$(printf '%s\n' ${sockets[@]+"${sockets[@]}"})" \
            "${SEAL_ARGS[@]}" "$RULER_IMAGE" python3 -c "$probe" "${targets[@]}" 2>&1 \
          | grep '^EXFIL ' | head -n 1 || true)"
  if [[ -n "$CACHE_STAGE" ]]; then
    reached=false
    [[ -e "$T_CACHE_SEED/.inference-judge-private-probe" ]] && reached=true
    cache_settle probe probe
  fi
  IJ_PROBE="${line#EXFIL }" IJ_NOTES="$notes" IJ_INSTRUMENT="$MODE" \
  IJ_PRIVATE="$T_CACHE_MOUNT" IJ_SEED="$T_CACHE_SEED" IJ_REACHED="$reached" \
  IJ_SOCKETS="$(printf '%s\n' ${sockets[@]+"${sockets[@]}"})" \
  IJ_PROBE_IMAGE="$RULER_IMAGE" "$PY" -I - "$out" <<'PYISO'
import json, os, sys
raw = os.environ.get("IJ_PROBE") or ""
try:
    probe = json.loads(raw)
except json.JSONDecodeError:
    probe = None
notes = [n for n in os.environ.get("IJ_NOTES", "").split(",") if n]
writable = ["/out"] if os.environ.get("IJ_INSTRUMENT") == "instrument" else []
reached = os.environ.get("IJ_REACHED")
private = {os.environ["IJ_PRIVATE"]: {
    "what": "a copy of the ruler's compile cache seed, made for this launch "
            "and deleted after it", "seed": os.environ.get("IJ_SEED"),
    "write_reached_seed": reached == "true"}} if reached else {}
sockets = [x for x in os.environ.get("IJ_SOCKETS", "").splitlines() if x]
# What the probe cannot vouch for: a privileged container, a host device and
# a host socket each reach the host by a road the probe does not walk. A
# socket counts by being there, listening or not (probe.sockets says which).
beyond = [n for n in notes if n == "privileged" or n.startswith("device ")] \
    + ["host socket " + x for x in sockets]
weaker = list(notes) + ["host socket " + x for x in sockets]
if probe is None:
    exfil = "unknown: the probe did not run (is %s built?)" % os.environ["IJ_PROBE_IMAGE"]
else:
    writable += probe.get("writable", [])
    shared = [m for m in probe.get("writable", []) if m not in private]
    weaker += ["writable mount " + m for m in shared]
    if reached == "true":
        weaker.append("the compile cache mount writes through to the seed")
    leaked = probe.get("net") == "ok" or shared or reached == "true"
    exfil = "leaked" if leaked else "blocked"
    if not leaked and beyond:
        exfil = ("not_proven: " + ", ".join(beyond) + " (the probe found "
                 "nothing, and these reach the host without it)")
out = {"network": "host" if "network=host" in notes else "none",
       "writable_mounts": writable, "exfil_probe": exfil,
       "seal": "weaker: " + ", ".join(weaker) if weaker else "full",
       "notes": notes, "probe": probe, "host_sockets": sockets}
if private:
    out["private_mounts"] = private
open(sys.argv[1], "w").write(json.dumps(out, indent=2))
print(json.dumps(out))
PYISO
}

wait_healthy() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: poll http://127.0.0.1:%s/v1/models until 200 (max %ss), from a ruler sidecar in the network namespace of %s\n' \
      "$T_PORT" "$HEALTH_TIMEOUT_S" "$CURRENT_CONTAINER"
    DEPS_INSTALLED="$DEPS_PRESENT"
    return 0
  fi
  local budget="$HEALTH_TIMEOUT_S" left t0
  left="$(remaining_s)"
  (( left < budget )) && budget="$left"
  t0="$(date -u +%s)"
  local deadline=$(( t0 + budget ))
  while (( $(date -u +%s) < deadline )); do
    if probe_health; then
      log "server healthy after $(( $(date -u +%s) - t0 ))s"
      [[ "$LAUNCHER" == "docker" ]] || DEPS_INSTALLED="$DEPS_PRESENT"
      return 0
    fi
    if ! sut_running; then
      log "the server exited before becoming healthy"
      break
    fi
    sleep "$( [[ "$LAUNCHER" == "local" ]] && echo 0.5 || echo 5 )"
  done
  return 1
}

# One health poll. Through the sidecar for a sealed launch: the server has no
# published port, so the ruler joins its network namespace to reach it.
probe_health() {
  local url="http://127.0.0.1:$T_PORT/v1/models"
  case "$LAUNCHER" in
    local) curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 ;;
    docker) with_timeout 30 docker run --rm --network "container:$CURRENT_CONTAINER" \
              --label "inference-judge.experiment=$ID" "$RULER_IMAGE" python3 -c \
              'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' \
              "$url" >/dev/null 2>&1 ;;
    *) return 1 ;;
  esac
}

sut_running() {
  case "$LAUNCHER" in
    local) [[ -n "$SUT_PID" ]] && kill -0 "$SUT_PID" 2>/dev/null ;;
    docker) docker inspect -f '{{.State.Running}}' "$CURRENT_CONTAINER" 2>/dev/null | grep -q true ;;
    *) return 0 ;;
  esac
}

# stop_container [LOGDIR] - stop the current launch, keep its log.
stop_container() {
  local logdir="${1:-$RESULTS_DIR}"
  if [[ "$DRY_RUN" == "1" ]]; then
    [[ -n "$CURRENT_CONTAINER" ]] || return 0
    printf 'WOULD RUN: docker logs %s > %s/server.log; docker rm -f %s\n' \
      "$CURRENT_CONTAINER" "$logdir" "$CURRENT_CONTAINER"
    CURRENT_CONTAINER=""
    return 0
  fi
  if [[ "$LAUNCHER" == "local" ]]; then
    if [[ -n "$SUT_PID" ]]; then
      kill -TERM "$SUT_PID" 2>/dev/null || true
      local i
      for i in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$SUT_PID" 2>/dev/null || break
        sleep 0.5
      done
      kill -KILL "$SUT_PID" 2>/dev/null || true
      wait "$SUT_PID" 2>/dev/null || true
    fi
    SUT_PID=""
    rm -f "$RESULTS_DIR/.sut.pid" 2>/dev/null || true
    return 0
  fi
  [[ -n "$CURRENT_CONTAINER" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  # Every docker call here is bounded: stop_container is
  # also what the watchdog runs after a TIMEOUT, and a watchdog that can hang
  # on a wedged docker daemon is not a watchdog.
  # server.log is written by the judge into the judge's own directory, never
  # read back out of the container's mount.
  mkdir -p "$logdir" 2>/dev/null || true
  with_timeout 60 docker logs "$CURRENT_CONTAINER" > "$logdir/server.log" 2>&1 || true
  with_timeout 60 docker stop -t 20 "$CURRENT_CONTAINER" >/dev/null 2>&1 || true
  with_timeout 120 docker rm -f "$CURRENT_CONTAINER" >/dev/null 2>&1 || true
  CURRENT_CONTAINER=""
  return 0
}

# Whatever this run started, by label: the watchdog and the EXIT trap run in
# the parent, which never saw the body's container names or pids.
stop_everything() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  local pid ids
  if [[ -f "$RESULTS_DIR/.sut.pid" ]]; then
    pid="$(cat "$RESULTS_DIR/.sut.pid" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$RESULTS_DIR/.sut.pid" 2>/dev/null || true
  fi
  [[ "$LAUNCHER" == "docker" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  ids="$(with_timeout 60 docker ps -aq --filter "label=inference-judge.experiment=$ID" 2>/dev/null || true)"
  [[ -n "$ids" ]] || return 0
  # shellcheck disable=SC2086
  with_timeout 120 docker rm -f $ids >/dev/null 2>&1 || true
  return 0
}

# Artifacts are whatever an --instrument run left in /out. The container is
# stopped before we touch them, links are dropped rather than followed, and
# nothing in here is ever used as an input to a gate.
collect_artifacts() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: drop symlinks and special files under %s (never follow them)\n' "$ARTIFACTS_DIR"
    return 0
  fi
  [[ -d "$ARTIFACTS_DIR" ]] || return 0
  find "$ARTIFACTS_DIR" -type l -print -delete 2>/dev/null || true
  find "$ARTIFACTS_DIR" ! -type d ! -type f -print -delete 2>/dev/null || true
  return 0
}

# Root inside the container leaves root-owned build trees in the private /src
# and /exp copies. bench cannot unlink those, but bench is in the docker group,
# so root in a throwaway container can. A cleanup failure is recorded, never
# promoted into a verdict.
cleanup_workspace() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "WOULD RUN: docker run --rm --network none -v %s:/s alpine sh -c 'rm -rf /s/* /s/.[!.]*' ; rmdir %s\n" \
      "${TMPROOT:-<tmp>}" "${TMPROOT:-<tmp>}"
    return 0
  fi
  [[ -n "${TMPROOT:-}" && -d "$TMPROOT" ]] || return 0
  if [[ "$LAUNCHER" == "docker" ]] && command -v docker >/dev/null 2>&1; then
    # bounded: the watchdog calls this too
    with_timeout 180 docker run --rm --network none -v "$TMPROOT:/s" "$CLEANUP_IMAGE" \
      sh -c 'rm -rf /s/* /s/.[!.]* 2>/dev/null; exit 0' >/dev/null 2>&1 || true
  fi
  rm -rf "$TMPROOT" 2>/dev/null || true
  if [[ -d "$TMPROOT" ]]; then
    CLEANUP_OK=false
    log "could not remove $TMPROOT (recorded as cleanup_ok:false)"
  fi
  return 0
}

# Custom experiment images accumulate at several GB each. Keep the newest few.
# The incumbent's image is tagged inference-judge-inc:* at promotion and is never pruned.
prune_images() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: keep the %s most recent inference-judge-exp:* images, docker rmi the rest (inference-judge-inc:* is kept)\n' "$KEEP_IMAGES"
    return 0
  fi
  [[ "$LAUNCHER" == "docker" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  local old
  old="$(with_timeout 60 docker images --filter 'reference=inference-judge-exp:*' \
           --format '{{.CreatedAt}}|{{.Repository}}:{{.Tag}}' 2>/dev/null \
         | sort -r | tail -n "+$(( KEEP_IMAGES + 1 ))" | cut -d'|' -f2 || true)"
  [[ -n "$old" ]] || return 0
  local img
  while read -r img; do
    [[ -n "$img" ]] || continue
    log "pruning old experiment image $img"
    with_timeout 120 docker rmi "$img" >/dev/null 2>&1 || true
  done <<< "$old"
  return 0
}

# ------------------------------------------------------- the measurement seam --
# meas SCRIPT ARGS... - a ruler script that talks to the server under test.
# A sealed launch has no published port, so the ruler runs in a sidecar that
# joins the server's network namespace: the bench files read-only and this
# run's results directory writable, both at their own paths, as bench's uid.
# The other launchers measure from this host.
meas() {
  if [[ "$LAUNCHER" != "docker" ]]; then
    py "$@"
    return $?
  fi
  local who
  who="$(id -u):$(id -g)"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: docker run --rm --network container:%s -u %s -v %s:%s:ro -v %s:%s %s python3 -I %s\n' \
      "${CURRENT_CONTAINER:-$CONTAINER}" "$who" "$BENCH_ROOT" "$BENCH_ROOT" \
      "$RESULTS_DIR" "$RESULTS_DIR" "$RULER_IMAGE" "$*"
    return 0
  fi
  local name="$CURRENT_CONTAINER-ruler-$RANDOM" rc=0
  local mounts=(-v "$BENCH_ROOT:$BENCH_ROOT:ro" -v "$RESULTS_DIR:$RESULTS_DIR")
  # the files a real target keeps outside the bench tree (the tokenizer in
  # the model directory, a pool somewhere else), each at its own path
  local f
  for f in "$TARGET_FILE" "$T_TOKENIZER" "$POOL" "$T_REF_PROMPTS" "$T_REF_TASKS"; do
    case "$f" in
      "$BENCH_ROOT"/*|"$RESULTS_DIR"/*) ;;
      /*) [[ -f "$f" ]] && mounts+=(-v "$f:$f:ro") ;;
    esac
  done
  log "+ [sidecar] python -I $*"
  if [[ -n "$MEAS_TIMEOUT" ]]; then
    with_timeout "$MEAS_TIMEOUT" docker run --rm --name "$name" \
      --network "container:$CURRENT_CONTAINER" --label "inference-judge.experiment=$ID" \
      -u "$who" -e HOME=/tmp "${mounts[@]}" "$RULER_IMAGE" \
      python3 -I -c "$PY_BOOT" "$BENCH_ROOT" "$@" || rc=$?
    (( rc == 124 )) && with_timeout 60 docker rm -f "$name" >/dev/null 2>&1
  else
    docker run --rm --name "$name" \
      --network "container:$CURRENT_CONTAINER" --label "inference-judge.experiment=$ID" \
      -u "$who" -e HOME=/tmp "${mounts[@]}" "$RULER_IMAGE" \
      python3 -I -c "$PY_BOOT" "$BENCH_ROOT" "$@" || rc=$?
  fi
  return "$rc"
}

# ---------------------------------------------------- 7. references, canary --
# The references (the canary and the quality2 R) are
# written by the operator's baseline run and by nobody else. They are staged
# in the run's own directory while it measures and installed together, each
# by a rename, at the very end of a run that finished: a baseline that dies
# half way leaves no half-written reference behind. An existing reference is
# never overwritten; to replace one, the operator deletes it first.
stage_ref() {
  local src="$1" name="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: stage %s as %s/staged/%s\n' "$src" "$RESULTS_DIR" "$name"
    return 0
  fi
  mkdir -p "$RESULTS_DIR/staged"
  cp -R "$src" "$RESULTS_DIR/staged/$name"
}

install_ref() {
  local src="$1" dst="$2" tmp
  [[ -e "$src" ]] || return 0
  if [[ -e "$dst" ]]; then
    log "reference $dst exists, kept (staged copy left in $src)"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  tmp="$dst.tmp.$$"
  rm -rf "$tmp"
  cp -R "$src" "$tmp"
  mv "$tmp" "$dst"
  log "reference installed: $dst"
}

commit_references() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: install %s/staged/{canary.json,reference} where no reference exists yet, each by a rename\n' "$RESULTS_DIR"
    return 0
  fi
  [[ -d "$RESULTS_DIR/staged" ]] || return 0
  install_ref "$RESULTS_DIR/staged/canary.json" "$CANARY_REF"
  install_ref "$RESULTS_DIR/staged/reference" "$T_REF_DIR"
}

# canary_step [DIR] - ten fixed prompts at temperature 0 against the stored
# reference. Only the baseline writes that reference; a production run that
# finds none is an infrastructure fault, not something to fill in.
canary_step() {
  local out="${1:-$RESULTS_DIR}" aborted="" limit=(--canary-timeout "$CANARY_TIMEOUT_S")
  [[ -n "$CANARY_MAX_TIMEOUTS" ]] && limit+=(--canary-max-timeouts "$CANARY_MAX_TIMEOUTS")
  if (( IS_BASELINE )) || [[ ! -f "$CANARY_REF" && "$DRY_RUN" != "1" ]]; then
    if (( UNDER_SUDO && ! IS_BASELINE )); then
      add_reason "no canary reference at $CANARY_REF: only the operator's baseline writes it"
      end_run INFRASTRUCTURE_FAILURE
    fi
    log "staging the canary reference instead of checking it"
    meas "$BENCH_ROOT/harness.py" --mode canary --base-url "$BASE_URL" \
         --model "$MODEL_NAME" \
         --canary-prompts "$BENCH_ROOT/canary/prompts.json" \
         "${limit[@]}" --out "$out"
    stage_ref "$out/canary.json" canary.json
    CANARY_MATCHED=10
    return 0
  fi
  meas "$BENCH_ROOT/harness.py" --mode canary --base-url "$BASE_URL" \
       --model "$MODEL_NAME" \
       --canary-prompts "$BENCH_ROOT/canary/prompts.json" \
       --canary-baseline "$CANARY_REF" \
       "${limit[@]}" --out "$out"
  if [[ "$DRY_RUN" == "1" ]]; then CANARY_MATCHED=10; return 0; fi
  CANARY_MATCHED="$(json_get "$out/canary.json" matched)"
  CANARY_MATCHED="${CANARY_MATCHED:-0}"
  log "canary matched $CANARY_MATCHED/10"
  if (( CANARY_MATCHED < CANARY_MIN_MATCH )); then
    # a wedged server: the harness stopped early and says so
    aborted="$(json_get "$out/canary.json" aborted)"
    [[ -n "$aborted" ]] && add_reason "canary stopped early: $aborted"
    add_reason "canary: $CANARY_MATCHED of 10 fixed prompts unchanged (need $CANARY_MIN_MATCH)"
    end_run CANARY_FAILED
  fi
}

# The stamps (DESIGN 2.6) of this run's ruler, compared by compare.py with
# the incumbent's before any comparison is made.
stamps_step() {
  py "$BENCH_ROOT/compare.py" stamps --bench-root "$BENCH_ROOT" \
     --target "$TARGET_FILE" --image-digest "${IMAGE_DIGEST:-none}" \
     --reference-dir "$T_REF_DIR" --out "$RESULTS_DIR/stamps.json"
}

# ---------------------------------------------------------- the launches --
# DESIGN 2.2-2.5. A verdict is drawn from whole launches: each one is a
# cold start of the server, health within the compile budget, one unscored
# warm sweep, then metric.blocks scored sweeps over the target's concurrency
# levels. compare.py reduces a launch to one value (log goodput per chip, or
# log whole-answer speed per chip for a mix of input sizes) and stats.py
# judges the launch values against the operator's calibration.
#
# With metric.screen_window_s the candidate is first launched alone (S): the
# warm sweep and one short window, screened against the incumbent's sentinel.
# A loser costs that one launch; the incumbent is never started for it.
SEED_BASE=""
DRAW_SEED=""                                  # the launch's draw: SEED_BASE, or its own
POOL=""
POOL_EXCLUDE=()
LAUNCH_FILES=()
QUALITY_FILES=()
SWEEP_TIMEOUT_S="$(cfg SWEEP_TIMEOUT_S "")"   # empty: derived from the schedule
WARM_TIMEOUT_S=""
SWEEP_WINDOW_S=""                             # the screen launch's shorter window
QUALITY_TIMEOUT_S="$(cfg QUALITY_TIMEOUT_S "")" # empty: 900 s and the compile budget
QUALITY_REQUEST_S=""                          # one quality request
LAUNCH_RECORDS=()                             # the sweeps of the launch just measured
ENTRY_ID=""
ENTRY_EXP=""
ENTRY_IMAGE=""

tps_setup() {
  SEED_BASE=$(( (RANDOM << 15 | RANDOM) % 1000000 ))
  DRAW_SEED="$SEED_BASE"
  # A sweep that outlives twice its own schedule plus two minutes is HUNG.
  # The warm sweep is also where a server that compiles a shape on first use
  # does its compiling, so it may take the compile budget on top.
  WARM_TIMEOUT_S="$SWEEP_TIMEOUT_S"
  if [[ -z "$SWEEP_TIMEOUT_S" ]]; then
    SWEEP_TIMEOUT_S="$(awk -v c="$T_CONCURRENCY" -v r="$T_RAMP_S" -v w="$T_WINDOW_S" \
                         'BEGIN { n = split(c, a, ","); printf "%d", 2 * n * (r + w) + 120 }')"
    WARM_TIMEOUT_S=$(( SWEEP_TIMEOUT_S + T_COMPILE_S ))
  fi
  # The canary and the quality prompts come after the warm sweep but need not
  # fall on a shape it compiled, so one request of theirs may wait as long as
  # a compile. A canary request that outlives even that is a dead server: one
  # such timeout ends the canary, not two.
  if [[ -z "$CANARY_TIMEOUT_OVR" ]]; then
    CANARY_TIMEOUT_S=$(( 120 + T_COMPILE_S ))
    CANARY_MAX_TIMEOUTS=1
  fi
  QUALITY_REQUEST_S=$(( 120 + T_COMPILE_S ))
  QUALITY_TIMEOUT_S="$(quality_limit_s)"
}

# Fresh prompts for every run: a pool drawn with this run's own seed, or the
# target's bench-only pool. Every request is nonce-prefixed by the harness.
pool_step() {
  local seed="${1:-$SEED_BASE}"
  if [[ -n "$T_PROMPT_POOL" ]]; then
    POOL="$T_PROMPT_POOL"
    return 0
  fi
  POOL="$RESULTS_DIR/pool/prompt_pool.jsonl"
  py "$BENCH_ROOT/harness.py" --mode pool --pool-size "$T_POOL_SIZE" \
     --isl "$T_ISL" --seed "$seed" --out "$RESULTS_DIR/pool"
}

# tps_sweep OUT PHASE SEED [EXCLUDE] - one closed-loop sweep over the levels.
tps_sweep() {
  local out="$1" phase="$2" seed="$3" prev="${4:-}" rc=0 x limit="$SWEEP_TIMEOUT_S"
  local args=(--mode tps --tps-load closed --concurrency "$T_CONCURRENCY"
              --isl "$T_ISL" --osl "$T_OSL" --ramp-s "$T_RAMP_S"
              --window-s "${SWEEP_WINDOW_S:-$T_WINDOW_S}"
              --prompt-pool "$POOL" --tokenizer "$T_TOKENIZER" --chips "$T_CHIPS"
              --slo "$T_SLO" --seed "$seed" --base-url "$BASE_URL" --model "$MODEL_NAME"
              --run-id "$RUN_ID" --phase "$phase" --out "$out")
  [[ -n "$T_LENGTH_TOL" ]] && args+=(--length-tol "$T_LENGTH_TOL")
  if [[ "$phase" == "warm" ]]; then
    limit="$WARM_TIMEOUT_S"
    # a mix of sizes: every stream sends each size once before the clock
    # starts, so each padded shape is compiled outside the scored windows
    [[ "$T_ISL" == *,* ]] && args+=(--warm-cycles 1)
    # and a request that waits for a compile is not a dead server
    (( T_COMPILE_S > 60 )) && args+=(--first-chunk-timeout "$T_COMPILE_S"
                                     --total-timeout "$T_COMPILE_S")
  fi
  if [[ -n "$prev" ]] && [[ -f "$prev" || "$DRY_RUN" == "1" ]]; then
    args+=(--pool-exclude "$prev")
  fi
  for x in ${POOL_EXCLUDE[@]+"${POOL_EXCLUDE[@]}"}; do args+=(--pool-exclude "$x"); done
  check_deadline
  MEAS_TIMEOUT="$limit" meas "$BENCH_ROOT/harness.py" "${args[@]}" || rc=$?
  if (( rc == 124 )); then
    add_reason "sweep $phase did not finish within ${limit}s"
    end_run HUNG
  fi
  (( rc == 0 )) || log "sweep $phase exited $rc; the launch reduction records what is missing"
  return 0
}

# quality2 against the fixed reference R, on a running candidate launch.
quality2_gate() {
  local dir="$1" rc=0
  if [[ "$DRY_RUN" != "1" && ! -f "$T_REF_DIR/manifest.json" ]]; then
    printf '{"passed": false, "reasons": ["no reference R at %s: the operator baseline writes it"]}\n' \
      "$T_REF_DIR" > "$dir/quality.json"
  else
    MEAS_TIMEOUT="$QUALITY_TIMEOUT_S" meas "$BENCH_ROOT/quality2.py" gate \
      --base-url "$BASE_URL" --target "$TARGET_FILE" --model "$MODEL_NAME" \
      --request-timeout "$QUALITY_REQUEST_S" \
      --reference "$T_REF_DIR" --out "$dir/quality.json" || rc=$?
    if (( rc == 124 )); then
      add_reason "quality2 gate did not finish within ${QUALITY_TIMEOUT_S}s"
      end_run HUNG
    fi
    if (( rc != 0 )) && [[ "$DRY_RUN" != "1" ]]; then
      printf '{"passed": false, "reasons": ["quality2 gate exited %s (run.log)"]}\n' \
        "$rc" > "$dir/quality.json"
    fi
  fi
  QUALITY_FILES+=("$dir/quality.json")
}

# The baseline also writes R, unless one exists (staged, installed at the end
# with the canary reference). R's floors are the reference's own launch-to-
# launch noise and a candidate is held to three times them, so R is cold
# launches of the baseline (DESIGN 2.4, reference.launches): the first is B
# itself, collected here; reference_finish starts the others. For the mock,
# one server measured three times stands in (quality2 refuses that for any
# other engine).
REF_DUMPS=()            # one `collect` per launch of R, launch 0 first
REF_INPUTS=()
reference_inputs() {
  REF_INPUTS=(--target "$TARGET_FILE")
  [[ -n "$T_REF_PROMPTS" ]] && REF_INPUTS+=(--prompts "$T_REF_PROMPTS")
  [[ -n "$T_REF_TASKS" ]] && REF_INPUTS+=(--tasks "$T_REF_TASKS")
  return 0
}

reference_step() {
  if [[ "$DRY_RUN" != "1" && -f "$T_REF_DIR/manifest.json" ]]; then
    log "reference R exists at $T_REF_DIR, kept"
    return 0
  fi
  [[ "$DRY_RUN" == "1" ]] || mkdir -p "$RESULTS_DIR/staged"
  if [[ -n "$T_REF_LAUNCHES" ]]; then
    reference_collect 0
    return 0
  fi
  local rc=0
  reference_inputs
  MEAS_TIMEOUT="$QUALITY_TIMEOUT_S" meas "$BENCH_ROOT/quality2.py" make-reference \
    --base-url "$BASE_URL" --model "$MODEL_NAME" "${REF_INPUTS[@]}" \
    --max-tokens "$T_REF_MAX_TOKENS" --request-timeout "$QUALITY_REQUEST_S" \
    --out "$RESULTS_DIR/staged/reference" || rc=$?
  if (( rc != 0 )); then
    add_reason "quality2 make-reference failed (exit $rc)"
    end_run INVALID
  fi
}

# reference_collect K - launch K of R, on the server that is running now.
reference_collect() {
  local k="$1" rc=0 out="$RESULTS_DIR/staged/reference-launches/$1" anchor=()
  (( k > 0 )) && anchor=(--anchor "${REF_DUMPS[0]}")
  reference_inputs
  MEAS_TIMEOUT="$QUALITY_TIMEOUT_S" meas "$BENCH_ROOT/quality2.py" collect \
    --base-url "$BASE_URL" --model "$MODEL_NAME" "${REF_INPUTS[@]}" \
    --max-tokens "$T_REF_MAX_TOKENS" --request-timeout "$QUALITY_REQUEST_S" \
    ${anchor[@]+"${anchor[@]}"} --out "$out" || rc=$?
  if (( rc != 0 )); then
    add_reason "reference launch $k: quality2 collect failed (exit $rc)"
    end_run INVALID
  fi
  REF_DUMPS+=("$out")
}

# After B has stopped: the other launches of R, each started cold from the
# seed (which B's compiled shapes are in by now), asked the quality prompts
# and nothing else, then R assembled from all of them.
reference_finish() {
  (( ${#REF_DUMPS[@]} )) || return 0
  local k label dir rc=0 dumps
  for (( k = 1; k < T_REF_LAUNCHES; k++ )); do
    label="R$k"
    dir="$RESULTS_DIR/launches/$label"
    run_cmd mkdir -p "$dir"
    check_deadline
    log "launch $label (reference launch $k of R)"
    start_container "$label" "$TMPROOT/exp" "$IMAGE"
    if ! wait_healthy; then
      stop_container "$dir"
      add_reason "reference launch $label never became healthy (launches/$label/server.log)"
      end_run INVALID
    fi
    reference_collect "$k"
    stop_container "$dir"
    cache_settle "$label" reference
  done
  dumps="$(IFS=,; printf '%s' "${REF_DUMPS[*]}")"
  reference_inputs
  py "$BENCH_ROOT/quality2.py" make-reference --launches-from "$dumps" \
     "${REF_INPUTS[@]}" --out "$RESULTS_DIR/staged/reference" || rc=$?
  if (( rc != 0 )); then
    add_reason "quality2 make-reference failed (exit $rc)"
    end_run INVALID
  fi
  [[ "$DRY_RUN" == "1" ]] || rm -rf "$RESULTS_DIR/staged/reference-launches"
}

# tps_launch_one LABEL ROLE EXP IMAGE - one cold launch, measured and reduced
# to launches/LABEL/launch.json. S is the screen launch: one short window,
# no canary and no quality gate, and it is not one of the verdict's launches.
tps_launch_one() {
  local label="$1" role="$2" sut_exp="$3" image="$4"
  local dir="$RESULTS_DIR/launches/$1" b prev nblocks="$T_BLOCKS" blocks=() stage="$1"
  [[ "$role" == "candidate" && -n "$KEPT_STAGE" ]] && stage="$KEPT_STAGE"
  [[ "$MODE" == "instrument" ]] && nblocks=1
  SWEEP_WINDOW_S=""
  [[ "$label" == "S" ]] && { nblocks=1; SWEEP_WINDOW_S="$T_SCREEN_WINDOW_S"; }
  run_cmd mkdir -p "$dir"
  check_deadline
  log "launch $label ($role)"
  start_container "$label" "$sut_exp" "$image" "$stage"
  if ! wait_healthy; then
    stop_container "$dir"
    if [[ "$role" == "candidate" ]]; then
      add_reason "launch $label: no healthy server within ${HEALTH_TIMEOUT_S}s (launches/$label/server.log)"
      end_run BUILD_FAILED
    fi
    add_reason "incumbent launch $label never became healthy: the box, not the candidate, needs looking at"
    end_run INVALID
  fi
  tps_sweep "$dir/warm" warm "$DRAW_SEED"
  prev="$dir/warm/tps_records.jsonl"
  LAUNCH_RECORDS=("$prev")
  # after the warm sweep: on a server that compiles on first use the canary's
  # first request would otherwise wait for the decode step to compile
  if [[ "$MODE" == "score" ]] && [[ "$label" == "C1" || "$label" == "B" ]]; then
    canary_step "$dir"
  fi
  for b in $(seq 1 "$nblocks"); do
    tps_sweep "$dir/b$b" "block$b" "$(( DRAW_SEED + b ))" "$prev"
    blocks+=(--block "$dir/b$b")
    prev="$dir/b$b/tps_records.jsonl"
    LAUNCH_RECORDS+=("$prev")
  done
  if [[ "$label" == "B" ]]; then
    reference_step
  elif [[ "$role" == "candidate" && "$MODE" == "score" && "$label" != "S" ]]; then
    quality2_gate "$dir"
  fi
  stop_container "$dir"
  py "$BENCH_ROOT/compare.py" tps-launch --label "$label" --role "$role" \
     "${blocks[@]}" --stall-ms "$T_STALL_MS" --out "$dir/launch.json"
  cache_settle "$label" "$role" "$stage"
  [[ "$label" == "S" ]] || LAUNCH_FILES+=("$dir/launch.json")
}

# The prompts of the launch just measured go to no later launch of this run.
# Called after a candidate launch that another candidate launch follows: the
# candidate keeps a writable cache between them (KEPT_STAGE) and could carry
# an answer across. The launches that follow still share one draw, so each
# incumbent launch answers the same prompts as the candidate launch next to it.
# So in an ABBA run I1 and I2 answer different prompts, and a calibration for
# it does the same between its launches (tps_calibrate_body).
retire_prompts() {
  local f
  for f in ${LAUNCH_RECORDS[@]+"${LAUNCH_RECORDS[@]}"}; do
    [[ -f "$f" || "$DRY_RUN" == "1" ]] && POOL_EXCLUDE+=("$f")
  done
  return 0
}

# board_entry best|prev - the leaderboard's best (or the one before it):
# ENTRY_ID, ENTRY_IMAGE, and ENTRY_EXP = a private copy of its snapshot.
board_entry() {
  local which="$1" line snap
  ENTRY_ID=""; ENTRY_EXP=""; ENTRY_IMAGE=""
  if [[ "$DRY_RUN" == "1" ]]; then
    ENTRY_ID="<$which>"
    ENTRY_EXP="${TMPROOT:-<tmp>}/$which-exp"
    ENTRY_IMAGE="inference-judge-inc:$T_ID-<$which>"
    printf 'WOULD RUN: rsync the %s entry of %s (its stored snapshot/exp and snapshot/src) to %s and %s\n' \
      "$which" "$LEADERBOARD_TPS" "$ENTRY_EXP" "${ENTRY_EXP%-exp}-src"
    return 0
  fi
  line="$("$PY" -I -c 'import json, sys
try:
    board = json.load(open(sys.argv[1]))
except Exception:
    board = {}
e = board.get("best") if sys.argv[2] == "best" else ((board.get("history") or [None])[-1])
e = e or {}
print("\t".join(str(e.get(k) or "") for k in ("id", "snapshot", "image")))' \
    "$LEADERBOARD_TPS" "$which" 2>/dev/null || true)"
  ENTRY_ID="$(printf '%s' "$line" | cut -f1)"
  snap="$(printf '%s' "$line" | cut -f2)"
  ENTRY_IMAGE="$(printf '%s' "$line" | cut -f3)"
  [[ -n "$ENTRY_ID" && -n "$snap" && -d "$snap" ]] || return 1
  ENTRY_EXP="$TMPROOT/$which-exp"
  mkdir -p "$ENTRY_EXP"
  safe_rsync "$snap/" "$ENTRY_EXP/"
  # the engine trees the entry was promoted with (keep_engine_trees), next to
  # its snapshot/exp; seal_args mounts them for this entry's launches
  local kept name
  kept="$(dirname "$snap")/src"
  if [[ -d "$kept" ]]; then
    for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
      mkdir -p "$TMPROOT/$which-src/$name"
      [[ -d "$kept/$name" ]] && safe_rsync "$kept/$name/" "$TMPROOT/$which-src/$name/"
    done
  else
    log "the $which entry $ENTRY_ID kept no engine trees: its launches get today's checkouts"
  fi
  # the image the entry was promoted with (deps baked in)
  [[ -n "$ENTRY_IMAGE" ]] || ENTRY_IMAGE="$BASE_IMAGE"
  return 0
}

# tps_screen_step LABEL - Step 1 on the screen launch S, or on C1 when the
# target has no screen window.
tps_screen_step() {
  local v
  if py "$BENCH_ROOT/compare.py" tps-screen \
       --launch "$RESULTS_DIR/launches/$1/launch.json" \
       --leaderboard "$LEADERBOARD_TPS" --tol "$T_SCREEN_TOL" \
       --out "$RESULTS_DIR/screen.json"; then
    return 0
  fi
    v="$(json_get "$RESULTS_DIR/screen.json" verdict)"
  end_run "${v:-REJECTED_SCREEN}"
}

# Before the first launch: whether this run can be judged at all. Stamps that
# differ from the calibration's or the incumbent's can only end INVALID
# (ruler_mismatch); with no calibration, or no incumbent yet, the run can only
# end INCONCLUSIVE. All of it is known here. Asked here it costs nothing; left
# to the verdict it costs every launch of the run. `nobest` for a
# confirmation, whose verdict holds the stamps against the calibration only.
tps_preflight_step() {
  local out="$RESULTS_DIR/preflight.json" why v board=(--leaderboard "$LEADERBOARD_TPS")
  [[ "${1:-}" == "nobest" ]] && board=()
  if py "$BENCH_ROOT/compare.py" tps-preflight --stamps "$RESULTS_DIR/stamps.json" \
       --calibration "$CALIBRATION" ${board[@]+"${board[@]}"} --out "$out"; then
    return 0
  fi
  while IFS= read -r why; do
    [[ -n "$why" ]] && add_reason "$why"
  done < <("$PY" -I -c 'import json, sys
try:
    print("\n".join(json.load(open(sys.argv[1]))["reasons"]))
except Exception:
    pass' "$out" 2>/dev/null)
  [[ -n "$REASONS" ]] || add_reason "compare.py tps-preflight failed (run.log)"
  v="$(json_get "$out" verdict)"
  end_run "${v:-INVALID}"
}

tps_verdict_step() {
  local f args=(tps-verdict --id "$ID" --design "$T_DESIGN"
                --calibration "$CALIBRATION" --stamps "$RESULTS_DIR/stamps.json"
                --leaderboard "$LEADERBOARD_TPS" --guard-tol "$T_GUARD_TOL"
                --out "$RESULTS_DIR/verdict.json")
  for f in ${LAUNCH_FILES[@]+"${LAUNCH_FILES[@]}"}; do args+=(--launch "$f"); done
  for f in ${QUALITY_FILES[@]+"${QUALITY_FILES[@]}"}; do args+=(--quality "$f"); done
  py "$BENCH_ROOT/compare.py" "${args[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then VERDICT="PROMOTED"; return 0; fi
  VERDICT="$(json_get "$RESULTS_DIR/verdict.json" verdict)"
  [[ -n "$VERDICT" ]] || VERDICT="INFRASTRUCTURE_FAILURE"
}

# A winner's image is kept as inference-judge-inc:<target>-<id> (never pruned): the
# next ABBA starts the incumbent from it and from its stored snapshot.
tps_promote_step() {
  [[ "$VERDICT" == "PROMOTED" || "$VERDICT" == "BASELINE" ]] || return 0
  local img="" ref=()
  (( IS_BASELINE )) && ref=(--reference)
  if [[ "$LAUNCHER" == "docker" ]]; then
    img="inference-judge-inc:$T_ID-$ID"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'WOULD RUN: docker tag %s %s\n' "$IMAGE" "$img"
    else
      with_timeout 60 docker tag "$IMAGE" "$img" || log "could not tag $img"
    fi
  fi
  local rc=0
  py "$BENCH_ROOT/compare.py" tps-promote --id "$ID" \
       --results "$RESULTS_DIR/results.json" --leaderboard "$LEADERBOARD_TPS" \
       --image "$img" --snapshot "$RESULTS_DIR/snapshot/exp" ${ref[@]+"${ref[@]}"} || rc=$?
  (( rc == 0 )) && return 0
  log "compare.py refused the promotion (exit $rc); recording REJECTED instead"
  if [[ "$VERDICT" == "PROMOTED" ]]; then
    add_reason "compare.py refused the promotion (exit $rc); its message is in run.log"
    VERDICT="REJECTED"
    finish
  fi
  return 0
}

tps_candidate_body() {
  copy_experiment
  snapshot_step
  build_image
  isolation_probe "$TMPROOT/exp"
  stamps_step
  tps_preflight_step
  pool_step
  if ! board_entry best; then      # the preflight saw the entry: its snapshot is gone
    add_reason "the incumbent on $LEADERBOARD_TPS has no stored snapshot: run the operator baseline again"
    end_run INCONCLUSIVE
  fi
  local inc_exp="$ENTRY_EXP" inc_image="$ENTRY_IMAGE" order=(I1 C1 C2 I2) l
  [[ "$T_DESIGN" == "ici" ]] && order=(I1 C1 I2)
  KEPT_STAGE="candidate"
  if [[ -n "$T_SCREEN_WINDOW_S" ]]; then
    tps_launch_one S candidate "$TMPROOT/exp" "$IMAGE"
    tps_screen_step S
    retire_prompts
  fi
  log "$T_DESIGN against the incumbent $ENTRY_ID: ${order[*]}"
  for l in "${order[@]}"; do
    case "$l" in
      I*) tps_launch_one "$l" incumbent "$inc_exp" "$inc_image" ;;
      C*) tps_launch_one "$l" candidate "$TMPROOT/exp" "$IMAGE" ;;
    esac
    [[ "$l" == "C1" && -z "$T_SCREEN_WINDOW_S" ]] && tps_screen_step C1
    [[ "$l" == "C1" && " ${order[*]} " == *" C2 "* ]] && retire_prompts
  done
  tps_verdict_step
  keep_engine_trees
  wrap_up tps_promote_step
}

tps_baseline_body() {
  copy_experiment
  snapshot_step
  build_image
  isolation_probe "$TMPROOT/exp"
  pool_step
  tps_launch_one B candidate "$TMPROOT/exp" "$IMAGE"
  if [[ "$DRY_RUN" != "1" ]]; then
    local status
    status="$(json_get "$RESULTS_DIR/launches/B/launch.json" status)"
    if [[ "$status" != "ok" ]]; then
      add_reason "baseline launch is not a measurement: ${status:-no launch.json}"
      end_run INVALID
    fi
  fi
  reference_finish
  commit_references
  stamps_step              # after R is in place: the stamps carry its hash
  VERDICT="BASELINE"
  keep_engine_trees
  wrap_up tps_promote_step
}

# Step 0 (operator): launches of the incumbent alone give sigma_L, sigma_B and
# the MDE every later verdict is judged against. sigma_L is also the scale of
# the drift check, I2 against I1, and in an ABBA run those two answer
# different prompts (retire_prompts). So here every launch has a draw of its
# own and leaves out what the launch before it was sent: the spread of the K
# launches then holds what the drift check sees, the box from launch to launch
# and one set of prompts against another. Replayed prompts would leave the
# second out and void honest runs. An ICI run keeps I1, C1 and I2 on one set,
# so its calibration replays one.
tps_calibrate_body() {
  copy_engines
  pool_step
  if ! board_entry best; then
    add_reason "no incumbent on $LEADERBOARD_TPS: run the operator baseline first"
    end_run INCONCLUSIVE
  fi
  IMAGE="$ENTRY_IMAGE"
  record_image_digest
  isolation_probe "$ENTRY_EXP"
  stamps_step
  local i args=(tps-calibrate --stamps "$RESULTS_DIR/stamps.json"
                --out "$RESULTS_DIR/calibration.json")
  for i in $(seq 1 "$CALIBRATE_LAUNCHES"); do
    [[ "$T_DESIGN" == "ici" ]] || DRAW_SEED=$(( SEED_BASE + 1000 * i ))
    tps_launch_one "K$i" incumbent "$ENTRY_EXP" "$ENTRY_IMAGE"
    args+=(--launch "$RESULTS_DIR/launches/K$i/launch.json")
    [[ "$T_DESIGN" == "ici" ]] || { POOL_EXCLUDE=(); retire_prompts; }
  done
  if ! py "$BENCH_ROOT/compare.py" "${args[@]}"; then
    add_reason "compare.py refused the calibration (run.log)"
    end_run INVALID
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: install %s/calibration.json as %s (rename)\n' "$RESULTS_DIR" "$CALIBRATION"
  else
    mkdir -p "$(dirname "$CALIBRATION")"
    cp "$RESULTS_DIR/calibration.json" "$CALIBRATION.tmp.$$"
    mv "$CALIBRATION.tmp.$$" "$CALIBRATION"
    log "calibration installed: $CALIBRATION"
  fi
  VERDICT="CALIBRATED"
  wrap_up
}

# Step 3 (operator): the promoted run against the best before it, I C I, on
# prompts the promotion never saw. The leaderboard keeps the confirmed effect.
tps_confirm_body() {
  copy_engines
  local best_id
  best_id="$(json_get "$LEADERBOARD_TPS" best.id)"
  if [[ "$DRY_RUN" != "1" && "$best_id" != "$ID" ]]; then
    add_reason "$ID is not the leaderboard best (${best_id:-none}); only the best is confirmed"
    end_run INVALID
  fi
  board_entry best || { add_reason "the best's snapshot is missing"; end_run INVALID; }
  local c_exp="$ENTRY_EXP" c_image="$ENTRY_IMAGE"
  board_entry prev || { add_reason "no previous best to confirm $ID against"; end_run INCONCLUSIVE; }
  local i_exp="$ENTRY_EXP" i_image="$ENTRY_IMAGE" f
  IMAGE="$c_image"
  record_image_digest
  isolation_probe "$c_exp"
  stamps_step
  tps_preflight_step nobest
  if [[ -n "$T_CONFIRM_POOL" ]]; then
    POOL="$T_CONFIRM_POOL"
  else
    POOL_EXCLUDE=()
    for f in "$BENCH_ROOT/results/$ID"/launches/*/*/tps_records.jsonl; do
      [[ -f "$f" ]] && POOL_EXCLUDE+=("$f")
    done
    pool_step "$(( SEED_BASE + 7919 ))"
  fi
  tps_launch_one I1 incumbent "$i_exp" "$i_image"
  tps_launch_one C1 candidate "$c_exp" "$c_image"
  tps_launch_one I2 incumbent "$i_exp" "$i_image"
  local args=(tps-confirm --id "$ID" --calibration "$CALIBRATION"
              --stamps "$RESULTS_DIR/stamps.json" --leaderboard "$LEADERBOARD_TPS"
              --out "$RESULTS_DIR/verdict.json")
  for f in "${LAUNCH_FILES[@]}"; do args+=(--launch "$f"); done
  py "$BENCH_ROOT/compare.py" "${args[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then VERDICT="PROMOTED"; else
    VERDICT="$(json_get "$RESULTS_DIR/verdict.json" verdict)"
    [[ -n "$VERDICT" ]] || VERDICT="INFRASTRUCTURE_FAILURE"
  fi
  wrap_up
}

# --instrument: one launch with /out mounted, the warm sweep and one block.
# Unscored: no screen, no quality, no verdict, never on the leaderboard.
tps_instrument_body() {
  copy_experiment
  build_image
  isolation_probe "$TMPROOT/exp"
  pool_step
  tps_launch_one P candidate "$TMPROOT/exp" "$IMAGE"
  VERDICT="INSTRUMENT"
  wrap_up
}

# ------------------------------------------------------------ 8. finishing --
# end_run VERDICT - the short way out of any phase: stop, tidy, write, leave.
end_run() {
  VERDICT="$1"
  stop_container
  collect_artifacts
  cleanup_workspace
  prune_images
  finish
  announce
  exit 0
}

# wrap_up [PROMOTE_FN] - the normal way out: the references a baseline staged
# are installed, results.json is written, and only then is anything promoted.
wrap_up() {
  stop_container
  collect_artifacts
  commit_references
  cleanup_workspace
  prune_images
  finish                 # results.json must exist before promote reads it
  [[ -n "${1:-}" ]] && "$1"   # may rewrite it as REJECTED if promotion is refused
  announce
  exit 0
}

# One line, once, at the very end: the driver greps for it. A dry run ran
# nothing and has no verdict; it says so in a line the driver cannot take
# for one.
announce() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'dry run: nothing was run and nothing was written, so there is no verdict\n'
    return 0
  fi
  printf 'VERDICT %s  %s/results.json\n' "$VERDICT" "$RESULTS_DIR"
}

run_kind() {
  case "$MODE" in
    instrument) echo instrument ;;
    calibrate) echo calibration ;;
    confirm) echo confirm ;;
    *) if (( IS_BASELINE )); then echo baseline; else echo candidate; fi ;;
  esac
}

finish() {
  local finished_iso duration
  finished_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  duration=$(( $(date -u +%s) - START_EPOCH ))
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: merge verdict.json + run metadata into %s/results.json (schema 2, atomic)\n' "$RESULTS_DIR"
    printf 'WOULD RUN: write %s/summary.txt\n' "$RESULTS_DIR"
    return 0
  fi
  mkdir -p "$RESULTS_DIR" 2>/dev/null || true
  # Merge the judge's verdict block with what this run knows about itself, and
  # write the two files the research side reads: results.json and summary.txt.
  # Every input read here lives in the judge's own directory; none of it comes
  # from the candidate's /out mount.
  IJ_ID="$ID" IJ_VERDICT="$VERDICT" IJ_STARTED="$STARTED_ISO" \
  IJ_RUN_ID="$RUN_ID" IJ_KIND="$(run_kind)" IJ_REASONS="$REASONS" \
  IJ_FINISHED="$finished_iso" IJ_DURATION="$duration" \
  IJ_IMAGE="$IMAGE" IJ_IMAGE_DIGEST="${IMAGE_DIGEST:-}" \
  IJ_CANARY="${CANARY_MATCHED:-0}" \
  IJ_DIR="$RESULTS_DIR" IJ_BASE_URL="${BASE_URL:-}" \
  IJ_CLEANUP_OK="$CLEANUP_OK" IJ_DEPS="$DEPS_INSTALLED" \
  IJ_METRIC_KIND="${T_METRIC:-}" IJ_TARGET_ID="${T_ID:-}" \
  IJ_TARGET_FILE="$TARGET_FILE" IJ_TARGET_HASH="${T_HASH:-}" \
  IJ_LAUNCHER="${LAUNCHER:-}" IJ_ENGINE_FLAGS="${SEAL_ARGS[*]-}" \
  IJ_RULER_LOCK_STATE="$RULER_LOCK_STATE" IJ_RULER_LOCK_PATH="$RULER_LOCK_PATH" \
  "$PY" -I - <<'PYFINISH'
import glob, json, os, pathlib

d = pathlib.Path(os.environ["IJ_DIR"])
env = os.environ.get


def load(name, default=None):
    p = d / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return default
    return default


verdict_block = load("verdict.json", {}) or {}
kind = env("IJ_KIND")
verdict = env("IJ_VERDICT")

# reasons: the judge's, then what run.sh itself saw. Every REJECTED* carries
# at least one.
reasons = list(verdict_block.get("reasons") or [])
screen_block = load("screen.json")
if verdict == "REJECTED_SCREEN" or (verdict or "").startswith("INVALID"):
    reasons += [r for r in ((screen_block or {}).get("reasons") or []) if r not in reasons]
reasons += [r for r in (env("IJ_REASONS") or "").splitlines() if r and r not in reasons]
if verdict == "REJECTED_SCREEN" and not reasons and screen_block:
    reasons.append("screen: " + json.dumps(screen_block, sort_keys=True))
if verdict == "TIMEOUT" and not reasons:
    reasons.append("the run's wall-clock budget was exhausted")
if (verdict or "").startswith("REJECTED") and not reasons:
    reasons.append("rejected; see rules")

launch_files = sorted(glob.glob(str(d / "launches" / "*" / "launch.json")))
launches = {}
for f in launch_files:
    try:
        l = json.loads(pathlib.Path(f).read_text())
    except (OSError, json.JSONDecodeError):
        continue
    launches[l.get("label") or pathlib.Path(f).parent.name] = {
        k: l.get(k) for k in ("role", "status", "headline", "value", "levels",
                              "decode_tps", "stream_tps", "by_isl",
                              "n_usable", "n_blocks", "n_stalled", "n_invalid",
                              "invalid_reasons")}

tokenizer_sha = None
for f in sorted(glob.glob(str(d / "launches" / "*" / "b*" / "summary.json"))):
    try:
        tokenizer_sha = (json.loads(pathlib.Path(f).read_text()).get("tokenizer")
                         or {}).get("sha256")
    except (OSError, json.JSONDecodeError):
        continue
    if tokenizer_sha:
        break

metric = verdict_block.get("metric")
if metric is None and kind in ("baseline", "instrument") and launches:
    only = next(iter(launches.values()))
    metric = {"kind": "tps", "headline": only.get("headline"),
              "levels": only.get("levels"), "decode_tps": only.get("decode_tps"),
              "stream_tps": only.get("stream_tps"), "by_isl": only.get("by_isl")}
if kind == "baseline":
    quality = {"passed": True,
               "note": "baseline: the reference R is made from this launch"}
elif verdict_block.get("quality") is not None:
    quality = verdict_block["quality"]
else:
    gates = [json.loads(pathlib.Path(f).read_text()) for f in
             sorted(glob.glob(str(d / "launches" / "C*" / "quality.json")))]
    quality = {"passed": bool(gates) and all(g.get("passed") is True for g in gates),
               "gates": gates} if gates else {}

isolation = load("isolation.json")
if isolation is None:
    isolation = {"network": "host", "writable_mounts": "n/a",
                 "exfil_probe": "none: " + (env("IJ_LAUNCHER") or "no") + " launcher",
                 "seal": "none"}

flags = (env("IJ_ENGINE_FLAGS") or "").split()
results = {
    "schema": 2,
    "id": env("IJ_ID"),
    "run_id": env("IJ_RUN_ID"),
    "kind": kind,
    "verdict": verdict,
    "reasons": reasons,
    "target": {"id": env("IJ_TARGET_ID"), "file": env("IJ_TARGET_FILE"),
               "hash": env("IJ_TARGET_HASH"), "metric": env("IJ_METRIC_KIND")},
    "stamps": load("stamps.json"),
    "ruler_lock": {"state": env("IJ_RULER_LOCK_STATE"),
                   "path": env("IJ_RULER_LOCK_PATH")},
    "attestation": {"weights_sha256": "not computed",
                    "tokenizer_sha256": tokenizer_sha,
                    "chat_template_sha256": "not computed",
                    "engine_flags": flags,
                    "engine_image_digest": env("IJ_IMAGE_DIGEST") or None},
    "metric": metric,
    "stats": verdict_block.get("stats"),
    "isolation": isolation,
    "started": env("IJ_STARTED"),
    "finished": env("IJ_FINISHED"),
    "duration_s": int(env("IJ_DURATION")),
    "engine_image": env("IJ_IMAGE"),
    "engine_image_digest": env("IJ_IMAGE_DIGEST") or None,
    "base_url": env("IJ_BASE_URL", ""),
    "deps_installed": env("IJ_DEPS") == "true",
    "cleanup_ok": env("IJ_CLEANUP_OK") != "false",
    "snapshot": str(d / "snapshot"),
    "canary": {"matched": int(env("IJ_CANARY") or 0), "total": 10},
    "quality": quality,
    "launches": launches,
    "design": verdict_block.get("design"),
    "screen": load("screen.json"),
}
# atomic: a reader never sees half a results.json
tmp = d / ".results.json.tmp"
tmp.write_text(json.dumps(results, indent=2))
os.replace(tmp, d / "results.json")


def ms(v):
    return "n/a" if v is None else f"{v:.0f} ms"


head = [
    f"experiment: {results['id']}  ({results['kind']}, run {results['run_id']})",
    f"verdict:    {results['verdict']}",
    f"target:     {results['target']['id']} ({results['target']['metric']})",
    f"ran:        {results['started']} to {results['finished']} "
    f"({results['duration_s']}s)",
    f"image:      {results['engine_image']} "
    f"({results['engine_image_digest'] or 'no digest'})",
    f"isolation:  network {isolation.get('network')}, exfil probe "
    f"{isolation.get('exfil_probe')}",
    f"cleanup:    {'ok' if results['cleanup_ok'] else 'INCOMPLETE'}",
    "",
]
if reasons:
    head += ["reasons:"] + [f"  - {r}" for r in reasons] + [""]

def tok(v):
    return "n/a" if v is None else f"{v:.1f}"

# the headline is per chip; one stream's own speed is the number a user
# sees: the whole answer (request sent to last token, what a mix of
# sizes is scored on), and the decoding alone
m = metric or {}
sized = bool(m.get("by_isl"))
if m.get("stream_tps") is not None:
    head[-1:-1] = [f"speed:      one stream {tok(m.get('stream_tps'))} tok/s for the "
                   f"whole answer, {tok(m.get('decode_tps'))} decoding; headline "
                   f"{tok(m.get('headline'))} tok/s per chip"]
for size, row in (m.get("by_isl") or {}).items():
    head[-1:-1] = [f"  isl {size}: {tok(row.get('stream_tps'))} tok/s "
                   f"({tok(row.get('decode_tps'))} decoding), TPOT "
                   f"{ms(row.get('p50_tpot_ms'))}, TTFT {ms(row.get('p50_ttft_ms'))}"]
lines = head + ["launches (whole-answer speed per chip over the mix of sizes, tok/s):"
                if sized else "launches (goodput per chip, tok/s):"]
for label, l in launches.items():
    hv = l.get("headline")
    lines.append(f"  {label:<3} {l.get('role'):<9} {l.get('status'):<16} "
                 f"headline {'n/a' if hv is None else f'{hv:.1f}'}  "
                 f"one stream {tok(l.get('stream_tps'))} "
                 f"({tok(l.get('decode_tps'))} decoding)  "
                 f"blocks {l.get('n_usable')}/{l.get('n_blocks')}")
st = results.get("stats") or {}
if st:
    lines += ["", f"{st.get('design')}: delta {st.get('delta')}  se {st.get('se')}  "
              f"mde {st.get('mde')}  passed {st.get('passed')}"]
lines += ["", f"quality2: passed={quality.get('passed')}"]
(d / "summary.txt").write_text("\n".join(lines) + "\n")
PYFINISH
  # bench owns the results; the researcher reads them
  chmod -R a+rX "$RESULTS_DIR" 2>/dev/null || true
}

# ---------------------------------------------------------------- the run ---
run_body() {
  log "experiment $ID ($MODE), target $T_ID ($T_METRIC, launcher $LAUNCHER), results in $RESULTS_DIR"
  case "$LAUNCHER" in
    local) BASE_URL="http://127.0.0.1:$PORT/v1" ;;
    *) BASE_URL="http://127.0.0.1:$T_PORT/v1" ;;     # as the sidecar sees it
  esac
  tps_setup
  case "$MODE" in
    calibrate) tps_calibrate_body ;;
    confirm) tps_confirm_body ;;
    instrument) tps_instrument_body ;;
    *) if (( IS_BASELINE )); then tps_baseline_body; else tps_candidate_body; fi ;;
  esac
}

# The watchdog. The body runs in its own process group; when the budget is
# gone the whole group dies, the container is removed and the run is written
# up as TIMEOUT. No phase can outlive this, whatever it is doing.
watchdog() {
  local body_pid rc=0 deadline rc_file="$RESULTS_DIR/.body-exit-code"
  rm -f "$rc_file" 2>/dev/null || true
  set -m
  # The exit code goes through a file: `wait` cannot be given a deadline in
  # portable bash, and a finished-but-unreaped child still answers kill -0.
  ( trap - INT TERM HUP PIPE; trap 'printf "%s" "$?" > "'"$rc_file"'"' EXIT; run_body ) &
  body_pid=$!
  BODY_PID="$body_pid"
  set +m
  deadline=$(( $(date -u +%s) + BUDGET_S ))
  while :; do
    if [[ -f "$rc_file" ]] || ! kill -0 "$body_pid" 2>/dev/null; then
      wait "$body_pid" 2>/dev/null || true
      rc="$(cat "$rc_file" 2>/dev/null || echo 0)"
      rm -f "$rc_file" 2>/dev/null || true
      return "${rc:-0}"
    fi
    if (( $(date -u +%s) >= deadline )); then
      log "watchdog: budget of ${BUDGET_S}s exhausted, killing the run"
      kill -TERM "-$body_pid" 2>/dev/null || kill -TERM "$body_pid" 2>/dev/null || true
      sleep 2
      kill -KILL "-$body_pid" 2>/dev/null || kill -KILL "$body_pid" 2>/dev/null || true
      wait "$body_pid" 2>/dev/null || true
      rm -f "$rc_file" 2>/dev/null || true
      # This cleanup is outside the body the watchdog monitors, so it gets its
      # own bound. cleanup_workspace runs in a subshell
      # here and cannot set CLEANUP_OK, so we re-check what it checks.
      bounded 300 stop_container
      bounded 300 stop_everything     # sidecars and launches, by label
      bounded 60 collect_artifacts
      bounded 240 cleanup_workspace
      if [[ -n "${TMPROOT:-}" && -d "${TMPROOT:-}" ]]; then CLEANUP_OK=false; fi
      VERDICT="TIMEOUT"
      add_reason "the run's wall-clock budget of ${BUDGET_S}s was exhausted"
      finish
      announce
      return 0
    fi
    sleep 1 || true      # a group-wide SIGTERM kills the sleep, not the run
  done
}

# The caller is not the run's owner. The researcher starts run.sh from a
# Claude Code session and that session ends before the run does: Claude Code
# then sends SIGTERM to every background task it started (measured: one
# SIGTERM to the task's process group, nothing after it), a
# closing terminal sends SIGHUP, a closed pipe gives SIGPIPE. Before this
# trap, that SIGTERM ran the EXIT trap in main - docker rm -f - under a body
# that was still measuring, and the run was written up as REJECTED against a
# dead port. Nothing the
# caller does ends an experiment; the watchdog is the only killer. SIGINT is
# different: ctrl-C in a foreground run is the operator, and it aborts.
BODY_PID=""
on_detach() {
  log "SIG$1 from the caller ignored: the run continues on its own, only the ${BUDGET_S}s watchdog stops it; results in $RESULTS_DIR"
}
on_interrupt() {
  log "SIGINT: aborting the run, removing the container, releasing the lock"
  if [[ -n "$BODY_PID" ]]; then
    kill -TERM "-$BODY_PID" 2>/dev/null || kill -TERM "$BODY_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "-$BODY_PID" 2>/dev/null || kill -KILL "$BODY_PID" 2>/dev/null || true
  fi
  exit 130
}

# The transcript of the run goes to results/<id>/run.log as well as to the
# caller, through tees that ignore the caller's signals and its closed pipe
# (-p where tee has it: writes to a gone stdout are then dropped, not fatal),
# so a run that outlives its caller keeps every line it prints.
start_transcript() {
  local log="$RESULTS_DIR/run.log" teep=""
  tee -p /dev/null </dev/null >/dev/null 2>&1 && teep="-p"
  # shellcheck disable=SC2086
  exec > >(trap '' INT TERM HUP PIPE; exec tee $teep -a "$log") \
       2> >(trap '' INT TERM HUP PIPE; exec tee $teep -a "$log" >&2)
}

# The EXIT trap belongs to the parent: a subshell does not inherit it, so
# whatever happens to the body, the container and the lock are released here.
final_cleanup() {
  local rc=$?
  if [[ "$DRY_RUN" != "1" && -n "$CONTAINER" && -n "$RESULTS_DIR" ]]; then
    stop_everything
  fi
  if [[ "$DRY_RUN" != "1" && -n "${TMPROOT:-}" && -d "${TMPROOT:-}" ]]; then
    rm -rf "$TMPROOT" 2>/dev/null || true
  fi
  release_lock
  return "$rc"
}


main() {
  parse_args "$@"
  resolve_target
  load_target_settings
  if [[ "$MODE" == "lock-ruler" ]]; then
    lock_ruler
  fi
  ruler_lock_check
  trap final_cleanup EXIT
  trap on_interrupt INT
  trap 'on_detach TERM' TERM
  trap 'on_detach HUP' HUP
  trap ':' PIPE
  prepare_dirs
  reserved_id_check

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'WOULD RUN: flock -n %s/.lock (or mkdir %s/.lock.d) - one run at a time, path published in %s/LOCK_PATH\n' \
      "$BENCH_ROOT" "$BENCH_ROOT" "$BENCH_ROOT"
    printf 'WOULD RUN: watchdog: kill the whole run after %ss and write TIMEOUT\n' "$BUDGET_S"
    printf 'WOULD RUN: ruler lock %s: %s\n' "$RULER_LOCK_PATH" "$RULER_LOCK_STATE"
    run_body
    exit 0
  fi

  if ! acquire_lock; then
    log "another experiment holds the bench lock"
    VERDICT="BUSY"
    finish
    announce
    exit 0
  fi

  start_transcript

  # The private copies are created here, not in the body: the watchdog has to
  # be able to remove them after it has killed the body.
  TMPROOT="$(mktemp -d "${TMPDIR:-/tmp}/inference-judge-$ID-XXXXXX")"

  watchdog
  exit $?
}

main "$@"

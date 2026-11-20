#!/bin/bash
# run-session.sh [STEP ...]: the GPU session of C4 (NOTES.md, "GPU session (C4)"), on the controller VM as root,
# once the driver has a ready GPU machine (setup-box.sh with the launcher and the firewall: the driver runs it).
# start.sh on the Mac ships the bench files and starts this detached.  Steps (default all, in order):
#   1 preflight   the machine ready; the firewall rule, the launcher, builder out of docker; judge.draw's buckets
#                 are the trace pool's sizes (else stop: the draw is the operator's to set first)
#   2 pools       trace pool built on the machine (box-pools.sh), warm-up rows and the first chunk loaded, the
#                 task pool generated here (judge only) and loaded, 3 reference draws on that first chunk with
#                 the answer lengths logged, then the other chunks
#   3 atom        the reference over REFDRAWS draws; plain ATOM as an engine (examples/atom-engine), MEAS times
#   4 tuned       AMD's tuned engine image built on the machine (box-build-tuned.sh), then judged MEAS times
#                 (examples/amd-tuned-engine, through the launcher as cand)
#   5 quality     measures A and B of every run (ij-judge quality)
#   6 submit      one real ./submit by agents (plain ATOM), timed
#   7 card        box-card-checks.sh, then the four cheaters through ./submit as agents
# A failed step is logged and the next one runs.  Stops at once: an ssh host key failure (changed or unknown
# key: never bypassed), a gated dataset (exit 4) or a different one (exit 6), a failed preflight (exit 5).  Logs: /var/lib/ij-ctl/gpu-session/<stamp>/ here,
# copied after every step to /root/gpu-session/<stamp>/ on the GPU machine.
set -u
A=/opt/ij-amd/amd
C=$A/controller
S=$A/measurements/2026-11-17/gpu-session
G=/var/lib/ij-ctl/gpu-session
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
L=$G/$STAMP
JD=/var/lib/ij-judge
IN=$JD/incoming
SIZES="${SIZES:-8000,12000}"         # judge.draw must use the same buckets (2000 and 4000 give no rows)
PER="${PER:-2000}"                   # rows per size: about 330 runs at judge.draw's 6 per bucket
MEAS="${MEAS:-5}"                    # measurement runs per engine (the bar's procedure asks at least 5)
REFDRAWS="${REFDRAWS:-5}"
TASK_SEEDS="${TASK_SEEDS:-4}"        # judge_tasks.py runs; about 10,000 distinct tasks each, repeats skipped
TASKS_PER_SEED="${TASKS_PER_SEED:-10000}"
WARM="${WARM:-12}"
install -d -m 700 "$L"
LOG=$L/session.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
J() { sudo -u judge -H "$C/judge/ij-judge" "$@"; }
ag() { local d="$1"; shift; sudo -u agents -H bash -c "cd $d && $*"; }

mapfile -t SSH < <(python3 - "$C/lib" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import ijctl
c = ijctl.load()
st = ijctl.box_state(c)
if st.get("state") == "ready":
    print("\n".join(ijctl.ssh_base(c, str(ijctl.ctl_dir(c) / "ssh" / "driver"), "root", st)))
EOF
)
keystop() {  # stop on any host key complaint in the given file
  if grep -qE "Host key verification failed|REMOTE HOST IDENTIFICATION HAS CHANGED|No .* host key is known" "$1"; then
    log "STOP: the GPU machine's ssh host key was refused (changed or unknown). Not bypassed; report it."
    exit 9
  fi
}
box() {  # box CMD [< stdin]: root on the GPU machine with the driver's key and its pinned host key
  [ "${#SSH[@]}" -gt 0 ] || { log "STOP: no ready GPU machine"; exit 8; }
  local e; e=$(mktemp)
  "${SSH[@]}" "$1" 2> >(tee "$e" >&2); local rc=$?
  sleep 0.2; keystop "$e"; rm -f "$e"; return $rc
}
pushlog() { tar -C "$L" -cf - . | box "mkdir -p /root/gpu-session/$STAMP && tar -xf - -C /root/gpu-session/$STAMP" 2>> "$LOG" || true; }
step() {  # step NAME FUNC: run, log the exit code and the time, copy the logs to the machine
  local t0 rc; t0=$(date +%s)
  log "== step $1"
  "$2" >> "$LOG" 2>&1; rc=$?
  keystop "$LOG"
  log "== step $1: rc $rc in $(( $(date +%s) - t0 )) s"
  pushlog
  return $rc
}

s_preflight() {
  python3 -c "import sys; sys.path.insert(0, '$C/lib'); import ijctl; st = ijctl.box_state(ijctl.load()); print(ijctl.describe(st)); sys.exit(st.get('state') != 'ready')" || return 1
  box "$C/box/ij-fw check" | grep -qx ok || { echo "the firewall rule for cand is not in place"; return 1; }
  box "test -x $C/box/ij-docker && grep -q 'NOPASSWD: $C/box/ij-docker' /etc/sudoers.d/ij-box" || { echo "the launcher is not installed"; return 1; }
  box "! id -nG builder | grep -qw docker" || { echo "builder is in the docker group"; return 1; }
  box "docker image inspect \$(python3 -c \"import json; print(json.load(open('$A/config.json'))['ref']['image'])\") > /dev/null" \
    || { echo "the reference image is not on the machine"; return 1; }
  local draw; draw=$(python3 -c "import sys; sys.path.insert(0, '$C/lib'); import ijctl; print(','.join(sorted(ijctl.load()['judge']['draw'], key=int)))")
  echo "judge.draw buckets: $draw; trace pool sizes: $SIZES"
  [ "$draw" = "$(echo "$SIZES" | tr ',' '\n' | sort -n | paste -sd,)" ] || {
    echo "judge.draw does not match the trace pool's sizes: set judge.draw in /etc/ij-amd/controller.json first (NOTES, C4)"
    return 1; }
}

load() {  # load FILE [--warm|--tasks]: the judge reads it from its own incoming directory
  J prepare-pool "$@"
}

s_pools() {
  [ -d "$G/bench" ] || { echo "no bench files in $G/bench (start.sh ships them)"; return 1; }
  tar -C "$G/bench" -cf - . | box "rm -rf /root/gpu-session/bench && install -d -m 700 /root/gpu-session/bench && tar -xf - -C /root/gpu-session/bench"
  box "bash $S/box-pools.sh $SIZES $PER"; local rc=$?
  box "tail -5 /root/gpu-session/pools.log"
  [ $rc = 4 ] && { log "STOP: the dataset is gated or refused (box-pools.sh exit 4); no login is used"; pushlog; exit 4; }
  [ $rc = 6 ] && { log "STOP: a dataset problem: no network, an HTTP error or not 2026-11-17's file (box-pools.sh exit 6, see pools.log)"; pushlog; exit 6; }
  [ $rc = 0 ] || return 1
  install -d -m 700 -o judge -g judge "$IN" "$IN/traces"
  box "tar -C /root/gpu-session/pools -cf - ." | sudo -u judge tar -xf - -C "$IN/traces"
  cat "$IN/traces/judge-traces-summary.json"
  sudo -u judge bash -c "head -n $WARM $IN/traces/judge-traces-000.jsonl > $IN/warm.jsonl"
  load "$IN/warm.jsonl" --warm || return 1
  load "$IN/traces/judge-traces-000.jsonl" || return 1
  # the task pool: generated here as judge, read by nobody else; seeds random and not kept
  install -d -m 700 -o judge -g judge "$IN/gen"
  install -m 644 "$G/bench/pools/judge_tasks.py" "$G/bench/pools/pool_tasks.py" "$IN/gen/"
  for i in $(seq 1 "$TASK_SEEDS"); do
    sudo -u judge -H bash -c "umask 077; python3 $IN/gen/judge_tasks.py $TASKS_PER_SEED --prefix hidden > $IN/tasks-$i.jsonl" || return 1
    load "$IN/tasks-$i.jsonl" --tasks || return 1
  done
  rm -rf "$IN/gen"
  # answer lengths on the trace rows before the rest goes in: protocol 2 times the judged answers themselves
  local before; before=$(ls "$JD/refdraws" 2>/dev/null)
  J reference-draws 3
  python3 - "$JD/refdraws" "$before" <<'EOF'
import glob, json, os, statistics, sys
rows, old = [], set(sys.argv[2].split())
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    if os.path.basename(f) not in old:             # only this step's draws
        rows += (json.load(open(f)).get("per_request") or [])
if rows:
    t = sorted(r.get("tokens", 0) for r in rows)
    print(f"reference answers on trace rows: {len(t)} requests; tokens min {t[0]}, median {statistics.median(t)}, "
          f"max {t[-1]}; finish {dict((k, sum(1 for r in rows if r.get('finish') == k)) for k in {r.get('finish') for r in rows})}")
    for r in rows:
        print(f"  {r.get('id')}: {r.get('prompt_tokens')} prompt tokens, {r.get('tokens')} answer tokens, {r.get('finish')}")
else:
    print("no per-request rows from the reference draws")
EOF
  for f in "$IN"/traces/judge-traces-*.jsonl; do
    [ "$f" = "$IN/traces/judge-traces-000.jsonl" ] && continue
    load "$f" || return 1
  done
  J status | tail -2
}

s_atom() {
  J reference-draws "$REFDRAWS"
  local i; for i in $(seq 1 "$MEAS"); do J measure atom "$A/examples/atom-engine" "plain ATOM, draw $i"; done
}

s_tuned() {
  box "LOG=/root/gpu-session/$STAMP/build-tuned.log bash $S/box-build-tuned.sh"; local rc=$?
  box "tail -20 /root/gpu-session/$STAMP/build-tuned.log"
  [ $rc = 0 ] || return 1
  local i; for i in $(seq 1 "$MEAS"); do J measure tuned "$A/examples/amd-tuned-engine" "AMD tuned dev973, draw $i"; done
}

s_quality() {
  J quality | tee "$L/quality.txt"
  J status
}

engine_repo() {  # engine_repo DIR SERVE-SOURCE-DIR NOTE: a throw-away clone of the agents' repo with that serve
  rm -rf "$1"
  sudo -u agents git clone -q /home/agents/engine "$1"
  sudo -u agents cp "$2"/* "$1/"
  ag "$1" "git add -A && git commit -qm '$3'"
}

s_submit() {
  engine_repo /home/agents/gpu-session-atom "$A/examples/atom-engine" "operator: plain ATOM, timed submit"
  local t0; t0=$(date +%s)
  ag /home/agents/gpu-session-atom "./submit --note 'operator: timed ./submit, plain ATOM'"; local rc=$?
  echo "./submit as agents: rc $rc in $(( $(date +%s) - t0 )) s"
  return $rc
}

s_card() {
  box "bash $S/box-card-checks.sh" | tee "$L/card-checks.txt"; local rc=${PIPESTATUS[0]}
  local n; for n in fabricate replay read-judge proxy-ref; do
    engine_repo "/home/agents/gpu-session-$n" "$A/examples/cheaters/$n" "operator: cheater $n"
    ag "/home/agents/gpu-session-$n" "./submit --note 'operator: cheater $n'"
    echo "cheater $n: ./submit rc $? (1 = failed as it must)"
  done
  ag "/home/agents/gpu-session-replay" "./submit --note 'operator: cheater replay, second'"
  echo "cheater replay, second: ./submit rc $? (1 = failed as it must)"
  return $rc
}

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(1 2 3 4 5 6 7)
log "run-session.sh $STAMP: steps ${STEPS[*]}; sizes $SIZES x $PER, $MEAS measurement runs, $REFDRAWS reference draws"
for s in "${STEPS[@]}"; do
  case "$s" in
    1) step preflight s_preflight || { log "STOP: preflight failed"; exit 5; } ;;
    2) step pools s_pools ;;
    3) step atom s_atom ;;
    4) step tuned s_tuned ;;
    5) step quality s_quality ;;
    6) step submit s_submit ;;
    7) step card s_card ;;
    *) log "unknown step $s" ;;
  esac
done
# the throw-away clones of steps 6 and 7 and the bench copy start.sh shipped
rm -rf /home/agents/gpu-session-* "$G/bench"
log "done; logs in $L and on the GPU machine in /root/gpu-session/$STAMP"
pushlog

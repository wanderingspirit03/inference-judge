#!/usr/bin/env bash
# Verify every precondition the autoresearch loop depends on.  Changes nothing.
#
#     sudo bash setup/check.sh [--target targets/<id>.yaml]
#
# The target defaults to the installed copy, /etc/inference-judge/target.yaml.
# Run as root for the permission checks.  Prints PASS / FAIL / WARN per item
# and exits non-zero if anything FAILed.
# shellcheck disable=SC2015  # ok/bad/warn always return 0, so A && ok || bad is safe
set -uo pipefail   # deliberately NOT -e: a failing check must not stop the run

# macOS ssh clients forward LC_* to the box, and sudo's audit plugin on some
# Ubuntu images aborts on a locale it cannot load ("sudo: unable to resolve
# ..." / audit plugin errors) -- which would make every `sudo -u bench` in
# here fail for a reason that has nothing to do with the install.  Pin it.
export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IJ_REPO="${IJ_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
usage() { sed -n '2,8p' "$0"; exit 0; }
TARGET_ARG="${IJ_TARGET:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --target) TARGET_ARG="${2:-}"; shift ;;
    --target=*) TARGET_ARG="${1#--target=}" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -z "$TARGET_ARG" ] && [ -f /etc/inference-judge/target.yaml ] && TARGET_ARG=/etc/inference-judge/target.yaml
# shellcheck source=setup/target.sh
. "$SCRIPT_DIR/target.sh"
# shellcheck source=setup/tpu.sh
. "$SCRIPT_DIR/tpu.sh"
# shellcheck source=setup/gpu.sh
. "$SCRIPT_DIR/gpu.sh"
target_load "$TARGET_ARG" || exit 2

MODEL_DIR="$T_MODEL_DIR"
MANIFEST="${T_MODEL_DIR%/}.sha256"
BENCH=/opt/bench
RESEARCH=/home/researcher/research
STATE_DIR=/home/driver/research-state
VENV=/home/researcher/.venv
IMAGE=inference-judge-base:latest
LOOP_ENV=/etc/inference-judge/loop.env
# loop_env KEY DEFAULT: one setting of the installed loop.env.  Read, not
# sourced: the file sets HOME and PATH for the driver.
loop_env() {
  local v
  v=$(sed -n "s/^$1=//p" "$LOOP_ENV" 2>/dev/null | tail -1)
  v=${v%\"}; v=${v#\"}
  printf '%s' "${v:-$2}"
}
# The agent that does the research, and its adapter (research/adapters/README.md).
AGENT=$(loop_env IJ_AGENT claude)
RESEARCHER_USER=$(loop_env IJ_RESEARCHER_USER researcher)
ADAPTER=/opt/inference-judge/adapters/$AGENT/session.sh
PROJECT="$RESEARCH/adapters/$AGENT/project"     # the agent's own files, copied to the top of the repo
# agent_session <verb>: the adapter, exactly the way the driver runs it: as the
# driver's user with loop.env, through the researcher's sudo prefix (sudo
# resets the environment, so IJ_AGENT_BIN is handed over after `env`; so is
# IJ_PRIMARY_MODEL, which `auth` needs for an agent that logs in per model).
agent_session() {
  # shellcheck disable=SC2016  # expanded by the inner shell
  sudo -u driver -H bash -c 'set -a; . "$1"; set +a; shift
    exec $IJ_RESEARCHER_SUDO env ${IJ_AGENT_BIN:+"IJ_AGENT_BIN=$IJ_AGENT_BIN"} \
      ${IJ_PRIMARY_MODEL:+"IJ_PRIMARY_MODEL=$IJ_PRIMARY_MODEL"} "$@"' _ "$LOOP_ENV" "$ADAPTER" "$@"
}

PASS=0; FAIL=0; WARNS=0
ok()   { printf '\033[32mPASS\033[0m  %-52s %s\n' "$1" "${2:-}"; PASS=$((PASS+1)); }
bad()  { printf '\033[31mFAIL\033[0m  %-52s %s\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }
warn() { printf '\033[33mWARN\033[0m  %-52s %s\n' "$1" "${2:-}"; WARNS=$((WARNS+1)); }

# check <label> <command...>   -- PASS if the command exits 0
check() { local label="$1"; shift; if out=$("$@" 2>&1); then ok "$label" "$(head -1 <<<"$out")"; else bad "$label" "$(head -1 <<<"$out")"; fi; }
# softcheck: WARN instead of FAIL
softcheck() { local label="$1"; shift; if out=$("$@" 2>&1); then ok "$label" "$(head -1 <<<"$out")"; else warn "$label" "$(head -1 <<<"$out")"; fi; }

# ruler_libs PYTHON...: bootstrap's own test that PYTHON imports what the ruler imports.
ruler_libs() { IJ_BOOTSTRAP_LIB=1 bash -c '. "$1"; shift; ruler_libs "$@"' _ "$SCRIPT_DIR/bootstrap.sh" "$@"; }

mode_of()  { stat -c '%a' "$1" 2>/dev/null; }
owner_of() { stat -c '%U:%G' "$1" 2>/dev/null; }

echo "=== host ==="
[ "$(id -u)" -eq 0 ] && ok "running as root" || warn "not running as root" "permission checks may be incomplete"
# shellcheck disable=SC1091
if . /etc/os-release 2>/dev/null && [ "${VERSION_ID:-}" = 24.04 ]; then ok "ubuntu 24.04" "$PRETTY_NAME"; else warn "ubuntu 24.04" "${PRETTY_NAME:-unknown}"; fi
free_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
need_gb=40; [ "$T_KIND" = none ] && need_gb=5
[ "${free_gb:-0}" -ge "$need_gb" ] && ok "free disk on /" "${free_gb}G" || bad "free disk on /" "${free_gb}G left (need $need_gb)"
ok "target" "$(target_summary)"

echo
echo "=== accelerator and docker ==="
case "$T_KIND" in
gpu)
  if command -v nvidia-smi >/dev/null; then
    ok "nvidia-smi present" "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    # The same four questions bootstrap's gpu step asks (setup/gpu.sh): the GPUs
    # the target describes, and the driver an engine built here needs.  A wrong
    # answer is a wrong machine; no amount of retrying bootstrap will fix it.
    check "gpus visible (accelerator.chips $T_CHIPS)" gpu_check_count
    check "gpu type"   gpu_check_type
    check "gpu memory" gpu_check_memory
    check "gpu driver" gpu_check_driver
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$busy" -eq 0 ] && ok "gpu idle" "no compute processes" || warn "gpu idle" "$busy process(es) still on the GPU"
  else bad "nvidia-smi present"; fi
  # A prebuilt engine image is probed in itself: the gpu gate, below.
  [ "$T_ENGINE_CLASS" = pulled ] || softcheck "docker GPU passthrough" docker run --rm --gpus all "$GPU_PROBE_IMAGE" nvidia-smi -L
  ;;
tpu)
  # The chips and the cache disk here; the image and the gate are checked below, once
  # "engine image" has said the image is there.
  n=$(tpu_chip_nodes)
  [ "$n" -eq "$T_CHIPS" ] && ok "tpu chips visible" "$n of $T_CHIPS" || bad "tpu chips visible" "$n of $T_CHIPS device nodes"
  if [ -n "$T_DISK_LABEL" ]; then
    mountpoint -q "$T_DISK_MOUNT" && ok "cache disk mounted" "$T_DISK_LABEL at $T_DISK_MOUNT" \
      || bad "cache disk mounted" "$T_DISK_LABEL not at $T_DISK_MOUNT: bash setup/bootstrap.sh"
    grep -qs "^LABEL=${T_DISK_LABEL}[[:space:]]" /etc/fstab \
      && ok "cache disk in fstab (survives a reboot)" || warn "cache disk in fstab (survives a reboot)" "re-run bootstrap"
  fi ;;
*)
  ok "accelerator" "kind $T_KIND: no driver, CUDA or passthrough to check" ;;
esac
check "docker daemon" docker info --format '{{.ServerVersion}}'
case "$T_ENGINE_CLASS" in
built)
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "$IMAGE built" "$(docker image inspect -f '{{index .Config.Labels "inference-judge.tags"}}' "$IMAGE")"
    case " $(gpu_built_engines) " in
      *" sglang "*)
        if docker run --rm "$IMAGE" test -f /opt/venv/sglang/.install-failed 2>/dev/null; then
          warn "image sglang venv" "install failed at build; sglang track unavailable"
        else ok "image sglang venv" "present"; fi ;;
      *) ok "image sglang venv" "not built: the target names no sglang" ;;
    esac
    # The exact wheel URL an experiment needs to re-install a patched /src/vllm.
    if u=$(docker run --rm "$IMAGE" cat /opt/vllm-wheel-url 2>/dev/null) && [ -n "$u" ]; then
      ok "image /opt/vllm-wheel-url" "$(basename "$u")"
    else bad "image /opt/vllm-wheel-url" "missing: patched editable installs would fall back to a nightly"; fi
  else bad "$IMAGE built"; fi
  ;;
mock|pulled)
  if docker image inspect "$T_IMAGE" >/dev/null 2>&1; then ok "engine image" "$T_IMAGE"
  else bad "engine image" "$T_IMAGE missing: re-run bootstrap"; fi
  if [ "$T_KIND" = tpu ]; then
    tpu_image_present && ok "engine image at its pinned digest" "${T_IMAGE##*@}" \
      || bad "engine image at its pinned digest" "${T_IMAGE##*@}: re-run bootstrap"
    # The chip gate starts a container on the chips, so it is skipped while a run holds them.
    if [ -n "$(docker ps -q 2>/dev/null)" ]; then
      warn "chip gate" "skipped: a container is running (the chips are in use)"
    else
      tpu_gate_args
      if tpu_gate_run ${TPU_GATE_ARGS[@]+"${TPU_GATE_ARGS[@]}"}; then ok "chip gate (jax sees $T_CHIPS chips)" "target's own flags"
      else bad "chip gate (jax sees $T_CHIPS chips)" "$(tail -1 <<<"$TPU_GATE_ERR") -- bash setup/bootstrap.sh says which: flags or node"; fi
    fi
  fi
  if [ "$T_KIND" = gpu ] && [ "$T_ENGINE_CLASS" = pulled ]; then
    # The gate starts a container on the GPUs, so it is skipped while a run holds them.
    if [ -n "$(docker ps -q 2>/dev/null)" ]; then
      warn "gpu gate" "skipped: a container is running (the GPUs are in use)"
    elif gpu_gate_run && [ "$GPU_GATE_SEEN" -eq "$T_CHIPS" ]; then
      ok "gpu gate ($T_CHIPS GPU(s) in the engine image)" "target's own flags"
    else
      bad "gpu gate ($T_CHIPS GPU(s) in the engine image)" "saw $GPU_GATE_SEEN; $(tail -1 <<<"$GPU_GATE_OUT") -- bash setup/bootstrap.sh says why"
    fi
  fi ;;
*) ok "engine image" "engine $T_ENGINE names none" ;;
esac

echo
echo "=== tooling ==="
check "uv"          uv --version
check "agent $AGENT (its adapter's version, as $RESEARCHER_USER)" agent_session version
check "jq"          jq --version
check "rsync"       bash -c 'rsync --version | head -1'
check "tmux"        tmux -V
check "shellcheck (pre-flight lints launch.sh)" shellcheck --version
check "python3 + PyYAML (bench/target.py)" python3 -c 'import yaml; print("yaml", yaml.__version__)'
[ "$T_KIND" = gpu ] && softcheck "nsys" nsys --version

echo
echo "=== users and trust boundary ==="
check "user bench exists"  id bench
check "user researcher exists" id researcher
check "user driver exists (driver)" id driver
if id -nG researcher 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
  bad "researcher NOT in docker group" "the researcher can bypass the ruler"
else ok "researcher NOT in docker group"; fi
if id -nG bench 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
  ok "bench in docker group"
else bad "bench in docker group" "run.sh cannot start containers"; fi
# Three users (DESIGN 1): only the driver may run the ruler, and only as bench.
SUD=/etc/sudoers.d/inference-judge-driver
if [ -f "$SUD" ]; then
  if visudo -cf "$SUD" >/dev/null 2>&1; then ok "driver sudoers valid" "$SUD"; else bad "driver sudoers valid" "$SUD"; fi
  grep -qx 'driver ALL=(bench) NOPASSWD: /opt/bench/run.sh' "$SUD" \
    && ok "driver may run /opt/bench/run.sh as bench" || bad "driver may run /opt/bench/run.sh as bench" "line missing in $SUD"
  grep -qx 'driver ALL=(researcher) NOPASSWD: ALL' "$SUD" \
    && ok "driver may act as researcher (sessions, pre-flight)" || bad "driver may act as researcher" "line missing in $SUD"
else bad "driver sudoers present" "$SUD missing: run setup/harden_research.sh"; fi
if sudo -l -U researcher 2>/dev/null | grep -q 'run.sh'; then
  bad "researcher has no route to run.sh" "an active sudoers rule still lets the researcher call the ruler"
else ok "researcher has no route to run.sh"; fi
# run.sh runs AS bench and reads the experiment out of the researcher's home.  Ubuntu
# 24.04 creates home directories 0750, which makes that impossible, and the
# failure looks like a missing launch.sh rather than a permission problem.
m=$(mode_of /home/researcher)
case "$m" in
  751|755) ok "/home/researcher traversable by bench" "mode $m" ;;
  *)       bad "/home/researcher traversable by bench" "mode $m -- chmod 751 /home/researcher" ;;
esac
if sudo -u bench test -r /home/researcher/research/experiments; then
  ok "bench can read experiments/"
else bad "bench can read experiments/" "run.sh will not find any launch.sh"; fi
if sudo -u bench test -r /home/researcher/research/experiments/TEMPLATE/launch.sh; then
  ok "bench can read TEMPLATE/launch.sh"
else bad "bench can read TEMPLATE/launch.sh"; fi

echo
echo "=== driver (frozen code, units, clock) ==="
# The loop, the guard and the adapters, installed and in the repo; and every
# top-level directory the agent's project/ put in the repo (.claude, .codex).
frozen=(/opt/inference-judge/loop /opt/inference-judge/guard /opt/inference-judge/adapters
        "$RESEARCH/loop" "$RESEARCH/guard" "$RESEARCH/adapters")
if [ -d "$PROJECT" ]; then
  while IFS= read -r d; do frozen+=("$RESEARCH/${d#"$PROJECT"/}"); done \
    < <(find "$PROJECT" -mindepth 1 -maxdepth 1 -type d | sort)
fi
for d in "${frozen[@]}"; do
  if [ "$(stat -c %U "$d" 2>/dev/null)" = root ] && [ -z "$(find "$d" -perm -o+w 2>/dev/null | head -1)" ]; then
    ok "$d root-owned, read-only"
  else bad "$d root-owned, read-only" "$(owner_of "$d") -- run setup/harden_research.sh"; fi
done
[ -x "$ADAPTER" ] && ok "adapter for agent $AGENT installed" "$ADAPTER" \
  || bad "adapter for agent $AGENT installed" "$ADAPTER missing (IJ_AGENT in $LOOP_ENV) -- run setup/harden_research.sh"
[ "$(owner_of /etc/inference-judge/loop.env) $(mode_of /etc/inference-judge/loop.env)" = "root:driver 640" ] \
  && ok "/etc/inference-judge/loop.env root:driver 0640" || bad "/etc/inference-judge/loop.env root:driver 0640" "$(owner_of /etc/inference-judge/loop.env) $(mode_of /etc/inference-judge/loop.env)"
# The guard reads its frozen paths here, whichever agent calls it.
[ "$(owner_of /etc/inference-judge/guard.env) $(mode_of /etc/inference-judge/guard.env)" = "root:root 644" ] \
  && ok "/etc/inference-judge/guard.env root:root 0644" || bad "/etc/inference-judge/guard.env root:root 0644" "$(owner_of /etc/inference-judge/guard.env) $(mode_of /etc/inference-judge/guard.env) -- run setup/harden_research.sh"
for u in inference-judge-driver.service inference-judge-runner.service inference-judge-heartbeat.service inference-judge-heartbeat.timer; do
  [ -f "/etc/systemd/system/$u" ] && ok "unit $u installed" || bad "unit $u installed"
done
[ "$(stat -c %U "$STATE_DIR" 2>/dev/null)" = driver ] && ok "state dir owned by driver" "$STATE_DIR" || bad "state dir owned by driver" "$STATE_DIR"
tb="$BENCH/targets/$T_ID.yaml"
[ "$(owner_of "$tb") $(mode_of "$tb")" = "bench:bench 640" ] \
  && ok "ruler's target copy" "$tb" || bad "ruler's target copy" "$tb: $(owner_of "$tb") $(mode_of "$tb")"
[ "$(head -1 "$BENCH/TARGET" 2>/dev/null)" = "$T_ID" ] \
  && ok "ruler measures this target" "$BENCH/TARGET = $T_ID" || bad "ruler measures this target" "$BENCH/TARGET: '$(head -1 "$BENCH/TARGET" 2>/dev/null)', want $T_ID"
if cmp -s "$tb" /etc/inference-judge/target.yaml; then ok "public target copy matches the ruler's"
else bad "public target copy matches the ruler's" "/etc/inference-judge/target.yaml differs from $tb"; fi
sudo -u researcher test -r /etc/inference-judge/target.yaml \
  && ok "researcher can read /etc/inference-judge/target.yaml" || bad "researcher can read /etc/inference-judge/target.yaml"
if grep -qs -- '--lock-ruler' "$BENCH/run.sh"; then
  if [ -s "$BENCH/RULER_LOCK" ]; then ok "ruler locked (DESIGN 2.6)" "$BENCH/RULER_LOCK"
  else warn "ruler locked (DESIGN 2.6)" "no RULER_LOCK yet: bootstrap writes it when it starts the clock"; fi
fi
if grep -qs 'IJ_RULER_IMAGE' "$BENCH/run.sh"; then
  check "ruler image inference-judge-ruler:latest" docker image inspect --format '{{.Id}}' inference-judge-ruler:latest
  check "ruler image has the ruler's libraries" ruler_libs docker run --rm --network none inference-judge-ruler:latest python3
fi
cs=$(sudo -u driver -H env IJ_STATE_DIR="$STATE_DIR" IJ_RESEARCH_DIR="$RESEARCH" IJ_LEGACY_STATE=/nonexistent \
       python3 -I /opt/inference-judge/loop/loopstate.py get clock_start 2>/dev/null)
if [ -n "$cs" ] && [ "$cs" != null ] && [ "$cs" != None ]; then ok "clock started (events.jsonl)" "epoch $cs"
else warn "clock started" "not yet: bootstrap or rehearse.sh starts it (loopstate start-clock)"; fi

if [ -n "$T_MODEL_ID" ]; then
  echo
  echo "=== model ==="
  if [ -d "$MODEL_DIR" ]; then
    ok "model dir present" "$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"
    [ "$(owner_of "$MODEL_DIR")" = "root:root" ] && ok "model owned by root" || bad "model owned by root" "$(owner_of "$MODEL_DIR")"
    w=$(find "$MODEL_DIR" -type f -perm -u+w 2>/dev/null | wc -l)
    [ "$w" -eq 0 ] && ok "model read-only" || bad "model read-only" "$w writable file(s)"
    [ -f "$MODEL_DIR/config.json" ] && ok "config.json present" || bad "config.json present"
    # The ruler mounts the parent of the model dir into a container, where a link to
    # anywhere else on this host dangles.
    l=$(find "$MODEL_DIR" -type l 2>/dev/null | head -1)
    [ -z "$l" ] && ok "model dir holds real files only" || bad "model dir holds real files only" "symlink: ${l#"$MODEL_DIR"/}"
    if [ -s "$MANIFEST" ]; then
      [ "$(head -1 "$MANIFEST.revision" 2>/dev/null)" = "$T_MODEL_REV" ] \
        && ok "manifest is for the pinned revision" "$T_MODEL_REV" || bad "manifest is for the pinned revision" "want $T_MODEL_REV"
      # The same verifier bootstrap uses (presence; IJ_VERIFY_MODEL=1 re-hashes).
      vmode=quick; [ "${IJ_VERIFY_MODEL:-0}" = 1 ] && vmode=full
      if out=$(IJ_BOOTSTRAP_LIB=1 bash -c '. "$1" && verify_model_manifest "$2" "$3" "$4"' _ \
                 "$SCRIPT_DIR/bootstrap.sh" "$MODEL_DIR" "$MANIFEST" "$vmode" 2>&1); then
        ok "Hub manifest ($vmode)" "$(tail -1 <<<"$out")"
      else bad "Hub manifest ($vmode)" "$(grep -m1 FAIL <<<"$out")"; fi
    else bad "Hub sha256 manifest present" "$MANIFEST"; fi
  else bad "model dir present" "$MODEL_DIR missing"; fi
fi

echo
echo "=== bench (the ruler) ==="
if [ -d "$BENCH" ]; then
  ok "/opt/bench present" "$(owner_of "$BENCH") mode $(mode_of "$BENCH")"
  [ "$(owner_of "$BENCH")" = "bench:bench" ] && ok "bench owned by bench" || bad "bench owned by bench" "$(owner_of "$BENCH")"
  case "$(mode_of "$BENCH")" in
    751|750) ok "/opt/bench not listable by the researcher" "mode $(mode_of "$BENCH")" ;;
    *) bad "/opt/bench not listable by the researcher" "mode $(mode_of "$BENCH")" ;;
  esac
  for f in run.sh target.py harness.py metrics.py compare.py; do
    [ -e "$BENCH/$f" ] && ok "bench/$f" || bad "bench/$f" "missing"
  done
  [ -x "$BENCH/run.sh" ] && ok "run.sh executable" || bad "run.sh executable"
  check "bench venv has the ruler's libraries" ruler_libs sudo -u bench -H "$BENCH/.venv/bin/python"
  [ "$(mode_of "$BENCH/results")" = 755 ] && ok "results dir world-readable" || bad "results dir world-readable" "$(mode_of "$BENCH/results")"
  f=$(find "$BENCH/results" -mindepth 2 -maxdepth 2 -name results.json 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    if sudo -u researcher test -r "$f"; then ok "researcher can read results.json" "$(basename "$(dirname "$f")")"
    else bad "researcher can read results.json" "$f"; fi
  else warn "results" "no run yet"; fi
  sudo -u researcher ls "$BENCH" >/dev/null 2>&1 \
    && bad "researcher cannot list /opt/bench" "the researcher can list the ruler" \
    || ok "researcher cannot list /opt/bench"
  sudo -u researcher cat "$BENCH/harness.py" >/dev/null 2>&1 \
    && bad "researcher cannot read the harness" || ok "researcher cannot read the harness"
  # The gate that keeps run.sh from accepting a hand-made "baseline-*" id.
  # It exists only for the seconds bootstrap is running a baseline.
  if [ -e "$BENCH/.allow-baseline" ]; then
    bad "no stale .allow-baseline" "present: anyone can write a fake baseline id"
  else ok "no stale .allow-baseline"; fi
  if [ -n "$T_COMPILE_SEED" ]; then
    if [ -n "$(find "$T_COMPILE_SEED" -type f 2>/dev/null | head -1)" ]; then
      ok "compile-cache seed filled" "$T_COMPILE_SEED ($(du -sh "$T_COMPILE_SEED" 2>/dev/null | cut -f1))"
    else warn "compile-cache seed filled" "$T_COMPILE_SEED is empty: the first launch compiles inside its budget"; fi
    [ "$(owner_of "$T_COMPILE_SEED")" = "bench:bench" ] && ok "seed owned by bench" || bad "seed owned by bench" "$(owner_of "$T_COMPILE_SEED")"
    sudo -u researcher test -w "$T_COMPILE_SEED" 2>/dev/null && bad "seed not writable by the researcher" "the researcher could plant a cache" || ok "seed not writable by the researcher"
  fi
  for k in metric.prompt_pool metric.confirm_pool; do
    p=$(tget_opt "$k" "")
    [ -n "$p" ] || continue
    [ -s "$p" ] && ok "$k" "$p" || warn "$k" "$p missing or empty"
  done
  ref=$(tget_opt reference.dir "")
  if [ -n "$ref" ] && [ -d "$BENCH/$ref" ]; then ok "stored reference" "$BENCH/$ref"
  else warn "stored reference" "${ref:-none named}: not installed yet (docs/RUNBOOK.md, Reference and calibration)"; fi
else bad "/opt/bench present"; fi

echo
echo "=== research side ==="
if [ -d "$RESEARCH" ]; then
  ok "$RESEARCH present" "$(owner_of "$RESEARCH")"
  [ "$(owner_of "$RESEARCH")" = "researcher:researcher" ] && ok "research owned by the researcher" || bad "research owned by the researcher" "$(owner_of "$RESEARCH")"
  files=(AGENTS.md LAB_NOTEBOOK.md LEADERBOARD.md PLAN.md README.md .gitignore
         loop/run.sh loop/runner.sh loop/preflight.sh loop/bg.sh loop/loopstate.py
         guard/test-hooks.sh "adapters/$AGENT/session.sh" experiments/TEMPLATE/launch.sh)
  files+=("experiments/$T_BASELINE/launch.sh")      # every target has a baseline (setup/target.sh)
  # the agent's own files: each one under adapters/$AGENT/project, at the same path from the top
  if [ -d "$PROJECT" ]; then
    while IFS= read -r f; do files+=("${f#"$PROJECT"/}"); done < <(find "$PROJECT" -type f | sort)
  fi
  for f in "${files[@]}"; do
    [ -e "$RESEARCH/$f" ] && ok "research/$f" || bad "research/$f" "missing"
    case "$f" in
      *.json) jq -e . "$RESEARCH/$f" >/dev/null 2>&1 && ok "$f is valid JSON" || bad "$f is valid JSON" ;;
    esac
  done
  grep -qx 'queue/' "$RESEARCH/.gitignore" 2>/dev/null && ok "research/.gitignore ignores queue/" || bad "research/.gitignore ignores queue/"
  for h in "$RESEARCH"/guard/*.sh; do
    [ -x "$h" ] && ok "hook executable" "$(basename "$h")" || bad "hook executable" "$(basename "$h")"
  done
  [ -d "$RESEARCH/.git" ] && ok "research is a git repo" "$(sudo -u "$RESEARCHER_USER" git -C "$RESEARCH" rev-parse --short HEAD 2>/dev/null)" || bad "research is a git repo"
  # the guard's hooks themselves (their own suite covers the rest)
  if printf '{"tool_name":"Write","tool_input":{"file_path":"/opt/bench/run.sh"}}' \
      | sudo -u "$RESEARCHER_USER" "$RESEARCH/guard/pretooluse-edit.sh" >/dev/null 2>&1; then
    bad "write hook blocks /opt/bench"
  else ok "write hook blocks /opt/bench"; fi
  if printf '{"tool_name":"Bash","tool_input":{"command":"docker run --gpus all x"}}' \
      | sudo -u "$RESEARCHER_USER" "$RESEARCH/guard/pretooluse-bash.sh" >/dev/null 2>&1; then
    bad "bash hook blocks docker"
  else ok "bash hook blocks docker"; fi
  # Reading your own results is the whole point of results/ being world-readable.
  if printf '{"tool_name":"Bash","tool_input":{"command":"jq .verdict /opt/bench/results/some-run/results.json"}}' \
      | sudo -u "$RESEARCHER_USER" "$RESEARCH/guard/pretooluse-bash.sh" >/dev/null 2>&1; then
    ok "bash hook allows reading results"
  else bad "bash hook allows reading results" "the researcher cannot read the ruler's answers"; fi
  if printf '{"tool_name":"Bash","tool_input":{"command":"cat /opt/bench/harness.py"}}' \
      | sudo -u "$RESEARCHER_USER" "$RESEARCH/guard/pretooluse-bash.sh" >/dev/null 2>&1; then
    bad "bash hook blocks other /opt/bench paths"
  else ok "bash hook blocks other /opt/bench paths"; fi
  if printf '{"tool_name":"Bash","tool_input":{"command":"echo x > /opt/bench/results/x"}}' \
      | sudo -u "$RESEARCHER_USER" "$RESEARCH/guard/pretooluse-bash.sh" >/dev/null 2>&1; then
    bad "bash hook blocks writes to results"
  else ok "bash hook blocks writes to results"; fi
  if out=$(sudo -u "$RESEARCHER_USER" bash "$RESEARCH/guard/test-hooks.sh" 2>&1); then
    ok "hook test suite" "$(tail -1 <<<"$out")"
  else bad "hook test suite" "$(tail -1 <<<"$out")"; fi
else bad "$RESEARCH present"; fi

if [ "$T_ENGINE_CLASS" = built ] || [ "$T_ENGINE_CLASS" = pulled ] || [ "$T_KIND" = tpu ]; then
  echo
  echo "=== engine checkouts and host venv ==="
  # The trees the target lists (engine.sources); a built engine's are at a release tag,
  # a prebuilt image's at the ref the target pins (provision.sources), and a tpu
  # engine's were copied out of the image and carry one baseline commit.
  for n in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    d=/home/researcher/$n-src
    if [ -d "$d/.git" ]; then ok "$n-src checkout" "$(git -C "$d" describe --tags --always 2>/dev/null)"; else bad "$n-src checkout" "missing or not a git repo"; fi
  done
  want=$(for n in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do printf '%s\n' "$n"; done | sort | tr '\n' ' ')
  if [ ! -f /etc/inference-judge/engine-sources ] && [ -z "$want" ]; then
    ok "/etc/inference-judge/engine-sources" "absent: the hooks open no engine tree, and this target lists none"
  elif [ "$(grep . /etc/inference-judge/engine-sources 2>/dev/null | sort | tr '\n' ' ')" = "$want" ]; then
    ok "/etc/inference-judge/engine-sources matches the target" "${want:-no engine tree}"
  else bad "/etc/inference-judge/engine-sources matches the target" "the edit hook would guard the wrong trees: re-run bootstrap"; fi
fi

if [ "$T_ENGINE_CLASS" = built ]; then
  # harden_research.sh retires the host GPU venv from the researcher unless
  # --keep-gpu-venv; either state is legitimate.
  if [ -x "$VENV/bin/python" ] && sudo -u researcher test -x "$VENV/bin/python"; then
    softcheck "host venv vllm importable" sudo -u researcher "$VENV/bin/python" -c 'import vllm;print(vllm.__version__)'
  else ok "host GPU venv" "retired from the researcher (harden) or absent"; fi
fi

echo
echo "=== agent log-in ($AGENT) ==="
# The adapter knows where its agent keeps a login: 0 logged in, 1 not, 2 cannot tell.
agent_session auth >/dev/null 2>&1; arc=$?
case "$arc" in
  0) ok "agent $AGENT is logged in" "as $RESEARCHER_USER" ;;
  2) ok "agent $AGENT has a login" "log-in state not inspected" ;;
  1) how=$(bash "$(dirname "$ADAPTER")/setup.sh" login 2>/dev/null)
     bad "agent $AGENT is logged in" "log in: ssh -t root@<ip> 'su - $RESEARCHER_USER', then: ${how:-the $AGENT command, as its documentation says}" ;;
  *) bad "agent $AGENT is logged in" "could not ask: $ADAPTER auth exited $arc" ;;
esac

echo
printf '\n%d passed, %d warnings, %d failed\n' "$PASS" "$WARNS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1

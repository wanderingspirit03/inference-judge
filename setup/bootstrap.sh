#!/usr/bin/env bash
# inference-judge bootstrap: turn a fresh Ubuntu 24.04 box into the
# autoresearch box for one TARGET (targets/<id>.yaml: the accelerator, the
# engine, the model and the metric; DESIGN 2.1).
#
# Run ONCE as root, from a checkout of this repo:
#     mkdir -p /var/log/inference-judge
#     bash setup/bootstrap.sh --target targets/<id>.yaml 2>&1 | tee /var/log/inference-judge/bootstrap.log
#     bash setup/bootstrap.sh --target targets/<id>.yaml --plan     # what it would do, as any user
#
# Everything that depends on the hardware comes from the target, read through
# bench/target.py's CLI (setup/target.sh):
#   accelerator.kind none  no NVIDIA driver, nvidia-smi, CUDA probe, nsys or GPU
#                          passthrough; the box can be a laptop container
#                          (setup/local_box/).
#   accelerator.kind gpu   an NVIDIA box (setup/gpu.sh): the GPUs counted against the
#                          target, docker's GPU passthrough, nsys; for a prebuilt
#                          engine image the gpu gate (nvidia-smi sees every GPU, in
#                          that image, with the target's own docker flags).  UNPROVEN
#                          ON REAL HARDWARE: no GPU has been through these steps.
#   accelerator.kind tpu   a TPU VM (setup/tpu.sh): chips counted, the cache disk
#                          mounted, the engine image loaded at its digest, the chip
#                          gate (jax sees every chip, in that image, with the target's
#                          own docker flags), a flat model dir, the compile-cache
#                          seed, and the engine source trees taken from the image.
#   engine                 mock: its image is built from targets/<id>.Dockerfile;
#                          vllm/sglang on inference-judge-base: built here from pinned tags
#                          (IJ_VLLM_TAG, and IJ_SGLANG_TAG when the target
#                          names sglang); anything else: the image the target names
#                          is pulled, and its engine.sources are fetched at the refs
#                          the target pins (provision.sources).
#   model.hf_id            downloaded only when the target names one, at a pinned
#                          40-hex revision, and verified file by file against the
#                          Hub's own hashes (sha256 for LFS files).
#
# The box gets three users (DESIGN 1): bench (the ruler, /opt/bench), researcher
# (the agent's sessions, ~/research) and driver (the driver and the runner; created by
# setup/harden_research.sh, which bootstrap runs).  The researcher's coding
# agent is the one --agent names (research/adapters/<name>/; default claude).  The run clock is started
# through the driver's own ledger (loopstate.py start-clock, as driver).
#
# DEFAULT BEHAVIOUR IS INSTALL-OR-RESUME.  Re-running after
# a failure fixes what is missing and touches nothing that already works: it
# does not force-checkout engine source you may have patched, does not make
# existing results private again, and does not restart the clock.  The two
# destructive operations are behind explicit flags:
#
#     --reset-clock         start the clock again even if it is already running
#     --reinstall-engines   force the engine checkouts back to the release tags
#
# setup/check.sh verifies the same postconditions without changing anything.
# setup/rehearse.sh exercises the funnel end to end before the clock starts.
#
# See setup/README.md for the operator procedure and docs/RUNBOOK.md for the
# end-to-end story.
set -euo pipefail

# macOS ssh clients forward LC_* to the box, and sudo's audit plugin on some
# Ubuntu images aborts on a locale it cannot load ("sudo: unable to resolve
# ..." / audit plugin errors) -- which would make every `sudo -u bench` in
# here fail for a reason that has nothing to do with the install.  Pin it.
export LC_ALL=C.UTF-8 LANG=C.UTF-8

usage() {
  cat <<'USAGE'
Usage: bash setup/bootstrap.sh --target targets/<id>.yaml [options]

  --help                print this and exit
  --target FILE         the target file (or IJ_TARGET=FILE).  Required.
  --agent NAME          the coding agent that does the research (or IJ_AGENT=NAME;
                        default claude).  One of research/adapters/: claude, codex,
                        pi, or an adapter of your own.
  --plan                print the steps this target would run or skip, and exit
  --skip-model          do not download the weights (for dry installs)
  --no-clock            install everything but do not start the run clock
  --reset-clock         restart the run clock even if it is already running.
                        DESTRUCTIVE: the researcher loses the hours already run.
  --reinstall-engines   force `git checkout -f <pinned ref>` in the engine checkouts (a
                        tpu target: copy the trees out of the image again).
                        DESTRUCTIVE: throws away uncommitted engine patches.

Required for a target whose engine is built here (vllm/sglang on inference-judge-base);
nothing is "latest":
  IJ_VLLM_TAG=<tag>       the vLLM release tag to build and test      (e.g. v0.29.0)
  IJ_SGLANG_TAG=<tag>     the SGLang release tag to build and test    (e.g. v0.5.19);
                             needed only when the target serves sglang or lists its tree
                             Both are recorded in /etc/inference-judge-engine-pins and in the image label.

Environment knobs (all optional):
  IJ_REPO=<path>          repo checkout to install from (default: parent of this script)
  IJ_TARGET=<file>        same as --target
  IJ_AGENT=<name>         same as --agent
  HF_TOKEN=<token>           HF token for gated models
  IJ_TORCH_BACKEND=cu130  torch CUDA channel. cu130 is the only supported
                             value; anything else is refused (see gpu_channel)
  IJ_VERIFY_MODEL=1       re-hash every weight file against the Hub manifest (slow)
  IJ_ACCEL_HOURS=24       the run's accelerator-hour budget (clock start)
  IJ_WALL_HOURS=<hours>   the run's wall-clock deadline, hours from clock start
                             (default: the value of IJ_ACCEL_HOURS)
  IJ_HARDEN_FLAGS=""      extra flags for setup/harden_research.sh (e.g. --no-venv)
  IJ_SKIP_MODEL=1         same as --skip-model
  IJ_NO_CLOCK=1           same as --no-clock
  IJ_RESET_CLOCK=1        same as --reset-clock
  IJ_REINSTALL_ENGINES=1  same as --reinstall-engines

Exit code 0 means every critical step passed.  The final checklist marks each
step PASS / WARN / FAIL / SKIP.  A FAILed critical step aborts immediately: the
clock is never started on top of a broken install.
Exit code 3 is a STOP, not a failure: the target is runnable: false, so bootstrap
got the machine as far as the chip gate (tpu) or the gpu gate and printed what to
do before running it again.  A target that is runnable: false never gets the clock or the ruler lock.
USAGE
}

# ------------------------------------------------------------------ settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IJ_REPO="${IJ_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IJ_TARGET="${IJ_TARGET:-}"
IJ_AGENT="${IJ_AGENT:-claude}"
IJ_SKIP_MODEL="${IJ_SKIP_MODEL:-0}"
IJ_VERIFY_MODEL="${IJ_VERIFY_MODEL:-0}"
IJ_NO_CLOCK="${IJ_NO_CLOCK:-0}"
IJ_RESET_CLOCK="${IJ_RESET_CLOCK:-0}"
IJ_REINSTALL_ENGINES="${IJ_REINSTALL_ENGINES:-0}"
IJ_ACCEL_HOURS="${IJ_ACCEL_HOURS:-24}"
IJ_WALL_HOURS="${IJ_WALL_HOURS:-$IJ_ACCEL_HOURS}"
IJ_HARDEN_FLAGS="${IJ_HARDEN_FLAGS:-}"
HF_TOKEN="${HF_TOKEN:-}"
PLAN_ONLY=0

BENCH_DIR=/opt/bench
RESEARCHER_USER=researcher
RESEARCHER_HOME=/home/researcher
RESEARCH_DIR="$RESEARCHER_HOME/research"
VENV_DIR="$RESEARCHER_HOME/.venv"
IMAGE=inference-judge-base:latest
# The driver's side (setup/harden_research.sh creates the user and the dir).
DRIVER_USER=driver
STATE_DIR=/home/driver/research-state
# Every setup log lands here.
LOG_DIR=/var/log/inference-judge
LOOPSTATE=/opt/inference-judge/loop/loopstate.py
LOOP_ENV="${IJ_LOOP_ENV_FILE:-/etc/inference-judge/loop.env}"
# The agent's adapter in the repo (research/adapters/README.md).  setup.sh is
# optional: an adapter without one is installed and logged in by the operator.
ADAPTER_DIR="$IJ_REPO/research/adapters/$IJ_AGENT"
# The installed copies of the target: the ruler's (run.sh reads the id from
# $BENCH_DIR/TARGET and the file from $BENCH_DIR/targets/<id>.yaml), and a
# read-only one the researcher may read (research/AGENTS.md points at it).
TARGET_BENCH=""                                 # $BENCH_DIR/targets/<id>.yaml, set by resolve_target
TARGET_ID_FILE="$BENCH_DIR/TARGET"
TARGET_PUBLIC=/etc/inference-judge/target.yaml
ENGINE_SOURCES_FILE="${IJ_ENGINE_SOURCES_FILE:-/etc/inference-judge/engine-sources}"
# The ruler image F's run.sh measures from (targets/cpu-mock.Dockerfile says so).
RULER_IMAGE="${IJ_RULER_IMAGE:-inference-judge-ruler:latest}"
RULER_DOCKERFILE="${IJ_RULER_DOCKERFILE:-$IJ_REPO/targets/cpu-mock.Dockerfile}"

# The versions of an engine built here are NOT defaulted: see
# require_engine_pins in setup/gpu.sh, which also holds what such a build needs
# of the driver.
ENGINE_PINS_FILE="${IJ_PINS_FILE:-/etc/inference-judge-engine-pins}"

# shellcheck source=setup/target.sh
. "$SCRIPT_DIR/target.sh"
# shellcheck source=setup/tpu.sh
. "$SCRIPT_DIR/tpu.sh"
# shellcheck source=setup/gpu.sh
. "$SCRIPT_DIR/gpu.sh"

# --------------------------------------------------------------- step plumbing
STEP_NAMES=()
STEP_RESULTS=()
STEP_NOTES=()

log()  { printf '\n\033[1m== %s\033[0m  (%s)\n' "$*" "$(date -u +%H:%M:%S)"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   \033[33mWARN\033[0m %s\n' "$*" >&2; }

record() { STEP_NAMES+=("$1"); STEP_RESULTS+=("$2"); STEP_NOTES+=("${3:-}"); }

# step <name> <function> [optional]
#   "optional" steps record WARN and continue; the default is to abort.
#
# This used to be `if ( set -e; "$fn" ); then`.  A compound
# command evaluated as the condition of `if` (or the left side of || / &&) runs
# in a context where errexit is IGNORED, and bash propagates that suppression
# into the subshell even though the subshell re-enables it -- so a step whose
# first command failed carried on and was recorded PASS as long as its last
# command succeeded.  `f || rc=$?` has exactly the same defect.
#
# The fix is to run the subshell as a plain statement, which is not a testing
# context, and to turn the OUTER errexit off just around it so the failure can
# be recorded instead of killing the script before the checklist prints.
step() {
  local name="$1" fn="$2" mode="${3:-critical}" rc=0
  log "$name"
  set +e
  ( set -e; "$fn" )
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    record "$name" PASS
    return 0
  fi
  if [ "$mode" = optional ]; then
    warn "$name failed with exit $rc (non-critical, continuing)"
    record "$name" WARN "exit $rc"
    return 0
  fi
  record "$name" FAIL "exit $rc"
  checklist
  echo "bootstrap aborted at: $name (exit $rc)" >&2
  exit 1
}

skip() { log "$1 (skipped)"; record "$1" SKIP "${2:-}"; }

checklist() {
  echo
  echo "================ bootstrap checklist ================"
  local i
  for i in "${!STEP_NAMES[@]}"; do
    printf '%-6s %s%s\n' "${STEP_RESULTS[$i]}" "${STEP_NAMES[$i]}" \
      "${STEP_NOTES[$i]:+  -- ${STEP_NOTES[$i]}}"
  done
  echo "====================================================="
}

# cd to the target user's home first: bootstrap is run from /root/<repo>, which
# the other users cannot traverse, and tools such as uv walk up from the cwd
# looking for config (uv.toml) and die with "Permission denied" if they can't.
as_researcher() { sudo -u researcher -H bash -lc "cd ~ && $1"; }
as_bench()  { sudo -u bench  -H bash -lc "cd ~ && $1"; }
# agent_login_hint: how the operator logs the agent in, as the researcher.  The
# adapter's setup.sh knows the words; one without a setup.sh leaves it to the
# agent's own documentation.
agent_login_hint() {
  local how=""
  [ -f "$ADAPTER_DIR/setup.sh" ] && how=$(bash "$ADAPTER_DIR/setup.sh" login 2>/dev/null || true)
  if [ -n "$how" ]; then printf 'su - %s, then:  %s\n' "$RESEARCHER_USER" "$how"
  else printf 'su - %s, then log the %s command in (its own documentation says how)\n' "$RESEARCHER_USER" "$IJ_AGENT"; fi
}
# The driver's ledger, as the driver: loopstate.py is the only writer of the clock.
as_driver_loopstate() {
  sudo -u "$DRIVER_USER" -H env IJ_STATE_DIR="$STATE_DIR" IJ_RESEARCH_DIR="$RESEARCH_DIR" \
    IJ_LEGACY_STATE=/nonexistent python3 -I "$LOOPSTATE" "$@"
}

# ----------------------------------------------------------------- the steps
s_preflight() {
  [ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; return 1; }
  install -d -m 0755 "$LOG_DIR"
  # shellcheck disable=SC1091
  . /etc/os-release
  info "os: $PRETTY_NAME  kernel: $(uname -r)"
  [ "${VERSION_ID:-}" = "24.04" ] || warn "expected Ubuntu 24.04, found ${VERSION_ID:-unknown}"
  info "$(target_summary)"
  [ "$T_RUNNABLE" = true ] || warn "target $T_ID is marked runnable: false: run.sh will refuse it, so the clock is not started (--plan shows where this stops)"
  local free_gb need
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  info "free disk on /: ${free_gb}G"
  need=$(disk_need_gb)
  [ "$free_gb" -ge "$need" ] || { echo "need >=${need}G free on / for target $T_ID (provision.min_free_gb, or the default for accelerator.kind $T_KIND)" >&2; return 1; }
  [ -d "$IJ_REPO/bench" ] && [ -d "$IJ_REPO/research" ] \
    || { echo "IJ_REPO=$IJ_REPO does not look like the repo" >&2; return 1; }
  info "repo: $IJ_REPO"
  # The number every promotion is measured against comes from this launcher.
  [ -f "$IJ_REPO/research/experiments/$T_BASELINE/launch.sh" ] \
    || { echo "target $T_ID has no baseline: research/experiments/$T_BASELINE/launch.sh is not in the repo (provision.baseline names another)" >&2; return 1; }
  info "baseline experiment: $T_BASELINE"
}

# disk_need_gb: the free space on / the install needs, in G.  The target knows its
# weights and its image, so it says (provision.min_free_gb: weights + engine image +
# room for experiment images and results); the defaults are for a target that does
# not.  A re-run that already has the weights and the engine image only needs the
# headroom.
DISK_HEADROOM_GB=40
disk_need_gb() {
  local need img="$T_IMAGE"
  need=$(tget_opt provision.min_free_gb "")
  if [ -z "$need" ]; then
    case "$T_KIND" in
      none) need=5 ;;     # a few images
      tpu)  need=40 ;;    # the image and the trees; weights and caches live on the cache disk
      *)    need=100 ;;   # a guess: an engine image, mid-sized weights, a build cache
    esac
  fi
  need=${need%%.*}
  [ "$T_ENGINE_CLASS" = built ] && img="$IMAGE"
  if [ "$need" -gt "$DISK_HEADROOM_GB" ] && [ -n "$T_MODEL_DIR" ] && [ -d "$T_MODEL_DIR" ] \
     && [ -n "$img" ] && docker image inspect "$img" >/dev/null 2>&1; then
    need=$DISK_HEADROOM_GB
  fi
  printf '%s\n' "$need"
}

# The GPU steps live in setup/gpu.sh (gpu_devices, gpu_passthrough, gpu_gate,
# gpu_nsight, and for an engine built here gpu_channel, gpu_host_venv,
# gpu_base_image); plan_steps names them.
# The TPU steps live in setup/tpu.sh (tpu_devices, tpu_mount_disk, tpu_engine_image,
# tpu_gate, tpu_models, tpu_compile_seed, tpu_engine_sources); plan_steps names them.

s_docker() {
  command -v docker >/dev/null || { echo "docker missing" >&2; return 1; }
  docker info >/dev/null || { echo "docker daemon not usable" >&2; return 1; }
  info "docker: $(docker --version)"
  case "$T_KIND" in
    gpu) gpu_passthrough ;;
    tpu)
      # The passthrough proof needs the engine image, which is not here yet:
      # tpu_gate runs it after the image is loaded.
      info "tpu: chip passthrough is proved by the chip gate, after the image is loaded" ;;
    *) info "accelerator.kind $T_KIND: no passthrough to probe" ;;
  esac
}

APT_PACKAGES=(build-essential ca-certificates cmake curl git gnupg jq ninja-build
              numactl pciutils python3-venv python3-yaml rsync shellcheck sudo tmux unzip wget)
s_apt() {
  export DEBIAN_FRONTEND=noninteractive
  local p missing=()
  for p in "${APT_PACKAGES[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    info "all ${#APT_PACKAGES[@]} apt packages already present"; return 0
  fi
  info "installing: ${missing[*]}"
  apt-get update
  apt-get install -y --no-install-recommends "${missing[@]}"
  info "apt packages present"
}

s_node() {
  if command -v node >/dev/null && [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -ge 22 ]; then
    info "node $(node -v) already installed"; return 0
  fi
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
  info "node $(node -v), npm $(npm -v)"
}

s_users() {
  id bench  >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/bench --shell /bin/bash bench
  id researcher >/dev/null 2>&1 || useradd           --create-home --home-dir "$RESEARCHER_HOME" --shell /bin/bash researcher
  # The researcher must be able to talk to no one else's files; no docker group and no
  # sudo at all (the driver runs the ruler on its behalf, through the queue).
  if id -nG researcher | tr ' ' '\n' | grep -qx docker; then
    warn "researcher was in the docker group; removing it (the agent must not drive the accelerator directly)"
    gpasswd -d researcher docker >/dev/null
  fi
  # Ubuntu >= 21.04 creates home directories mode 0750, so
  # `bench` cannot TRAVERSE /home/researcher and every `run.sh <id>` would fail at
  # "copy experiments/<id>" on a fresh box.  0751 gives traverse without list.
  chmod 0751 "$RESEARCHER_HOME"
  info "users: $(id bench | cut -d' ' -f1-2), $(id researcher | cut -d' ' -f1-2)"
  info "$RESEARCHER_HOME is mode $(stat -c '%a' "$RESEARCHER_HOME") (bench may traverse, not list)"
}

# Everything bench has to read out of the researcher's home, and the check that it
# actually can.  Called after the research tree and the engine checkouts exist,
# and again after setup/harden_research.sh, which sets /home/researcher to 0750
# (bench is not in group researcher, so that alone would stop every run.sh).
# The engine source trees of the target: engine.sources, each /home/researcher/<name>-src.
engine_src_dirs() {
  local n
  for n in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do printf '%s/%s-src\n' "$RESEARCHER_HOME" "$n"; done
}

ensure_bench_access() {
  chmod 0751 "$RESEARCHER_HOME"
  local d
  # shellcheck disable=SC2046  # tree names are [a-z0-9_-]
  for d in "$RESEARCH_DIR" "$RESEARCH_DIR/experiments" $(engine_src_dirs); do
    [ -d "$d" ] && chmod 0755 "$d"
  done
  [ -d "$RESEARCH_DIR/experiments" ] && find "$RESEARCH_DIR/experiments" -mindepth 1 -type d -exec chmod 0755 {} +
  return 0
}

assert_bench_access() {
  local d f fail=0
  # shellcheck disable=SC2046
  for d in "$RESEARCH_DIR/experiments" $(engine_src_dirs); do
    [ -d "$d" ] || continue
    if sudo -u bench test -r "$d" && sudo -u bench test -x "$d"; then
      info "bench can read $d"
    else
      echo "bench cannot read $d -- every run.sh would fail at the copy step" >&2
      echo "  $(stat -c '%U:%G %a' "$d") ; $RESEARCHER_HOME is $(stat -c '%a' "$RESEARCHER_HOME")" >&2
      fail=1
    fi
  done
  # The exact thing run.sh does first: can bench see an experiment's launch.sh?
  f=$(find "$RESEARCH_DIR/experiments" -mindepth 2 -maxdepth 2 -name launch.sh 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    if sudo -u bench test -r "$f"; then info "bench can read ${f#"$RESEARCH_DIR"/}"
    else echo "bench cannot read ${f#"$RESEARCH_DIR"/}" >&2; fail=1; fi
  fi
  [ "$fail" -eq 0 ]
}

s_uv() {
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
  fi
  chmod 0755 /usr/local/bin/uv /usr/local/bin/uvx 2>/dev/null || true
  info "uv: $(uv --version)"
}

# Where the source of an engine this repo knows lives; any other tree names its own
# (provision.sources.<name>.url).
engine_source_url() {
  local url
  url=$(tget_opt "provision.sources.$1.url" "")
  if [ -z "$url" ]; then
    case "$1" in
      vllm)   url=https://github.com/vllm-project/vllm.git ;;
      sglang) url=https://github.com/sgl-project/sglang.git ;;
    esac
  fi
  printf '%s' "$url"
}

# The ref a tree is checked out at: the operator's release tag for an engine built
# here, else what the target pins (provision.sources.<name>.ref).
engine_source_ref() {
  if [ "$T_ENGINE_CLASS" = built ]; then gpu_engine_tag "$1"
  else tget_opt "provision.sources.$1.ref" ""; fi
}

# s_engine_sources: /home/researcher/<name>-src for every name in engine.sources, and
# for no other tree, each at a pinned ref.  The ruler copies every one of them into
# a launch and refuses to start without them.  Every name is checked before the
# first one is fetched (no network before a refusal).
s_engine_sources() {
  local n url ref dir rev names=() urls=() refs=() i
  [ "${#T_SOURCES[@]}" -gt 0 ] || return 0
  for n in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    url=$(engine_source_url "$n"); ref=$(engine_source_ref "$n")
    if [ -z "$url" ]; then
      echo "engine.sources names '$n', and provision.sources.$n.url is missing in $T_ID: where is its source?" >&2; return 1
    fi
    [[ "$url" =~ ^[A-Za-z0-9._:/@+~-]+$ ]] || { echo "provision.sources.$n.url is '$url': not a plain git URL" >&2; return 1; }
    if [ -z "$ref" ]; then
      echo "engine.sources names '$n', and provision.sources.$n.ref is missing in $T_ID:" >&2
      echo "  pin the commit the engine image was built from (a 40-hex commit or a release tag)" >&2; return 1
    fi
    if ! [[ "$ref" =~ ^[0-9a-f]{40}$ ]] && ! [[ "$ref" =~ $GPU_RELEASE_TAG_RE ]]; then
      echo "the ref for $n is '$ref': a 40-hex commit or a release tag like v0.29.0 is required (never main or a branch)" >&2; return 1
    fi
    names+=("$n"); urls+=("$url"); refs+=("$ref")
  done
  for i in "${!names[@]}"; do
    n="${names[$i]}"; dir="$RESEARCHER_HOME/$n-src"
    checkout_at_ref "${urls[$i]}" "$dir" "${refs[$i]}"
    chown -R researcher:researcher "$dir"
    # bench rsyncs these into the container at run start, so it must be able to
    # traverse them.
    chmod 0755 "$dir"
    info "$n at $(cd "$dir" && git describe --tags --always)"
    # A prebuilt image that records the commit it was built from should agree with
    # the tree the researcher edits, or an edit is a patch against other code.
    if [ "$T_ENGINE_CLASS" = pulled ] && [ "$n" = "$T_ENGINE" ]; then
      rev=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$T_IMAGE" 2>/dev/null || true)
      if [[ "$rev" =~ ^[0-9a-f]{40}$ ]] && [[ "${refs[$i]}" =~ ^[0-9a-f]{40}$ ]] && [ "$rev" != "${refs[$i]}" ]; then
        warn "$n-src is pinned at ${refs[$i]}, but the image says it was built from $rev (label org.opencontainers.image.revision): fix provision.sources.$n.ref in $T_ID"
      fi
    fi
  done
}

# checkout_at_ref URL DIR REF.  A checkout that already exists
# is left exactly as it is unless --reinstall-engines says otherwise.  A commit is
# fetched into DIR.new and moved into place, so a failed fetch leaves nothing that
# the next run would take for a checkout.
checkout_at_ref() {
  local url="$1" dir="$2" ref="$3" is_tag=0
  [[ "$ref" =~ ^[0-9a-f]{40}$ ]] || is_tag=1
  if [ -d "$dir/.git" ]; then
    if [ "$IJ_REINSTALL_ENGINES" = 1 ]; then
      warn "--reinstall-engines: forcing $dir back to $ref, discarding local changes"
      if [ "$is_tag" = 1 ]; then
        as_researcher "cd '$dir' && git fetch --tags --depth 1 origin '$ref' && git checkout -f '$ref'"
      else
        as_researcher "cd '$dir' && git fetch --depth 1 origin '$ref' && git checkout -f FETCH_HEAD"
      fi
    else
      local at dirty
      at=$(as_researcher "cd '$dir' && git describe --tags --always 2>/dev/null" || echo unknown)
      dirty=$(as_researcher "cd '$dir' && git status --porcelain 2>/dev/null | wc -l" || echo 0)
      info "$dir already exists at $at (${dirty} modified path(s)); leaving it alone"
      info "  use --reinstall-engines to force it back to $ref"
    fi
  elif [ "$is_tag" = 1 ]; then
    as_researcher "git clone --depth 1 --branch '$ref' '$url' '$dir'"
  else
    # `git clone --branch` takes no commit; fetching one by its sha does.
    if ! as_researcher "rm -rf '$dir.new' && git init -q '$dir.new' && cd '$dir.new' \
          && git remote add origin '$url' && git fetch --depth 1 origin '$ref' && git checkout -q FETCH_HEAD"; then
      rm -rf "$dir.new"
      echo "could not fetch $ref from $url: is provision.sources.<name>.ref a commit of that repository?" >&2
      return 1
    fi
    mv "$dir.new" "$dir"
  fi
}

# ------------------------------------------------------------------- weights
# The manifest is the Hub's, not ours: for every file of the
# pinned revision, the sha256 the Hub stores for LFS files, or the git blob
# sha1 for small non-LFS files.  The local files are then checked against it,
# so a truncated or swapped shard fails here instead of serving garbage.  A
# file count (the old check) passes a truncated shard.
#
# Manifest format, one line per file:  <sha256|gitsha1> <hex> <relative path>
#
# verify_model_manifest DIR MANIFEST [full|quick]
#   full : hash every file (sha256, or git blob sha1) and compare; used on a
#          fresh download and with IJ_VERIFY_MODEL=1
#   quick: every file named in the manifest is present (a re-run's fast check)
# Prints one FAIL line per bad file and an OK/FAIL summary; exit 1 on any FAIL.
verify_model_manifest() {
  python3 -I - "$1" "$2" "${3:-full}" <<'PY'
import hashlib, os, sys
root, manifest, mode = sys.argv[1], sys.argv[2], sys.argv[3]
bad, n = [], 0
for line in open(manifest, encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip():
        continue
    algo, want, rel = line.split(" ", 2)
    n += 1
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        bad.append(f"missing: {rel}"); continue
    if mode == "quick":
        continue
    if algo == "sha256":
        h = hashlib.sha256()
    elif algo == "gitsha1":
        h = hashlib.sha1(b"blob %d\0" % os.path.getsize(p))
    else:
        bad.append(f"unknown hash {algo}: {rel}"); continue
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        bad.append(f"hash mismatch: {rel}")
for b in bad[:20]:
    print("FAIL " + b)
print(f"{'FAIL' if bad else 'OK'} {n - len(bad)}/{n} files {'present' if mode == 'quick' else 'verified'}")
sys.exit(1 if bad else 0)
PY
}

s_model() {
  local dir="$T_MODEL_DIR" rev="$T_MODEL_REV" manifest
  [ -n "$dir" ] || { echo "target $T_ID names model.hf_id but no model.local_dir" >&2; return 1; }
  if ! [[ "$rev" =~ ^[0-9a-f]{40}$ ]]; then
    echo "model.revision for $T_MODEL_ID is '$rev': a 40-hex commit sha is required (never main/unpinned)" >&2
    echo "  find it with: huggingface-cli repo-info $T_MODEL_ID  (or the Hub's 'Files' tab)" >&2
    return 1
  fi
  manifest="${dir%/}.sha256"
  if [ -s "$manifest" ] && [ -d "$dir" ] && [ "$(head -1 "$manifest.revision" 2>/dev/null)" = "$rev" ]; then
    if [ "$IJ_VERIFY_MODEL" = 1 ]; then
      info "re-hashing every file against the Hub manifest (slow)"
      verify_model_manifest "$dir" "$manifest" full
    else
      verify_model_manifest "$dir" "$manifest" quick
    fi
    return 0
  fi
  mkdir -p "$dir"
  chmod -R u+w "$dir"
  # provision.model_ignore: files of the Hub repo this target does not serve
  # (other formats of the same weights), as a list of globs.  Never downloaded,
  # never in the manifest.
  MODEL_ID="$T_MODEL_ID" MODEL_REV="$rev" MODEL_DIR="$dir" MANIFEST="$manifest" HF_TOKEN="$HF_TOKEN" \
  MODEL_IGNORE="$(tget_opt provision.model_ignore "[]")" \
  uv run --python 3.12 --with huggingface_hub python - <<'PY'
import json, os
from huggingface_hub import HfApi, snapshot_download
tok = os.environ.get("HF_TOKEN") or None
repo, rev, out = os.environ["MODEL_ID"], os.environ["MODEL_REV"], os.environ["MODEL_DIR"]
ignore = [str(g) for g in json.loads(os.environ.get("MODEL_IGNORE") or "[]")] + [".cache/*"]
p = snapshot_download(repo_id=repo, revision=rev, local_dir=out, max_workers=8, token=tok,
                      ignore_patterns=ignore)
print("downloaded to", p)
# The Hub's own hashes for exactly this revision.
import fnmatch
info = HfApi().model_info(repo, revision=rev, files_metadata=True, token=tok)
lines = []
for s in sorted(info.siblings, key=lambda s: s.rfilename):
    if any(fnmatch.fnmatch(s.rfilename, g) for g in ignore):
        continue
    if s.lfs is not None and getattr(s.lfs, "sha256", None):
        lines.append(f"sha256 {s.lfs.sha256} {s.rfilename}")
    elif s.blob_id:
        lines.append(f"gitsha1 {s.blob_id} {s.rfilename}")
open(os.environ["MANIFEST"], "w").write("\n".join(lines) + "\n")
print(f"hub manifest: {len(lines)} files")
PY
  rm -rf "$dir/.cache" 2>/dev/null || true
  info "verifying every file against the Hub manifest"
  verify_model_manifest "$dir" "$manifest" full
  printf '%s\n' "$rev" > "$manifest.revision"
  chmod 0444 "$manifest" "$manifest.revision"
  chown -R root:root "$dir"
  find "$dir" -type d -exec chmod 0555 {} +
  find "$dir" -type f -exec chmod 0444 {} +
  info "model is root-owned and read-only: $(du -sh "$dir" | cut -f1), $(wc -l < "$manifest") files"
}

# ------------------------------------------------------------------- images
# run.sh empties root-owned private copies with a throwaway alpine container and
# snapshots the winning source tree with a throwaway git container (finding N1).
# Pull both now so neither depends on the network in the middle of a run.
s_helper_images() {
  docker image inspect alpine:3 >/dev/null 2>&1 || docker pull -q alpine:3
  docker image inspect alpine/git:latest >/dev/null 2>&1 || docker pull -q alpine/git:latest
}

# build_image_from <tag> <dockerfile>: build once; rebuild when the Dockerfile changed.
build_image_from() {
  local tag="$1" df="$2" sum have
  sum=$(sha256sum "$df" | cut -c1-16)
  have=$(docker image inspect -f '{{index .Config.Labels "inference-judge.dockerfile"}}' "$tag" 2>/dev/null || true)
  if [ "$have" = "$sum" ]; then info "$tag already built from $(basename "$df") ($sum)"; return 0; fi
  docker build -q -t "$tag" --label "inference-judge.dockerfile=$sum" -f "$df" "$(dirname "$df")"
  info "$tag built from $(basename "$df"): $(docker image inspect -f '{{.Size}}' "$tag") bytes"
}

# The engine image the target names: built from targets/<id>.Dockerfile for the
# mock, pulled (by tag or digest) for anything else that is not built here.
s_engine_image() {
  case "$T_ENGINE_CLASS" in
    mock)
      local df="$IJ_REPO/targets/$T_ID.Dockerfile"
      [ -f "$df" ] || { echo "engine mock needs $df to build $T_IMAGE" >&2; return 1; }
      build_image_from "$T_IMAGE" "$df" ;;
    pulled)
      case "$T_IMAGE" in
        *@sha256:*) ;;
        *) warn "engine.image '$T_IMAGE' is a tag, not a digest (name@sha256:...): a tag can move, and results are only valid for one image" ;;
      esac
      docker image inspect "$T_IMAGE" >/dev/null 2>&1 && { info "$T_IMAGE present"; return 0; }
      docker pull "$T_IMAGE" ;;
  esac
}

# ruler_libs PYTHON...: PYTHON can import every library the ruler imports.  The ruler's
# python runs in two places, the bench venv (bench/pyproject.toml) and the ruler
# image ($RULER_DOCKERFILE); both lists must carry these.  tokenizers counts a tps
# target's tokens from the model's tokenizer.json: without it the first measured
# run is a hard stop.  check.sh asks the same question through this function.
RULER_LIBS="httpx numpy yaml tokenizers"
ruler_libs() {
  # shellcheck disable=SC2086  # RULER_LIBS is a word list
  "$@" -c 'import importlib.util, sys
missing = [m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("this python is missing: " + " ".join(missing))
print("imports " + " ".join(sys.argv[1:]))' $RULER_LIBS
}

# The image the ruler's measuring side runs in, when run.sh uses one.
s_ruler_image() {
  [ -f "$RULER_DOCKERFILE" ] || { echo "no $RULER_DOCKERFILE to build $RULER_IMAGE from" >&2; return 1; }
  build_image_from "$RULER_IMAGE" "$RULER_DOCKERFILE"
  ruler_libs docker run --rm --network none "$RULER_IMAGE" python3 \
    || { echo "$RULER_IMAGE cannot import what the ruler needs: add it to $RULER_DOCKERFILE and re-run" >&2; return 1; }
}

# --------------------------------------------------------------- the ruler
s_bench() {
  mkdir -p "$BENCH_DIR"
  # --delete keeps the install clean, but rsync does not delete files that are
  # excluded, so the run's own state survives a re-bootstrap: results, the
  # stored leaderboard distributions, the canary baseline, the stored
  # reference and the lock files.
  rsync -a --delete \
    --exclude '.venv/' --exclude '.venv-*/' --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.cache/' --exclude '.pytest_cache/' --exclude '*.egg-info/' --exclude '/bench/' \
    --exclude 'results/' --exclude 'leaderboard/' \
    --include '/canary/prompts.json' --exclude '/canary/*.json' \
    --exclude '/reference/' --exclude '/calibration/' --exclude '/.allow-baseline' --exclude '/cache/' \
    --exclude '/TARGET' --exclude '/targets/' --exclude '/RULER_LOCK' --exclude '/LOCK_PATH' \
    "$IJ_REPO/bench/" "$BENCH_DIR/"
  mkdir -p "$BENCH_DIR/results" "$BENCH_DIR/leaderboard"
  chown -R bench:bench "$BENCH_DIR"
  # 0751 on the top directory: /opt/bench/results/<id> must be world-readable so
  # the researcher can read results.json, and that needs +x on /opt/bench.  The
  # researcher still cannot list /opt/bench or read the harness or the pools.
  chmod 0751 "$BENCH_DIR"
  # results/ is pruned from the walk; run.sh owns the modes
  # inside it, so a re-bootstrap does not make past runs private again.
  # cache/ (the compile-cache seed) is pruned the same way: it is filled by tpu_compile_seed.
  find "$BENCH_DIR" -mindepth 1 \( -path "$BENCH_DIR/results" -o -path "$BENCH_DIR/cache" \) -prune -o -type d -exec chmod 0750 {} +
  find "$BENCH_DIR" -mindepth 1 \( -path "$BENCH_DIR/results" -o -path "$BENCH_DIR/cache" \) -prune -o -type f -exec chmod 0640 {} +
  find "$BENCH_DIR" -mindepth 1 \( -path "$BENCH_DIR/results" -o -path "$BENCH_DIR/cache" \) -prune -o -name '*.sh' -exec chmod 0750 {} +
  chmod 0755 "$BENCH_DIR/results"
  local kept
  kept=$(find "$BENCH_DIR/results" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  info "results/ left untouched: $kept existing run(s)"
  [ -x "$BENCH_DIR/run.sh" ] || { echo "$BENCH_DIR/run.sh missing or not executable" >&2; return 1; }
  # bench needs docker to build and run experiment containers.
  usermod -aG docker bench
  info "bench installed: $(find "$BENCH_DIR" -mindepth 1 -maxdepth 1 -printf '%f ' | head -c 400)"
}

s_bench_venv() {
  if [ ! -x "$BENCH_DIR/.venv/bin/python" ]; then
    as_bench "uv venv --python 3.12 '$BENCH_DIR/.venv'"
  fi
  if [ -f "$BENCH_DIR/pyproject.toml" ]; then
    as_bench "uv pip install --python '$BENCH_DIR/.venv/bin/python' -e '$BENCH_DIR'"
  elif [ -f "$BENCH_DIR/requirements.txt" ]; then
    as_bench "uv pip install --python '$BENCH_DIR/.venv/bin/python' -r '$BENCH_DIR/requirements.txt'"
  else
    as_bench "uv pip install --python '$BENCH_DIR/.venv/bin/python' httpx numpy pyyaml fastapi uvicorn huggingface_hub tokenizers"
  fi
  chown -R bench:bench "$BENCH_DIR/.venv"
  chmod 0750 "$BENCH_DIR/.venv"
  ruler_libs sudo -u bench -H "$BENCH_DIR/.venv/bin/python" \
    || { echo "the bench venv cannot import what the ruler needs: add it to $BENCH_DIR/pyproject.toml and re-run" >&2; return 1; }
  # run.sh calls plain `python3` (sudo resets the environment and the sudoers
  # line is exactly one command), so the same libraries also go into the system
  # interpreter; Ubuntu 24.04 marks it externally managed, hence
  # --break-system-packages.  The box exists for this one job.
  if ! python3 -c 'import httpx,numpy,yaml' 2>/dev/null; then
    uv pip install --system --break-system-packages httpx numpy pyyaml \
      || pip3 install --break-system-packages httpx numpy pyyaml
  fi
  python3 -c 'import httpx,numpy,yaml;print("system python3 ok for run.sh")'
}

# The target, installed twice: the ruler's copy (bench-owned; part of the ruler
# hash) and a read-only public copy the researcher reads instead of guessing the
# hardware (research/AGENTS.md points at it).  Plus the stored reference R, if
# the repo carries one for this target.
s_target() {
  local n
  install -d -o root -g root -m 0755 "$(dirname "$TARGET_PUBLIC")"
  install -d -o bench -g bench -m 0750 "$(dirname "$TARGET_BENCH")"
  install -o bench -g bench -m 0640 "$T_FILE" "$TARGET_BENCH"
  printf '%s\n' "$T_ID" > "$TARGET_ID_FILE.tmp" && mv "$TARGET_ID_FILE.tmp" "$TARGET_ID_FILE"
  chown bench:bench "$TARGET_ID_FILE"; chmod 0640 "$TARGET_ID_FILE"
  rm -f "$BENCH_DIR/target.yaml"            # the pre-TARGET layout
  install -o root -g root -m 0644 "$T_FILE" "$TARGET_PUBLIC"
  local ref
  ref=$(tget_opt reference.dir "")
  if [ -n "$ref" ] && [ -d "$IJ_REPO/$ref" ] && [ ! -e "$BENCH_DIR/$ref" ]; then
    mkdir -p "$BENCH_DIR/$(dirname "$ref")"
    cp -R "$IJ_REPO/$ref" "$BENCH_DIR/$ref"
    local top="$BENCH_DIR/${ref%%/*}"
    chown -R bench:bench "$top"
    find "$top" -type d -exec chmod 0550 {} +
    find "$top" -type f -exec chmod 0440 {} +
    info "stored reference installed read-only: $BENCH_DIR/$ref"
  fi
  # The engine trees the researcher may edit, one name per line.  Root-owned, so the
  # edit hook, the preflight and check.sh read the target's list from a file the
  # researcher cannot change (and sudo's environment stripping cannot lose).
  for n in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do printf '%s\n' "$n"; done > "$ENGINE_SOURCES_FILE.tmp" \
    && mv "$ENGINE_SOURCES_FILE.tmp" "$ENGINE_SOURCES_FILE"
  chown root:root "$ENGINE_SOURCES_FILE"; chmod 0644 "$ENGINE_SOURCES_FILE"
  sudo -u bench python3 -I "$BENCH_DIR/target.py" validate "$TARGET_BENCH"
  info "target $T_ID installed: $TARGET_ID_FILE + $TARGET_BENCH (ruler), $TARGET_PUBLIC (read-only, for the researcher)"
}

s_research() {
  local protect=()
  if [ -d "$RESEARCH_DIR/.git" ]; then
    info "research repo exists; preserving notebook, leaderboard, hypotheses, plan and queue"
    protect=(--exclude 'LAB_NOTEBOOK.md' --exclude 'LEADERBOARD.md' \
             --exclude 'HYPOTHESES.md' --exclude 'HYPOTHESES.jsonl' --exclude 'PLAN.md' \
             --exclude 'SUMMARY.md' --exclude 'NEXT.md' --exclude 'ALERTS.md' --exclude 'queue/' \
             --exclude 'loop/state.json' --exclude 'audit.log' --exclude 'loop/*.log')
  fi
  mkdir -p "$RESEARCH_DIR"
  rsync -a "${protect[@]+"${protect[@]}"}" \
    --exclude '.git/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
    "$IJ_REPO/research/" "$RESEARCH_DIR/"
  mkdir -p "$RESEARCH_DIR/profiles"
  # The agent's own files (CLAUDE.md and .claude/ for Claude Code, .codex/ for
  # Codex) sit at the top of the research repo.
  if [ -d "$ADAPTER_DIR/project" ]; then
    cp -R "$ADAPTER_DIR/project/." "$RESEARCH_DIR/"
  fi
  chown -R "$RESEARCHER_USER:$RESEARCHER_USER" "$RESEARCH_DIR"
  chmod +x "$RESEARCH_DIR"/guard/*.sh "$RESEARCH_DIR"/loop/*.sh
  find "$RESEARCH_DIR/adapters" -mindepth 2 -maxdepth 2 \( -name session.sh -o -name setup.sh \) -exec chmod +x {} +
  find "$RESEARCH_DIR/experiments" -mindepth 2 -maxdepth 2 -name launch.sh -exec chmod +x {} +
  ensure_bench_access
  as_researcher "git config --global user.name  'inference-judge researcher'"
  as_researcher "git config --global user.email 'researcher@inference-judge.local'"
  as_researcher "git config --global init.defaultBranch main"
  if [ ! -d "$RESEARCH_DIR/.git" ]; then
    as_researcher "cd '$RESEARCH_DIR' && git init -q && git add -A && git commit -qm 'bootstrap: research repo installed'"
  fi
  info "research installed at $RESEARCH_DIR"
}

# The researcher's coding agent.  Not installed when its command (the adapter's
# name) is already on the PATH (an operator's install, or the local box's
# stub).  An adapter with no setup.sh cannot be installed from here.
s_agent() {
  if ! command -v "$IJ_AGENT" >/dev/null; then
    if [ -f "$ADAPTER_DIR/setup.sh" ]; then
      bash "$ADAPTER_DIR/setup.sh" install
    else
      warn "the $IJ_AGENT adapter has no setup.sh: install the \`$IJ_AGENT\` command yourself, on the PATH of user $RESEARCHER_USER, before the loop starts"
      return 0
    fi
  fi
  info "agent $IJ_AGENT: $("$IJ_AGENT" --version 2>/dev/null || echo 'installed, version unavailable')"
  info "log in ONCE as the researcher before starting the loop:  $(agent_login_hint)"
}

# The trust boundary (setup/harden_research.sh): the driver user, root-owned
# loop, guard and adapters, the agent's managed settings, the driver's sudoers (the only route to
# run.sh), accelerator devices to group bench, systemd units, /etc/inference-judge/loop.env.
# Then bench's access to the experiments again: harden sets /home/researcher 0750.
s_harden() {
  # shellcheck disable=SC2086  # a word list on purpose
  AGENT="$IJ_AGENT" bash "$SCRIPT_DIR/harden_research.sh" $IJ_HARDEN_FLAGS
  ensure_bench_access
  assert_bench_access
}

# sudo replaces PATH with secure_path, so an agent outside it must be named.
s_loop_env() {
  local env="$LOOP_ENV" c
  [ -f "$env" ] || { echo "$env missing (harden_research.sh installs it)" >&2; return 1; }
  c=$(command -v "$IJ_AGENT" || true)
  case "$c" in
    ""|"/usr/local/bin/$IJ_AGENT"|"/usr/bin/$IJ_AGENT"|"/bin/$IJ_AGENT") : ;;
    *) if ! grep -q '^IJ_AGENT_BIN=' "$env"; then
         printf 'IJ_AGENT_BIN=%s\n' "$c" >> "$env"
         info "$IJ_AGENT is outside sudo's secure_path; IJ_AGENT_BIN=$c added to $env"
       fi ;;
  esac
  info "$env: $(grep -c '^[A-Z]' "$env") settings"
}

# ------------------------------------------------------------------- the clock
# The clock lives in the driver's ledger (events.jsonl, a run_config event),
# written only by loopstate.py, as driver.  The researcher cannot move it.
#
# A re-bootstrap in hour 9 must not give the researcher 24
# fresh hours.  Resuming never moves the clock; --reset-clock says so out loud.
s_clock() {
  local existing
  existing=$(as_driver_loopstate get clock_start)
  if [ -n "$existing" ] && [ "$existing" != null ] && [ "$existing" != None ]; then
    if [ "$IJ_RESET_CLOCK" != 1 ]; then
      info "clock already started at epoch $existing -- left alone"
      info "  use --reset-clock if you really mean to give the researcher a new run"
      lock_ruler
      return 0
    fi
    warn "--reset-clock: replacing the clock that started at epoch $existing"
    # A new run locks the ruler as it is now (DESIGN 2.6).
    rm -f "$BENCH_DIR/RULER_LOCK"
  fi
  as_driver_loopstate start-clock --accel-hours "$IJ_ACCEL_HOURS" --wall-hours "$IJ_WALL_HOURS"
  lock_ruler
}

# lock_ruler: DESIGN 2.6, the ruler lock, written at clock start.  The
# driver's user asks the ruler to lock itself through its one sudo route; from
# then on run.sh refuses to measure if any ruler file or the target changes.
lock_ruler() {
  grep -qs -- '--lock-ruler' "$BENCH_DIR/run.sh" || { info "this run.sh has no --lock-ruler; nothing to lock"; return 0; }
  if [ -e "$BENCH_DIR/RULER_LOCK" ]; then info "ruler already locked: $BENCH_DIR/RULER_LOCK"; return 0; fi
  sudo -u "$DRIVER_USER" -H sudo -n -u bench "$BENCH_DIR/run.sh" --lock-ruler
}

# -------------------------------------------------------------------- the plan
# plan_steps prints the install for the loaded target, one line per step:
#     step|<name>|<function>|critical|optional
#     skip|<name>|<reason>
#     stop|<name>|<function>     run_plan prints <function>'s advice and exits 3
# It is the single place that decides what runs for which target; main runs it,
# --plan prints it (and the tests read it).  It only reads T_* and the flags.
plan_steps() {
  local stops=0
  echo "step|preflight (root, os, disk, repo)|s_preflight|critical"
  case "$T_KIND" in
    gpu)  echo "step|gpu: nvidia-smi works and shows $T_CHIPS GPU(s), as the target describes|gpu_devices|critical" ;;
    tpu)  echo "step|tpu: the host shows all $T_CHIPS chips|tpu_devices|critical" ;;
    *)    echo "skip|accelerator driver (nvidia-smi, CUDA)|accelerator.kind $T_KIND" ;;
  esac
  # A prebuilt engine image is its own passthrough probe: the gpu gate, below.
  if [ "$T_KIND" = gpu ] && [ "$T_ENGINE_CLASS" != pulled ]; then echo "step|docker: daemon + GPU passthrough|s_docker|critical"
  else echo "step|docker: daemon|s_docker|critical"; fi
  echo "step|apt packages|s_apt|critical"
  if [ "$T_KIND" = gpu ]; then echo "step|nsight-systems (nsys)|gpu_nsight|optional"
  else echo "skip|nsight-systems (nsys)|accelerator.kind $T_KIND"; fi
  # Node exists only to install the agent; an operator's install (or the
  # local box's stub) already on the PATH needs neither, and an adapter with
  # no setup.sh installs nothing.
  if command -v "$IJ_AGENT" >/dev/null 2>&1; then
    echo "skip|node 22 + npm|$IJ_AGENT already on PATH"
  elif [ ! -f "$ADAPTER_DIR/setup.sh" ]; then
    echo "skip|node 22 + npm|the $IJ_AGENT adapter has no setup.sh"
  else
    echo "step|node 22 + npm|s_node|critical"
  fi
  echo "step|users bench and researcher|s_users|critical"
  echo "step|uv installed system-wide|s_uv|critical"
  if [ "$T_KIND" = tpu ]; then
    if [ -n "$T_DISK_LABEL" ]; then echo "step|tpu: cache disk $T_DISK_LABEL mounted at $T_DISK_MOUNT|tpu_mount_disk|critical"
    else echo "skip|cache disk|target names no provision.cache_disk"; fi
  fi
  case "$T_ENGINE_CLASS" in
    built)
      echo "step|engine channel (driver -> cu130 + tags)|gpu_channel|critical"
      echo "step|clone $(join_and ${T_SOURCES[@]+"${T_SOURCES[@]}"}) at release tags|s_engine_sources|critical"
      echo "step|researcher venv (torch, triton, engines)|gpu_host_venv|critical"
      echo "step|build $IMAGE|gpu_base_image|critical" ;;
    mock)
      echo "skip|engine build (vLLM/SGLang from pinned tags)|engine $T_ENGINE"
      echo "step|engine image $T_IMAGE (targets/$T_ID.Dockerfile)|s_engine_image|critical" ;;
    pulled)
      echo "skip|engine build (vLLM/SGLang from pinned tags)|engine $T_ENGINE uses a prebuilt image"
      if [ "$T_KIND" = tpu ]; then
        echo "step|engine image (docker load from the disk's tar, else pull; digest checked)|tpu_engine_image|critical"
        echo "step|tpu: chip gate (jax sees $T_CHIPS chips in the engine image)|tpu_gate|critical"
        # A sketch of a target is made runnable on the VM: its launch flags can
        # only be proven there, and the seal probe needs exactly the steps above.
        if [ "$T_RUNNABLE" != true ]; then
          stops=1
          echo "stop|target is runnable: false (probe the seal, edit the target, run this again)|tpu_not_runnable"
        fi
        if [ "${#T_SOURCES[@]}" -eq 0 ]; then echo "skip|engine source trees|the target lists none (engine.sources)"
        else echo "step|engine source trees (${T_SOURCES[*]}) from the image|tpu_engine_sources|critical"; fi
      else
        echo "step|engine image $T_IMAGE (pull)|s_engine_image|critical"
        if [ "$T_KIND" = gpu ]; then
          echo "step|gpu: gate ($T_CHIPS GPU(s) visible in the engine image)|gpu_gate|critical"
          # As for a tpu sketch: what the file claims can only be seen on the box.
          if [ "$T_RUNNABLE" != true ]; then
            stops=1
            echo "stop|target is runnable: false (read the target's header, finish its edits, run this again)|gpu_not_runnable"
          fi
        fi
        if [ "${#T_SOURCES[@]}" -eq 0 ]; then echo "skip|engine source trees|the target lists none (engine.sources)"
        else echo "step|engine source trees (${T_SOURCES[*]}) at the target's pinned refs|s_engine_sources|critical"; fi
      fi ;;
    *)
      echo "skip|engine image|engine $T_ENGINE names no image" ;;
  esac
  if grep -qs 'IJ_RULER_IMAGE' "$IJ_REPO/bench/run.sh"; then
    echo "step|ruler image $RULER_IMAGE|s_ruler_image|critical"
  else
    echo "skip|ruler image|bench/run.sh does not use one"
  fi
  echo "step|helper images (alpine, alpine/git)|s_helper_images|optional"
  if [ -z "$T_MODEL_ID" ]; then
    echo "skip|model download|target names no model.hf_id"
  elif [ "$IJ_SKIP_MODEL" = 1 ]; then
    echo "skip|model download|--skip-model"
  elif [ "$T_KIND" = tpu ] && [ -n "$T_DISK_HF" ]; then
    echo "step|model $T_MODEL_ID@${T_MODEL_REV:-?} from the disk's Hub cache (flat, hash-checked)|tpu_models|critical"
  else
    echo "step|model $T_MODEL_ID@${T_MODEL_REV:-?} + Hub sha256 manifest|s_model|critical"
  fi
  echo "step|install /opt/bench|s_bench|critical"
  echo "step|bench venv|s_bench_venv|critical"
  if [ "$T_KIND" = tpu ] && [ -n "$T_COMPILE_SEED" ]; then
    echo "step|compile-cache seed $T_COMPILE_SEED|tpu_compile_seed|critical"
  fi
  echo "step|install the target ($TARGET_BENCH, $TARGET_PUBLIC)|s_target|critical"
  echo "step|install /home/researcher/research|s_research|critical"
  if command -v "$IJ_AGENT" >/dev/null 2>&1; then
    echo "skip|agent $IJ_AGENT|$IJ_AGENT already on PATH: $(command -v "$IJ_AGENT")"
  elif [ -f "$ADAPTER_DIR/setup.sh" ]; then
    echo "step|agent $IJ_AGENT (adapters/$IJ_AGENT/setup.sh install)|s_agent|critical"
  else
    echo "step|agent $IJ_AGENT (no adapters/$IJ_AGENT/setup.sh: install the \`$IJ_AGENT\` command yourself)|s_agent|critical"
  fi
  echo "step|harden: driver user, frozen loop, sudoers, units|s_harden|critical"
  echo "step|/etc/inference-judge/loop.env|s_loop_env|critical"
  # TODO(F): the reference R and the calibration runs are bench/run.sh
  # subcommands; until they are wired here the operator runs them by hand
  # (docs/RUNBOOK.md, "Reference and calibration").
  echo "skip|reference + calibration|run by hand: docs/RUNBOOK.md, Reference and calibration"
  if [ "$IJ_NO_CLOCK" = 1 ]; then
    echo "skip|research clock|--no-clock"
  elif [ "$T_RUNNABLE" != true ] && [ "$stops" = 0 ]; then
    # The clock step locks the ruler through run.sh, which refuses a sketch.
    echo "skip|research clock|target is runnable: false, run.sh will not measure on it"
  else
    echo "step|research clock (loopstate start-clock, as $DRIVER_USER) and RULER_LOCK|s_clock|critical"
  fi
}

# join_and A B C: "A and B and C", for a step's name.
join_and() {
  local out="${1-}"
  [ $# -eq 0 ] || shift
  while [ $# -gt 0 ]; do out="$out and $1"; shift; done
  printf '%s' "$out"
}

print_plan() {
  target_summary
  printf 'agent: %s (research/adapters/%s)\n' "$IJ_AGENT" "$IJ_AGENT"
  local kind name rest
  while IFS='|' read -r kind name rest; do
    case "$kind" in
      step) printf 'RUN   %s%s\n' "$name" "$( [ "${rest#*|}" = optional ] && echo '  (optional)')" ;;
      skip) printf 'SKIP  %s  -- %s\n' "$name" "$rest" ;;
      stop) printf 'STOP  %s  -- the lines below run once it is runnable: true\n' "$name" ;;
    esac
  done < <(plan_steps)
}

# ------------------------------------------------------------------- sequence
run_plan() {
  local kind name fn mode
  while IFS='|' read -r kind name fn mode; do
    case "$kind" in
      skip) skip "$name" "$fn" ;;
      stop)
        record "$name" STOP
        checklist
        "$fn"
        exit 3 ;;
      step)
        step "$name" "$fn" "$mode" <&3
        if [ "$fn" = gpu_channel ]; then
          # step() runs the function in a subshell, so pick the result up from
          # the file gpu_channel wrote.
          # shellcheck disable=SC1090
          . "$GPU_CHANNEL_FILE"
          record "engine channel resolved" PASS "$VLLM_TAG / ${SGLANG_TAG:-no sglang} / $TORCH_BACKEND"
        fi ;;
    esac
  done 3<&0 < <(plan_steps)
}

next_steps() {
  # A target has no incumbent until the operator measures one; without it and
  # its calibration every candidate ends INCONCLUSIVE.
  cat <<NEXT

Next, in this order (docs/RUNBOOK.md has the detail):
  1. log the agent in, once:  $(agent_login_hint)
  2. bash setup/check.sh
  *. the reference run and the calibration, by hand: docs/RUNBOOK.md, Reference and calibration
  3. bash setup/rehearse.sh           # proves the funnel; restarts the clock on PASS
  4. systemctl enable --now inference-judge-runner inference-judge-driver inference-judge-heartbeat.timer
  5. watch:  cat $STATE_DIR/BRIEF.md ; journalctl -fu inference-judge-driver
NEXT
}

main() {
  if [ "$PLAN_ONLY" = 1 ]; then
    print_plan
    exit 0
  fi
  log "inference-judge bootstrap starting on $(hostname) at $(date -u +%FT%TZ)"
  info "$(target_summary)"
  info "agent: $IJ_AGENT (research/adapters/$IJ_AGENT)"
  run_plan
  checklist
  next_steps
}

# ------------------------------------------------------------------ the CLI
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --target) [ $# -ge 2 ] || { echo "--target needs a file" >&2; exit 2; }; IJ_TARGET="$2"; shift ;;
      --target=*) IJ_TARGET="${1#--target=}" ;;
      --agent) [ $# -ge 2 ] || { echo "--agent needs a name" >&2; exit 2; }; IJ_AGENT="$2"; shift ;;
      --agent=*) IJ_AGENT="${1#--agent=}" ;;
      --plan) PLAN_ONLY=1 ;;
      --skip-model) IJ_SKIP_MODEL=1 ;;
      --no-clock) IJ_NO_CLOCK=1 ;;
      --reset-clock) IJ_RESET_CLOCK=1 ;;
      --reinstall-engines) IJ_REINSTALL_ENGINES=1 ;;
      *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
  done
}

# resolve_target: which target this box is for.  It must be named: guessing the
# hardware is how a box gets the wrong install.
resolve_target() {
  if [ -z "$IJ_TARGET" ]; then
    echo "--target is required. One of:" >&2
    find "$IJ_REPO/targets" -maxdepth 1 -name '*.yaml' -printf '  --target targets/%f\n' 2>/dev/null >&2 \
      || ls "$IJ_REPO/targets"/*.yaml >&2
    exit 2
  fi
  case "$IJ_TARGET" in /*) : ;; *) [ -f "$IJ_TARGET" ] || IJ_TARGET="$IJ_REPO/$IJ_TARGET" ;; esac
  IJ_TARGET="$(cd "$(dirname "$IJ_TARGET")" 2>/dev/null && pwd)/$(basename "$IJ_TARGET")"
  # bench/target.py needs PyYAML.  Nothing is installed before the target is
  # read and the pins are checked (no network before a refusal).
  if ! "${IJ_TARGET_PYTHON:-python3}" -c 'import yaml' 2>/dev/null; then
    echo "bench/target.py needs PyYAML: apt-get install -y python3-yaml, then re-run" >&2
    exit 2
  fi
  target_load "$IJ_TARGET" || exit 2
  TARGET_BENCH="$BENCH_DIR/targets/$T_ID.yaml"
}

# resolve_agent: the agent must have an adapter in this repo (research/adapters/).
resolve_agent() {
  local ok=1
  ADAPTER_DIR="$IJ_REPO/research/adapters/$IJ_AGENT"
  case "$IJ_AGENT" in ""|[!A-Za-z0-9]*|*[!A-Za-z0-9_-]*) ok=0 ;; esac
  if [ "$ok" = 0 ] || [ ! -f "$ADAPTER_DIR/session.sh" ]; then
    echo "--agent '$IJ_AGENT': no adapter at research/adapters/$IJ_AGENT/session.sh" >&2
    local d have=""
    for d in "$IJ_REPO"/research/adapters/*/session.sh; do
      [ -f "$d" ] || continue
      d=$(basename "$(dirname "$d")")
      [ "$d" = TEMPLATE ] || have="$have $d"
    done
    echo "adapters here:${have:- none}" >&2
    exit 2
  fi
}

if [ "${IJ_BOOTSTRAP_LIB:-0}" != 1 ]; then
  parse_args "$@"
  resolve_agent
  resolve_target
  # Pins are needed only where the engines are built here, and are checked
  # before anything else runs.
  if [ "$T_ENGINE_CLASS" = built ] && [ "$PLAN_ONLY" != 1 ]; then require_engine_pins; fi
  main
fi

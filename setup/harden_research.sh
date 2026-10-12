#!/usr/bin/env bash
# The real boundary around the researcher.  The guard scripts in
# research/guard are a tripwire: they read a command line and can be
# bypassed by any process the researcher starts.  This script sets the Unix
# permissions that cannot be bypassed.  Linux only; run as root, once, after
# setup/bootstrap.sh and before the loop starts.  Idempotent.
#
#     sudo [AGENT=<name>] setup/harden_research.sh [--dry-run] [--no-venv] [--no-systemd]
#                                   [--keep-direct-bench] [--keep-gpu-venv]
#
# AGENT (or IJ_AGENT; default claude) is the coding agent that does the
# research: a folder in research/adapters/ (claude, codex, pi or your own).
#
# Three users:
#   bench   (BENCH_USER)      owns /opt/bench and the accelerators
#   driver   (DRIVER_USER)     runs the driver, the runner and the heartbeat;
#                             owns the state dir; may run the bench's run.sh
#                             and act as the researcher (sessions, pre-flight,
#                             git commits, reaping detached jobs)
#   researcher (RESEARCHER_USER) runs the agent's sessions; writes the research repo
#
# What it does:
#   1. creates the driver user (in the researcher's group, so it can read the
#      research repo and the queue);
#   2. installs the driver, the runner, the guard and the adapters root-owned
#      in INSTALL_DIR (/opt/inference-judge), and makes research/loop,
#      research/guard, research/adapters and the agent's own folder (.claude
#      for claude, .codex for codex) root-owned and read-only too (the repo
#      copies the researcher sees);
#   3. installs the agent's managed settings, when its adapter has any (the
#      file `adapters/<agent>/setup.sh managed-path` names, or MANAGED_SETTINGS):
#      they load only the root-owned guard; and writes GUARD_ENV
#      (/etc/inference-judge/guard.env), where the guard reads the frozen
#      paths whichever agent calls it;
#   4. creates the state dir owned by the driver, 0755, so the researcher can
#      read BRIEF.md and the ledgers but write nothing there;
#   5. sudoers: the driver may run /opt/bench/run.sh as bench and anything as
#      the researcher; unless --keep-direct-bench, the researcher's own
#      run.sh right is disabled (measurement goes through the queue only);
#   6. accelerator devices (/dev/nvidia*, /dev/accel*, /dev/vfio/*) to group
#      bench, mode 0660, now and on every boot (udev rule + nvidia module
#      options);
#   7. a CPU-only venv for the researcher, and (unless --keep-gpu-venv) the
#      bootstrap venv with torch/vllm made unreadable to the researcher;
#   8. the systemd units from setup/systemd/, and /etc/inference-judge/loop.env if absent
#      (an existing one keeps its settings; only IJ_AGENT and IJ_FROZEN_PATHS
#      are brought up to date)
#      (installed, not started: `systemctl enable --now inference-judge-driver`).
#
# Every path and user is an environment variable (defaults below), so the
# script can be exercised in a throwaway container: research/loop/tests/test_harden.py.
set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT="${AGENT:-${IJ_AGENT:-claude}}"
RESEARCHER_USER="${RESEARCHER_USER:-researcher}"
DRIVER_USER="${DRIVER_USER:-driver}"
BENCH_USER="${BENCH_USER:-bench}"
BENCH_GROUP="${BENCH_GROUP:-$BENCH_USER}"
RESEARCHER_HOME="${RESEARCHER_HOME:-/home/$RESEARCHER_USER}"
DRIVER_HOME="${DRIVER_HOME:-/home/$DRIVER_USER}"
RESEARCH_DIR="${RESEARCH_DIR:-$RESEARCHER_HOME/research}"
STATE_DIR="${STATE_DIR:-$DRIVER_HOME/research-state}"
INSTALL_DIR="${INSTALL_DIR:-/opt/inference-judge}"
BENCH_RUN="${BENCH_RUN:-/opt/bench/run.sh}"
MANAGED_SETTINGS="${MANAGED_SETTINGS:-}"      # default: what the adapter's setup.sh managed-path prints
GUARD_ENV="${GUARD_ENV:-/etc/inference-judge/guard.env}"
SUDOERS_DIR="${SUDOERS_DIR:-/etc/sudoers.d}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
ENV_FILE="${ENV_FILE:-/etc/inference-judge/loop.env}"
UDEV_RULE="${UDEV_RULE:-/etc/udev/rules.d/70-inference-judge-accelerators.rules}"
MODPROBE_CONF="${MODPROBE_CONF:-/etc/modprobe.d/inference-judge-nvidia.conf}"
DEV_DIR="${DEV_DIR:-/dev}"
CPU_VENV="${CPU_VENV:-$RESEARCHER_HOME/.venv-cpu}"
GPU_VENV="${GPU_VENV:-$RESEARCHER_HOME/.venv}"
CPU_PACKAGES="${CPU_PACKAGES:-numpy scipy pandas}"

DRY=0 VENV=1 SYSTEMD=1 QUEUE_ONLY=1 RETIRE_GPU_VENV=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --no-venv) VENV=0 ;;
    --no-systemd) SYSTEMD=0 ;;
    --keep-direct-bench) QUEUE_ONLY=0 ;;
    --keep-gpu-venv) RETIRE_GPU_VENV=0 ;;
    -h|--help) sed -n '/^set -euo pipefail/q;2,$p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '== %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'harden_research: %s\n' "$*" >&2; exit 1; }
# run <cmd...>: do it, or in --dry-run print it.
run()  { if [ "$DRY" = 1 ]; then printf '   + %s\n' "$*"; else "$@"; fi; }
# put <path> <mode> <owner:group>: write stdin to path atomically.
put() {
  local path="$1" mode="$2" owner="$3" tmp
  if [ "$DRY" = 1 ]; then
    printf '   + write %s (%s %s):\n' "$path" "$mode" "$owner"; sed 's/^/       | /'; return 0
  fi
  mkdir -p "$(dirname "$path")"
  tmp="$(mktemp "$(dirname "$path")/.inference-judge.XXXXXX")"
  cat > "$tmp"; chmod "$mode" "$tmp"; chown "$owner" "$tmp"; mv -f "$tmp" "$path"
}

if [ "$DRY" = 0 ]; then
  [ "$(uname -s)" = Linux ] || die "Linux only (use --dry-run elsewhere)"
  [ "$(id -u)" = 0 ] || die "run as root"
fi
# need <condition-text> <test...>: die, or in --dry-run only warn.
need() {
  local msg="$1"; shift
  "$@" && return 0
  if [ "$DRY" = 1 ]; then warn "$msg (dry run)"; else die "$msg"; fi
}
for u in "$RESEARCHER_USER" "$BENCH_USER"; do
  need "user $u does not exist; run setup/bootstrap.sh first" id "$u" >/dev/null 2>&1
done
# The agent: a name, and an adapter in the research repo (adapters/README.md).
ADAPTER_DIR="$RESEARCH_DIR/adapters/$AGENT"
case "$AGENT" in
  ""|[!A-Za-z0-9]*|*[!A-Za-z0-9_-]*) die "AGENT='$AGENT' is not the name of an adapter" ;;
esac
[ -x "$ADAPTER_DIR/session.sh" ] \
  || die "no adapter for agent $AGENT: $ADAPTER_DIR/session.sh is missing or not executable"
# Frozen in the research repo: the loop, the guard, the adapters, and every
# top-level directory the agent's project/ put there (.claude, .codex).  The
# root-owned install is made from the first three.
FROZEN_DIRS=("$RESEARCH_DIR/loop" "$RESEARCH_DIR/guard" "$RESEARCH_DIR/adapters")
shopt -s nullglob
for d in "$ADAPTER_DIR"/project/* "$ADAPTER_DIR"/project/.[!.]* "$ADAPTER_DIR"/project/..?*; do
  if [ -d "$d" ]; then FROZEN_DIRS+=("$RESEARCH_DIR/$(basename "$d")"); fi
done
shopt -u nullglob
FROZEN=""
for d in "${FROZEN_DIRS[@]}"; do
  need "$RESEARCH_DIR has no ${d#"$RESEARCH_DIR"/}/ (setup/bootstrap.sh installs it)" test -d "$d"
  FROZEN="${FROZEN:+$FROZEN:}$d"
done
FROZEN_PATHS="$INSTALL_DIR:$FROZEN"            # what the driver hashes every pass (loop.env)
FROZEN="$FROZEN:$STATE_DIR:$INSTALL_DIR"       # what the guard refuses to touch
RESULTS_DIR="$(dirname "$BENCH_RUN")/results"
need "jq is missing (loop/run.sh classifies sessions with it; bootstrap installs it)" \
  command -v jq >/dev/null
# adapter_setup <verb>: ask the adapter's setup.sh (managed-path, managed-config).
# The file sits in the researcher's repo, so root does not run it: the
# researcher's own user does, and root only writes down what it printed.
adapter_setup() {
  local as=()
  if [ "$(id -u)" = 0 ] && id "$RESEARCHER_USER" >/dev/null 2>&1; then as=(sudo -u "$RESEARCHER_USER"); fi
  ${as[@]+"${as[@]}"} env INSTALL_DIR="$INSTALL_DIR" RESEARCH_DIR="$RESEARCH_DIR" STATE_DIR="$STATE_DIR" \
    RESULTS_DIR="$RESULTS_DIR" FROZEN="$FROZEN" bash "$ADAPTER_DIR/setup.sh" "$1"
}

# ------------------------------------------------------------ 1. driver user
say "1. driver user $DRIVER_USER"
if ! id "$DRIVER_USER" >/dev/null 2>&1; then
  run useradd --system --create-home --home-dir "$DRIVER_HOME" --shell /bin/bash "$DRIVER_USER"
fi
run usermod -aG "$RESEARCHER_USER" "$DRIVER_USER"          # read the repo and the queue
run chmod 0750 "$RESEARCHER_HOME"
run chmod 0711 "$DRIVER_HOME"                             # the researcher may traverse to the state dir

# ----------------------------------------------------- 2. frozen code, root-owned
say "2. frozen code: $INSTALL_DIR, ${FROZEN_DIRS[*]}"
run install -d -o root -g root -m 0755 "$INSTALL_DIR"
# The source of the root-owned install: the research repo's own copies.
for d in loop guard adapters; do
  run rm -rf "$INSTALL_DIR/$d.new"
  run cp -R "$RESEARCH_DIR/$d" "$INSTALL_DIR/$d.new"
  run find "$INSTALL_DIR/$d.new" \( -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +
  run chown -R root:root "$INSTALL_DIR/$d.new"
  run chmod -R u+rwX,go+rX,go-w "$INSTALL_DIR/$d.new"
  run rm -rf "$INSTALL_DIR/$d"
  run mv "$INSTALL_DIR/$d.new" "$INSTALL_DIR/$d"
done
run rm -rf "$INSTALL_DIR/hooks" "$INSTALL_DIR/hooks.new"      # the guard's folder before it was guard/
for d in "${FROZEN_DIRS[@]}"; do
  run chown -R root:root "$d"
  run chmod -R a-w,go+rX "$d"
done

# --------------------------------------------------- 3. managed settings
say "3. the $AGENT agent's managed settings, and $GUARD_ENV"
# Settings the agent reads before any other, root-owned: only the root-owned
# guard runs.  What they say is the adapter's to know (setup.sh managed-config).
managed=""
if [ -f "$ADAPTER_DIR/setup.sh" ]; then
  managed=$(adapter_setup managed-path) || die "$ADAPTER_DIR/setup.sh managed-path failed"
  [ -n "$managed" ] || say "   the $AGENT adapter names no managed settings file: none installed"
else
  say "   the $AGENT adapter has no setup.sh: no managed settings file"
fi
if [ -n "$managed" ]; then
  MANAGED_SETTINGS="${MANAGED_SETTINGS:-$managed}"
  config=$(adapter_setup managed-config) || die "$ADAPTER_DIR/setup.sh managed-config failed"
  [ -n "$config" ] || die "$ADAPTER_DIR/setup.sh managed-config printed nothing"
  printf '%s\n' "$config" | put "$MANAGED_SETTINGS" 0644 root:root
  if [ "$DRY" = 0 ] && command -v python3 >/dev/null 2>&1; then
    case "$MANAGED_SETTINGS" in
      *.json)
        python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$MANAGED_SETTINGS" || die "managed settings are not valid JSON" ;;
      *.toml)
        if python3 -c 'import tomllib' 2>/dev/null; then
          python3 -c 'import sys,tomllib; tomllib.load(open(sys.argv[1], "rb"))' "$MANAGED_SETTINGS" || die "managed settings are not valid TOML"
        fi ;;
    esac
  fi
fi
# The guard reads this whichever agent calls it (research/guard/_lib.sh): plain
# KEY=value lines, and a variable already in the environment wins.
put "$GUARD_ENV" 0644 root:root <<ENV
IJ_PROTECTED_ROOTS=$FROZEN
IJ_STATE_DIR=$STATE_DIR
IJ_RESEARCH_DIR=$RESEARCH_DIR
IJ_RESULTS_DIR=$RESULTS_DIR
ENV

# ---------------------------------------------------------- 4. state dir
say "4. state dir $STATE_DIR (driver-owned, researcher-readable)"
run install -d -o "$DRIVER_USER" -g "$DRIVER_USER" -m 0755 "$STATE_DIR"

# ------------------------------------------------------------- 5. sudoers
say "5. sudoers"
SUDOERS_DRIVER="$SUDOERS_DIR/inference-judge-driver"
put "$SUDOERS_DRIVER" 0440 root:root <<SUDO
# Installed by setup/harden_research.sh.  The driver user runs the ruler (as
# $BENCH_USER, exactly one program) and acts as the researcher (the agent's
# sessions, pre-flight, git commits, reaping the researcher's jobs).
Defaults:$DRIVER_USER !requiretty
$DRIVER_USER ALL=($BENCH_USER) NOPASSWD: $BENCH_RUN
$DRIVER_USER ALL=($RESEARCHER_USER) NOPASSWD: ALL
SUDO
if [ "$DRY" = 0 ] && command -v visudo >/dev/null 2>&1; then
  visudo -cf "$SUDOERS_DRIVER" >/dev/null || { rm -f "$SUDOERS_DRIVER"; die "visudo rejected $SUDOERS_DRIVER"; }
fi
if [ "$QUEUE_ONLY" = 1 ]; then
  for f in "$SUDOERS_DIR"/*; do
    [ -f "$f" ] || continue
    [ "$f" = "$SUDOERS_DRIVER" ] && continue
    if grep -Eq "^[[:space:]]*${RESEARCHER_USER}[[:space:]].*[(]${BENCH_USER}[)]" "$f"; then
      say "   disabling the researcher's direct bench right in $f (queue only)"
      # sudo ignores files in sudoers.d whose names contain a dot
      run mv "$f" "$f.disabled-by-harden"
    fi
  done
fi

# ---------------------------------------------------------- 6. accelerators
say "6. accelerator devices -> group $BENCH_GROUP, 0660"
shopt -s nullglob
devs=("$DEV_DIR"/nvidia* "$DEV_DIR"/accel* "$DEV_DIR"/vfio/*)
shopt -u nullglob
if [ "${#devs[@]}" -eq 0 ]; then
  say "   no accelerator devices under $DEV_DIR (fine in a container or a CPU box)"
fi
for d in ${devs[@]+"${devs[@]}"}; do
  [ -d "$d" ] && continue                       # e.g. /dev/nvidia-caps: its nodes are inside
  run chgrp "$BENCH_GROUP" "$d"
  run chmod 0660 "$d"
done
put "$UDEV_RULE" 0644 root:root <<UDEV
# Installed by setup/harden_research.sh: accelerators belong to group $BENCH_GROUP.
KERNEL=="nvidia*", GROUP="$BENCH_GROUP", MODE="0660"
KERNEL=="accel*", GROUP="$BENCH_GROUP", MODE="0660"
SUBSYSTEM=="vfio", GROUP="$BENCH_GROUP", MODE="0660"
UDEV
# /dev/nvidia* are created by nvidia-modprobe from these module options, not udev.
if getent group "$BENCH_GROUP" >/dev/null 2>&1; then
  gid=$(getent group "$BENCH_GROUP" | cut -d: -f3)
  put "$MODPROBE_CONF" 0644 root:root <<MOD
# Installed by setup/harden_research.sh
options nvidia NVreg_DeviceFileUID=0 NVreg_DeviceFileGID=$gid NVreg_DeviceFileMode=0660
MOD
fi
id -nG "$RESEARCHER_USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$BENCH_GROUP" \
  && warn "$RESEARCHER_USER is in group $BENCH_GROUP: remove it (gpasswd -d $RESEARCHER_USER $BENCH_GROUP)"
for g in docker video render; do
  if id -nG "$RESEARCHER_USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$g"; then
    say "   removing $RESEARCHER_USER from group $g"
    run gpasswd -d "$RESEARCHER_USER" "$g"
  fi
done

# ------------------------------------------------------------ 7. CPU venv
say "7. researcher python: CPU-only venv $CPU_VENV"
if [ "$VENV" = 1 ]; then
  if [ ! -x "$CPU_VENV/bin/python" ]; then
    run sudo -u "$RESEARCHER_USER" -H python3 -m venv "$CPU_VENV"
    # shellcheck disable=SC2086  # a word list on purpose
    run sudo -u "$RESEARCHER_USER" -H "$CPU_VENV/bin/pip" install -q $CPU_PACKAGES
  fi
else
  say "   skipped (--no-venv)"
fi
if [ "$RETIRE_GPU_VENV" = 1 ] && [ -d "$GPU_VENV" ] && [ "$GPU_VENV" != "$CPU_VENV" ]; then
  say "   retiring $GPU_VENV (torch/vllm on the host) from the researcher"
  run chown -R root:root "$GPU_VENV"
  run chmod 0700 "$GPU_VENV"
fi

# -------------------------------------------------------------- 8. systemd
say "8. systemd units and $ENV_FILE"
if [ ! -f "$ENV_FILE" ]; then
  sed -e "s|/home/driver|$DRIVER_HOME|g" -e "s|/home/researcher/research|$RESEARCH_DIR|g" \
      -e "s|/home/researcher|$RESEARCHER_HOME|g" \
      -e "s|/opt/inference-judge|$INSTALL_DIR|g" -e "s|-u researcher|-u $RESEARCHER_USER|" \
      -e "s|^IJ_RESEARCHER_USER=.*|IJ_RESEARCHER_USER=$RESEARCHER_USER|" \
      -e "s|^IJ_RESEARCH_DIR=.*|IJ_RESEARCH_DIR=$RESEARCH_DIR|" \
      -e "s|^IJ_STATE_DIR=.*|IJ_STATE_DIR=$STATE_DIR|" \
      -e "s|^IJ_AGENT=.*|IJ_AGENT=$AGENT|" \
      -e "s|^IJ_FROZEN_PATHS=.*|IJ_FROZEN_PATHS=$FROZEN_PATHS|" \
      "$SETUP_DIR/systemd/loop.env.example" | put "$ENV_FILE" 0640 "root:$DRIVER_USER"
elif [ "$(sed -n 's/^IJ_AGENT=//p' "$ENV_FILE" | tail -1)" != "$AGENT" ] \
     || ! grep -qxF "IJ_FROZEN_PATHS=$FROZEN_PATHS" "$ENV_FILE"; then
  # An existing file keeps the operator's settings.  The two lines that follow
  # the agent are brought up to date, and another agent's command is dropped
  # (IJ_AGENT_BIN; setup/bootstrap.sh writes it again when one is needed).
  say "   $ENV_FILE exists: setting IJ_AGENT=$AGENT and IJ_FROZEN_PATHS in it"
  keep='^IJ_AGENT=|^IJ_FROZEN_PATHS=|^IJ_CLAUDE_BIN='
  [ "$(sed -n 's/^IJ_AGENT=//p' "$ENV_FILE" | tail -1)" = "$AGENT" ] || keep="$keep|^IJ_AGENT_BIN="
  { grep -Ev "$keep" "$ENV_FILE" || true
    printf 'IJ_AGENT=%s\nIJ_FROZEN_PATHS=%s\n' "$AGENT" "$FROZEN_PATHS"; } | put "$ENV_FILE" 0640 "root:$DRIVER_USER"
fi
if [ "$SYSTEMD" = 1 ]; then
  for u in "$SETUP_DIR"/systemd/inference-judge-*.service "$SETUP_DIR"/systemd/inference-judge-*.timer; do
    sed -e "s|^User=driver|User=$DRIVER_USER|" -e "s|^Group=driver|Group=$DRIVER_USER|" \
        -e "s|/home/driver|$DRIVER_HOME|g" -e "s|/opt/inference-judge|$INSTALL_DIR|g" \
        -e "s|/etc/inference-judge/loop.env|$ENV_FILE|" "$u" | put "$SYSTEMD_DIR/$(basename "$u")" 0644 root:root
  done
  if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    run systemctl daemon-reload
    say "   installed; start with: systemctl enable --now inference-judge-driver inference-judge-runner inference-judge-heartbeat.timer"
  else
    say "   installed; systemd is not running here, so nothing was reloaded"
  fi
else
  say "   skipped (--no-systemd)"
fi

# -------------------------------------------------------------- summary
say "done ($AGENT). Check: sudo -u $RESEARCHER_USER touch $RESEARCH_DIR/loop/x   (must fail)"
say "             sudo -u $RESEARCHER_USER cat $STATE_DIR/BRIEF.md      (must work once the driver ran)"
say "             sudo -u $DRIVER_USER sudo -n -u $RESEARCHER_USER true  (must work)"

#!/usr/bin/env bash
# tpu_deadman.sh -- a spend guard that runs ON the TPU VM.  It deletes the VM at a
# deadline whether or not anyone is connected: it is a systemd timer, so it survives
# an SSH disconnect, a laptop that sleeps, a crashed tmux and a reboot.
#
#   sudo bash tools/tpu_deadman.sh arm HOURS [--zone Z] [--project P] [--name N]
#       deletes this VM HOURS from now.  Name, zone and project come from the metadata
#       server unless given.  Checks it can authenticate and SAYS SO LOUDLY (and exits
#       3, with the timer armed all the same) if deletion cannot be confirmed.
#       Arming again replaces the deadline (that is how to shorten it).
#   bash tools/tpu_deadman.sh status      what it will delete, when, how long is left, and
#                                         where the results were last saved (exit 1 when
#                                         nothing is armed)
#   sudo bash tools/tpu_deadman.sh extend HOURS    add HOURS to the deadline
#   sudo bash tools/tpu_deadman.sh disarm          cancel (the hourly save stops too)
#   sudo bash tools/tpu_deadman.sh save            copy the results to the cache disk now
#
# The results outlive the VM.  The boot disk is deleted with the VM, so the run's
# results, research notes and ledger are copied to the cache disk, which is not:
# every hour while armed, and once more before the delete.  The disk is the one
# setup/bootstrap.sh mounted; it writes the mount point to <state>/save-to, and that
# file is all this tool knows about it.  The copy lands in
#     <mount>/inference-judge-saved/<node>-<UTC time of the first arm>/<the path it had on the VM>
# root-only.  It only adds: a file removed on the VM stays in the copy.  To read it
# after the VM is gone, attach the disk to any VM and mount it.  A copy that cannot be
# made (no disk on file, disk not mounted, out of time) is logged as NOT SAVED and
# never holds up the delete.
#
# What it does at the deadline: the save, then gcloud compute tpus tpu-vm delete <name>,
# tried 5 times with a growing pause.  If every try fails it runs `shutdown -h now` and
# logs why; a halted VM may still be billed, so the log line says to delete the node
# from the laptop.  The deadline stays on file and the unit also runs at boot, so a VM
# that is started again tries the delete again.  Log: <state>/deadman.log and the
# journal (journalctl -u inference-judge-deadman).
#
# The one rule: cancelling can never fire the action.  The thing systemd runs
# (`_fire`) does not trust that it was started on time: it re-reads the deadline
# file and refuses to act unless now >= deadline.  disarm removes that file FIRST,
# then the timer; a timer that outlives a failed disarm, a manual
# `systemctl start inference-judge-deadman`, a boot, or a timer that fires early all find no
# deadline, or one in the future, and do nothing.  The file is read again before every
# try of the delete and before the shutdown.  (A shell `sleep && delete` timer whose
# sleep is killed goes straight on to the delete; that is the failure this is built
# against.)
#
# Not for a TPU created through a queued resource: delete the queued resource instead.
#
# Test seams (environment): IJ_DEADMAN_DIR (state), IJ_UNIT_DIR (systemd units),
# IJ_METADATA_URL, IJ_DEADMAN_NOW (epoch seconds), IJ_DEADMAN_SOURCES (what is
# saved), IJ_DEADMAN_SAVE_S (seconds one whole save may take).
set -uo pipefail

STATE_DIR="${IJ_DEADMAN_DIR:-/var/lib/inference-judge-deadman}"
UNIT_DIR="${IJ_UNIT_DIR:-/etc/systemd/system}"
META="${IJ_METADATA_URL:-http://metadata.google.internal/computeMetadata/v1}"
UNIT=inference-judge-deadman
SAVE_UNIT=inference-judge-deadman-save
MAX_HOURS=168
DELETE_TRIES=5
DEADLINE_FILE="$STATE_DIR/deadline"
TARGET_FILE="$STATE_DIR/node"
STATUS_FILE="$STATE_DIR/auth"
LOG="$STATE_DIR/deadman.log"
INSTALLED="$STATE_DIR/tpu_deadman.sh"
# gcloud runs as the same account in `arm` (checking) and in `_fire` (acting): the unit
# runs as root with HOME=/root, so the checks pin the same home and drop any CLOUDSDK_CONFIG.
# It is also the same PROGRAM: arm records where gcloud is, because the unit runs with
# systemd's PATH, which has no /snap/bin and no ~/google-cloud-sdk/bin.
GCLOUD_HOME="${IJ_DEADMAN_HOME:-/root}"
GCLOUD_FILE="$STATE_DIR/gcloud"
GCLOUD=$(head -1 "$GCLOUD_FILE" 2>/dev/null || true)
GC=(env -u CLOUDSDK_CONFIG HOME="$GCLOUD_HOME" "${GCLOUD:-gcloud}")
# What is saved, and where to: the cache disk's mount point, written by bootstrap.
SAVE_TO_FILE="$STATE_DIR/save-to"
SAVE_ID_FILE="$STATE_DIR/save-id"
SAVED_FILE="$STATE_DIR/saved"
SAVE_S="${IJ_DEADMAN_SAVE_S:-600}"
SAVE_SOURCES="${IJ_DEADMAN_SOURCES:-/opt/bench/results /opt/bench/leaderboard /opt/bench/calibration /opt/bench/reference /opt/bench/canary /home/researcher/research /home/driver/research-state /var/log/inference-judge /opt/bench/cache}"

now() { if [ -n "${IJ_DEADMAN_NOW:-}" ]; then echo "$IJ_DEADMAN_NOW"; else date +%s; fi; }
log() { local m; m="$(date -u +%FT%TZ) inference-judge-deadman: $*"; echo "$m"; echo "$m" >> "$LOG" 2>/dev/null || true; }
die() { echo "tpu_deadman: $*" >&2; exit 2; }

utc() { date -u -d "@$1" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -u -r "$1" '+%Y-%m-%d %H:%M:%S'; }

left() {
  local s=$(( $1 - $(now) ))
  if [ "$s" -le 0 ]; then echo "overdue by $(( -s ))s"; else printf '%dh%02dm' $(( s / 3600 )) $(( s % 3600 / 60 )); fi
}

hours_to_seconds() {
  [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "HOURS must be a positive number, got '${1:-}'"
  awk -v h="$1" -v max="$MAX_HOURS" 'BEGIN { if (h <= 0 || h > max) exit 1; printf "%d\n", h * 3600 }' \
    || die "HOURS must be between 0 and $MAX_HOURS, got '$1'"
}

meta() { curl -fsS --max-time 3 -H 'Metadata-Flavor: Google' "$META/$1" 2>/dev/null; }

# tpu-env is the TPU VM's own record of its node: lines like  NODE_ID: 'my-node'.
tpu_env() { meta instance/attributes/tpu-env | sed -n "s/^$1: *'*\([^']*[^' ]\)'*[[:space:]]*\$/\1/p" | head -1; }

load_node() { # NAME ZONE PROJECT from the file arm wrote
  [ -s "$TARGET_FILE" ] || return 1
  NAME=; ZONE=; PROJECT=
  while IFS='=' read -r k v; do
    case "$k" in NAME) NAME=$v ;; ZONE) ZONE=$v ;; PROJECT) PROJECT=$v ;; esac
  done < "$TARGET_FILE"
  [ -n "$NAME" ] && [ -n "$ZONE" ] && [ -n "$PROJECT" ]
}

write_atomic() { printf '%s\n' "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"; }

# boxed SECONDS CMD...: CMD, killed if it is still running after SECONDS (exit 124).
boxed() {
  local s="$1"; shift
  if command -v timeout >/dev/null 2>&1; then timeout -k 10 "$s" "$@"; else "$@"; fi
}

# due: the deadline is on file and has passed.  Asked again before every act.
due() { [ -s "$DEADLINE_FILE" ] && [ "$(now)" -ge "$(cat "$DEADLINE_FILE")" ]; }

# install_timer: the units for the deadline in the deadline file.  An absolute UTC
# time (not "in N hours", which a restart would push out) and Persistent (a VM that was
# off at the deadline fires on boot).  The service is also wanted by every boot: a
# timer whose one moment has passed and has fired does not fire again, so a VM halted
# after a failed delete and started later would otherwise run unguarded.  Before the
# deadline a boot's run refuses, like any other early start.
install_timer() {
  local deadline; deadline=$(cat "$DEADLINE_FILE")
  cat > "$UNIT_DIR/$UNIT.service" <<EOF
[Unit]
Description=Delete this TPU VM at the deadline (a spend guard)
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
Environment=HOME=$GCLOUD_HOME
ExecStart=/bin/bash $INSTALLED _fire

[Install]
WantedBy=multi-user.target
EOF
  cat > "$UNIT_DIR/$UNIT.timer" <<EOF
[Unit]
Description=Deadline for deleting this TPU VM, $(utc "$deadline") UTC

[Timer]
OnCalendar=$(utc "$deadline") UTC
Persistent=true
AccuracySec=1s
Unit=$UNIT.service

[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload || return 1
  systemctl enable "$UNIT.timer" >/dev/null 2>&1 || return 1
  systemctl enable "$UNIT.service" >/dev/null 2>&1 || return 1
  systemctl restart "$UNIT.timer" || return 1
  [ "$(systemctl is-active "$UNIT.timer" 2>/dev/null)" = active ]
}

# install_save_timer: `save` every hour, so a VM that dies some other way (preempted,
# deleted by hand, a kernel panic) has lost at most an hour.
install_save_timer() {
  cat > "$UNIT_DIR/$SAVE_UNIT.service" <<EOF
[Unit]
Description=Copy this TPU VM's results to the cache disk, which outlives it

[Service]
Type=oneshot
ExecStart=/bin/bash $INSTALLED save
EOF
  cat > "$UNIT_DIR/$SAVE_UNIT.timer" <<EOF
[Unit]
Description=Hourly copy of this TPU VM's results to the cache disk

[Timer]
OnCalendar=hourly
AccuracySec=1min
Unit=$SAVE_UNIT.service

[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload || return 1
  systemctl enable "$SAVE_UNIT.timer" >/dev/null 2>&1 || return 1
  systemctl restart "$SAVE_UNIT.timer" || return 1
  [ "$(systemctl is-active "$SAVE_UNIT.timer" 2>/dev/null)" = active ]
}

# save_dir MOUNT: this VM's own directory on the cache disk.  Named once (the node and
# the time it was first armed) and remembered on the boot disk, so a later VM with the
# same node name saves beside this one, never over it.
save_dir() {
  local name
  if [ ! -s "$SAVE_ID_FILE" ]; then
    if load_node; then name=$NAME; else name=$(hostname); fi
    write_atomic "$SAVE_ID_FILE" "$name-$(utc "$(now)" | tr -d ':-' | tr ' ' T)Z" || return 1
  fi
  printf '%s/inference-judge-saved/%s' "$1" "$(head -1 "$SAVE_ID_FILE")"
}

# saving_to: one line on where the results go, for arm and status.
saving_to() {
  local mnt; mnt=$(head -1 "$SAVE_TO_FILE" 2>/dev/null || true)
  if [ -n "$mnt" ]; then echo "under $mnt/inference-judge-saved"
  else echo "NOWHERE YET: no cache disk on file (setup/bootstrap.sh writes $SAVE_TO_FILE when it mounts the disk)"; fi
}

# not_saved WHY: nothing was copied.  Loud in the log, and kept for `status`.
not_saved() {
  log "RESULTS NOT SAVED: $1.  They are on the boot disk only, which is deleted with the VM."
  write_atomic "$SAVED_FILE" "NOT SAVED at $(utc "$(now)") UTC: $1" 2>/dev/null || true
  return 1
}

# cmd_save: copy every source that exists to the cache disk, inside SAVE_S seconds in
# all, then flush.  Returns 0 only if everything that exists was copied; says which
# were and which were not.  It only reads the sources and never deletes from the copy.
cmd_save() {
  local mnt dest src rc t0=$SECONDS left_s ok=0 lost=0 n
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  [ -w "$STATE_DIR" ] || die "cannot write $STATE_DIR: run with sudo"
  mnt=$(head -1 "$SAVE_TO_FILE" 2>/dev/null || true)
  [ -n "$mnt" ] || { not_saved "no cache disk on file ($SAVE_TO_FILE; setup/bootstrap.sh writes it when it mounts the disk)"; return 1; }
  # Unmounted, the mount point is a directory on the boot disk: a copy there dies with the VM.
  mountpoint -q "$mnt" || { not_saved "$mnt is not a mounted disk"; return 1; }
  if ! { dest=$(save_dir "$mnt") && mkdir -p "$dest" && chmod 0700 "$mnt/inference-judge-saved"; }; then
    not_saved "cannot create this VM's directory under $mnt/inference-judge-saved"; return 1
  fi
  for src in $SAVE_SOURCES; do
    if [ ! -d "$src" ]; then log "save: $src is not there (the run has not made it)"; continue; fi
    left_s=$(( SAVE_S - (SECONDS - t0) ))
    rc=124
    if [ "$left_s" -gt 0 ]; then
      rc=0
      mkdir -p "$dest$src" && boxed "$left_s" rsync -a --exclude '*.safetensors' --exclude '__pycache__/' \
        --exclude '.venv/' --exclude '.cache/' "$src/" "$dest$src/" >> "$LOG" 2>&1 || rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
      n=$(find "$dest$src" -type f | wc -l | tr -d ' ')
      log "save: saved $src ($n files in the copy)"; ok=$((ok + 1))
    elif [ "$rc" -eq 124 ]; then
      log "save: NOT SAVED $src: ran out of time (${SAVE_S}s for the whole save)"; lost=$((lost + 1))
    else
      log "save: NOT SAVED $src: rsync exit $rc (its own words are above, in $LOG)"; lost=$((lost + 1))
    fi
  done
  # Without the flush the last files can still be in memory when the VM is deleted.
  boxed 120 sync || log "save: the flush to the disk did not finish (exit $?)"
  write_atomic "$SAVED_FILE" "$(utc "$(now)") UTC: $ok saved, $lost not saved, in $dest"
  log "save: $ok saved, $lost not saved, in $dest"
  [ "$lost" -eq 0 ]
}

# check_auth: can gcloud, here, act on this node?  Prints what it found and records it
# for `status`.  Returns 0 only when the node can be read as an authenticated account;
# the delete permission itself cannot be tested without deleting, so that is said too.
check_auth() {
  local acct scopes problems=""
  acct=$("${GC[@]}" auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
  [ -n "$acct" ] || problems="$problems no active gcloud account;"
  if [ -n "$acct" ] && ! "${GC[@]}" compute tpus tpu-vm describe "$NAME" --zone "$ZONE" --project "$PROJECT" \
        --format='value(name)' >/dev/null 2>&1; then
    problems="$problems $acct cannot read node $NAME in $ZONE/$PROJECT;"
  fi
  scopes=$(meta instance/service-accounts/default/scopes || true)
  if [ -n "$scopes" ] && ! grep -q 'cloud-platform' <<<"$scopes"; then
    problems="$problems the VM's service account scopes lack cloud-platform (deletes are likely refused);"
  fi
  if [ -n "$problems" ]; then
    write_atomic "$STATUS_FILE" "UNCONFIRMED:$problems"
    return 1
  fi
  write_atomic "$STATUS_FILE" "read access confirmed as $acct; delete permission untested"
  return 0
}

cmd_arm() {
  local hours="${1:-}" name="" zone="" project="" secs deadline
  [ $# -gt 0 ] && shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --zone) zone="${2:-}"; shift ;;
      --project) project="${2:-}"; shift ;;
      --name) name="${2:-}"; shift ;;
      *) die "unknown argument: $1" ;;
    esac
    shift
  done
  secs=$(hours_to_seconds "$hours") || exit 2
  [ -w "$UNIT_DIR" ] || die "cannot write $UNIT_DIR: run with sudo"
  mkdir -p "$STATE_DIR" || die "cannot create $STATE_DIR"
  [ -n "$name" ]    || name=$(tpu_env NODE_ID)
  [ -n "$zone" ]    || zone=$(tpu_env ZONE)
  [ -n "$zone" ]    || zone=$(meta instance/zone | sed 's|.*/||')
  [ -n "$project" ] || project=$(meta project/project-id)
  [ -n "$name" ]    || die "cannot tell this node's name from the metadata server: pass --name"
  [ -n "$zone" ]    || die "cannot tell the zone: pass --zone"
  [ -n "$project" ] || die "cannot tell the project: pass --project"
  GCLOUD=$(type -P gcloud) || die "gcloud is not installed on this VM"
  case "$GCLOUD" in /*) ;; *) GCLOUD="$PWD/$GCLOUD" ;; esac
  GC=(env -u CLOUDSDK_CONFIG HOME="$GCLOUD_HOME" "$GCLOUD")
  NAME=$name; ZONE=$zone; PROJECT=$project
  deadline=$(( $(now) + secs ))
  # Order matters.  The deadline and the node first, the timer last: a timer never
  # exists without the file that authorises it.
  if ! { cp "$0" "$INSTALLED.tmp" && chmod 0755 "$INSTALLED.tmp" && mv -f "$INSTALLED.tmp" "$INSTALLED"; }; then
    die "cannot install $INSTALLED"
  fi
  write_atomic "$TARGET_FILE" "$(printf 'NAME=%s\nZONE=%s\nPROJECT=%s' "$NAME" "$ZONE" "$PROJECT")"
  write_atomic "$GCLOUD_FILE" "$GCLOUD"
  write_atomic "$DEADLINE_FILE" "$deadline"
  install_timer || die "the systemd timer did not come up: $UNIT.timer is NOT active, so NOTHING WILL DELETE THIS VM"
  log "armed: delete $NAME ($ZONE/$PROJECT) at $(utc "$deadline") UTC, in $(left "$deadline")"
  save_dir "" >/dev/null || true                       # names this VM's directory by its first arm
  if install_save_timer; then
    echo "results are saved every hour and once more before the delete, $(saving_to)"
  else
    echo "WARNING: the hourly save timer did not come up; the results are still saved once, before the delete" >&2
  fi
  if check_auth; then
    echo "armed. $(cat "$STATUS_FILE")."
    echo "  deleting is the one thing this cannot test; if it fails at the deadline the VM is shut down instead (and may still be billed)."
    return 0
  fi
  {
    echo "################################################################"
    echo "# DELETION IS NOT CONFIRMED.  The timer is armed, but at the deadline"
    echo "# the delete will probably FAIL and the VM will only be shut down,"
    echo "# which may not stop the billing."
    echo "#  $(sed 's/^UNCONFIRMED://' "$STATUS_FILE")"
    echo "# Fix it (gcloud auth login, or recreate the VM with the cloud-platform scope),"
    echo "# then run arm again.  Until then also watch the clock yourself."
    echo "################################################################"
  } >&2
  return 3
}

cmd_status() {
  local deadline
  if [ ! -s "$DEADLINE_FILE" ] || ! load_node; then
    echo "not armed: nothing will delete this VM"
    return 1
  fi
  deadline=$(cat "$DEADLINE_FILE")
  echo "armed: deletes $NAME ($ZONE/$PROJECT) at $(utc "$deadline") UTC"
  echo "time left: $(left "$deadline")"
  echo "timer: $(systemctl is-active "$UNIT.timer" 2>/dev/null || echo unknown)"
  echo "delete: $(cat "$STATUS_FILE" 2>/dev/null || echo 'not checked')"
  if [ ! -s "$SAVE_TO_FILE" ]; then echo "results: NOT SAVED, $(saving_to)"
  else echo "results: $(cat "$SAVED_FILE" 2>/dev/null || echo "no save yet; they go $(saving_to)")"; fi
  if [ "$(systemctl is-active "$UNIT.timer" 2>/dev/null)" != active ]; then
    echo "WARNING: the timer is not active, so nothing will fire: arm again" >&2
    return 4
  fi
}

cmd_disarm() {
  [ -w "$UNIT_DIR" ] || die "cannot write $UNIT_DIR: run with sudo"
  # The deadline file goes first.  Whatever happens to the timer after this, _fire
  # finds no deadline and does nothing.
  rm -f "$DEADLINE_FILE"
  systemctl disable --now "$UNIT.timer" >/dev/null 2>&1 || true
  systemctl disable "$UNIT.service" >/dev/null 2>&1 || true
  systemctl stop "$UNIT.service" >/dev/null 2>&1 || true
  systemctl disable --now "$SAVE_UNIT.timer" >/dev/null 2>&1 || true
  rm -f "$UNIT_DIR/$UNIT.timer" "$UNIT_DIR/$UNIT.service" "$UNIT_DIR/$SAVE_UNIT.timer" "$UNIT_DIR/$SAVE_UNIT.service"
  systemctl daemon-reload >/dev/null 2>&1 || true
  log "disarmed: this VM will not be deleted by inference-judge-deadman, and its results are no longer saved every hour"
  echo "disarmed (the hourly save stopped too; 'save' still works by hand)"
}

cmd_extend() {
  local secs deadline
  secs=$(hours_to_seconds "${1:-}") || exit 2
  [ -w "$UNIT_DIR" ] || die "cannot write $UNIT_DIR: run with sudo"
  if [ ! -s "$DEADLINE_FILE" ] || ! load_node; then die "not armed: use arm"; fi
  deadline=$(cat "$DEADLINE_FILE")
  [ "$deadline" -gt "$(now)" ] || die "the deadline has passed: use arm"
  deadline=$(( deadline + secs ))
  [ $(( deadline - $(now) )) -le $(( MAX_HOURS * 3600 )) ] || die "that is more than $MAX_HOURS hours from now"
  write_atomic "$DEADLINE_FILE" "$deadline"
  install_timer || die "the systemd timer did not come up: $UNIT.timer is NOT active, so NOTHING WILL DELETE THIS VM"
  log "extended: delete $NAME at $(utc "$deadline") UTC, in $(left "$deadline")"
  echo "deadline now $(utc "$deadline") UTC, $(left "$deadline") from now"
}

# _fire: what systemd runs.  Acts only if now >= the deadline FILE (see the header).
cmd_fire() {
  local deadline rc=0 try pause=30
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  if [ ! -s "$DEADLINE_FILE" ]; then
    log "started with no deadline on file (disarmed): doing nothing"
    return 0
  fi
  deadline=$(cat "$DEADLINE_FILE")
  if [ "$(now)" -lt "$deadline" ]; then
    log "started $(( deadline - $(now) ))s before the deadline: refusing, and re-arming the timer"
    install_timer || log "re-arming the timer failed"
    return 0
  fi
  load_node || { log "no node on file: cannot delete; shutting down instead"; shutdown -h now; return 1; }
  log "DEADLINE REACHED: saving the results, then deleting TPU VM $NAME ($ZONE/$PROJECT)"
  # The hourly save must not run beside the last one, nor start after it.
  systemctl stop "$SAVE_UNIT.timer" "$SAVE_UNIT.service" >/dev/null 2>&1 || true
  cmd_save || log "RESULTS NOT FULLY SAVED (see above): deleting the VM all the same"
  for try in $(seq 1 "$DELETE_TRIES"); do
    due || { log "disarmed or extended while deleting: stopping here"; return 0; }
    boxed 600 "${GC[@]}" compute tpus tpu-vm delete "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
      log "delete of $NAME requested (exit 0, try $try)"
      rm -f "$DEADLINE_FILE"
      return 0
    fi
    log "delete of $NAME failed (exit $rc, try $try of $DELETE_TRIES)"
    if [ "$try" -lt "$DELETE_TRIES" ]; then sleep "$pause"; pause=$(( pause * 2 )); fi
  done
  due || { log "disarmed or extended while deleting: stopping here"; return 0; }
  # The deadline stays on file: if this VM is started again, the boot runs this again.
  log "DELETE FAILED (exit $rc) for $NAME, $DELETE_TRIES tries: falling back to 'shutdown -h now'.  The TPU node may STILL BE BILLED: delete it from your laptop: gcloud compute tpus tpu-vm delete $NAME --zone $ZONE --project $PROJECT --quiet"
  shutdown -h now
}

case "${1:-}" in
  arm)     shift; cmd_arm "$@" ;;
  status)  cmd_status ;;
  disarm)  cmd_disarm ;;
  extend)  shift; cmd_extend "$@" ;;
  save)    cmd_save ;;
  _fire)   cmd_fire ;;
  -h|--help|"") sed -n '2,48p' "$0"; [ -n "${1:-}" ] ;;
  *) echo "unknown command: $1 (arm, status, extend, disarm, save)" >&2; exit 2 ;;
esac

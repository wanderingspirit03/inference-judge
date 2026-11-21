#!/bin/bash
# box.sh: drive the rerun's GPU machine from this Mac.  The machine's id and address are kept in ./box.env.
#   box.sh create [region]   one gpu-mi350x1-288gb-spot droplet from gpu-amd-base, in ric1 ($3/h); another
#                            region only with ALLOW_REGION=1 (the limit agreed with the user is ric1)
#   box.sh push              make the bundle anew, copy it and unpack it as /root/rerun
#   box.sh prepare           start prepare_box.sh there (in tmux); "box.sh tail prep/prepare.log" follows it
#   box.sh smoke             start the short trial there (RERUN_SMOKE=1, writes out-smoke/)
#   box.sh run               start rerun.py there (in tmux); starting it again continues where it stopped
#   box.sh pull              copy /root/rerun/out, out-smoke and prep logs here, into ./results/
#   box.sh watch             pull every 2 minutes until the tmux session there ends or the machine is gone
#   box.sh restore           copy ./results/out back to a new machine, so run continues from it
#   box.sh tail [file]       last lines of out/rerun.log (or another file under /root/rerun)
#   box.sh ssh [command]     a shell or one command on the machine
#   box.sh destroy           pull once more, then delete the droplet (asks nothing: run it only when done)
# The first connection records the machine's host key in ./known_hosts; a later change of that key is refused.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="$HOME/.ssh/ij_amd_ed25519"
KEYID=60938946                       # the DigitalOcean key "ij-amd-probe", whose private half is $KEY
SIZE=gpu-mi350x1-288gb-spot
ENVF="$HERE/box.env"
[ -f "$ENVF" ] && . "$ENVF"
# the host key is made here and given to the machine at creation (as the old driver did), so the first
# connection is already checked against it
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$HERE/known_hosts" -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
need() { [ -n "${BOX_IP:-}" ] || { echo "no machine: run box.sh create <region> first"; exit 1; }; }
case "${1:-}" in
  create)
    [ -z "${BOX_ID:-}" ] || { echo "box.env already names droplet $BOX_ID; destroy it first"; exit 1; }
    REGION="${2:-ric1}"
    [ "$REGION" = ric1 ] || [ "${ALLOW_REGION:-0}" = 1 ] || { echo "only ric1 without ALLOW_REGION=1"; exit 1; }
    [ "$(doctl compute droplet list --tag-name ij-amd-rerun --format ID --no-header | wc -l | tr -d ' ')" = 0 ] || { echo "a droplet with the tag ij-amd-rerun exists already"; exit 1; }
    T=$(mktemp -d); ssh-keygen -q -t ed25519 -N "" -C ij-amd-rerun -f "$T/hk"
    python3 - "$T" "$KEY.pub" <<'PY'
import json, sys
t, pub = sys.argv[1], sys.argv[2]
doc = {"ssh_deletekeys": True, "ssh_genkeytypes": [], "disable_root": False,
       "ssh_keys": {"ed25519_private": open(f"{t}/hk").read(), "ed25519_public": open(f"{t}/hk.pub").read().strip()},
       "write_files": [{"path": "/root/.ssh/authorized_keys", "permissions": "0600", "append": True, "content": open(pub).read()}]}
open(f"{t}/user-data", "w").write("#cloud-config\n" + json.dumps(doc, indent=1) + "\n")
PY
    HK=$(cut -d' ' -f1,2 "$T/hk.pub")
    ID=$(doctl compute droplet create "ij-amd-rerun-$(date -u +%m%d%H%M)" --size "$SIZE" --image gpu-amd-base \
          --region "$REGION" --ssh-keys "$KEYID" --tag-name ij-amd-rerun --user-data-file "$T/user-data" \
          --wait --format ID --no-header)
    IP=$(doctl compute droplet get "$ID" --format PublicIPv4 --no-header)
    printf 'BOX_ID=%s\nBOX_IP=%s\nBOX_REGION=%s\nBOX_CREATED="%s"\n' "$ID" "$IP" "$REGION" "$(date -u +%FT%TZ)" > "$ENVF"
    echo "$IP $HK" > "$HERE/known_hosts"; rm -rf "$T"; echo "created droplet $ID at $IP" ;;
  push)
    need; "$HERE/bundle.sh"
    scp -i "$KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$HERE/known_hosts" -o StrictHostKeyChecking=yes \
      "$HERE/rerun-bundle.tar.gz" "root@$BOX_IP:/root/"
    "${SSH[@]}" "root@$BOX_IP" "cd /root && tar -xzf rerun-bundle.tar.gz && echo 'droplet $BOX_ID ${BOX_REGION:-}' > rerun/MACHINE && sha256sum rerun-bundle.tar.gz && ls rerun" ;;
  prepare)
    need; "${SSH[@]}" "root@$BOX_IP" 'command -v tmux >/dev/null || (apt-get update -q >/dev/null 2>&1; apt-get install -y -q tmux >/dev/null 2>&1); tmux new-session -d -s prep "bash /root/rerun/prepare_box.sh > /root/rerun/prep.out 2>&1"; echo started' ;;
  run)
    need; "${SSH[@]}" "root@$BOX_IP" 'tmux has-session -t rerun 2>/dev/null && { echo "already running"; exit 0; }; tmux new-session -d -s rerun "cd /root/rerun && python3 rerun.py >> /root/rerun/out/stdout.txt 2>&1"; echo started' ;;
  smoke)
    need; "${SSH[@]}" "root@$BOX_IP" 'tmux has-session -t rerun 2>/dev/null && { echo "already running"; exit 0; }; mkdir -p /root/rerun/out-smoke; tmux new-session -d -s rerun "cd /root/rerun && RERUN_SMOKE=1 python3 rerun.py >> /root/rerun/out-smoke/stdout.txt 2>&1"; echo started' ;;
  pull)
    need; mkdir -p "$HERE/results"
    rsync -az -e "${SSH[*]}" "root@$BOX_IP:/root/rerun/out/" "$HERE/results/out/"
    rsync -az -e "${SSH[*]}" --exclude 'src/' "root@$BOX_IP:/root/rerun/out-smoke" "root@$BOX_IP:/root/rerun/prep" "$HERE/results/" 2>/dev/null || true
    ls "$HERE/results/out" | wc -l ;;
  watch)
    need
    while true; do
      if ! "$0" pull > /dev/null 2>&1; then
        sleep 20; "$0" pull > /dev/null 2>&1 || { echo "MACHINE UNREACHABLE at $(date -u +%T): $(doctl compute droplet get "$BOX_ID" --format Status --no-header 2>&1 | tail -1)"; exit 2; }
      fi
      "${SSH[@]}" "root@$BOX_IP" 'tmux has-session -t rerun 2>/dev/null || tmux has-session -t prep 2>/dev/null' || { "$0" pull > /dev/null 2>&1; echo "SESSION ENDED at $(date -u +%T)"; "$0" tail 2>/dev/null | tail -4; exit 0; }
      sleep 120
    done ;;
  restore)
    need; rsync -az -e "${SSH[*]}" "$HERE/results/out/" "root@$BOX_IP:/root/rerun/out/"; echo restored ;;
  tail) need; "${SSH[@]}" "root@$BOX_IP" "tail -n ${3:-25} /root/rerun/${2:-out/rerun.log}" ;;
  ssh) need; shift; "${SSH[@]}" "root@$BOX_IP" "$@" ;;
  destroy)
    need; "$0" pull || echo "WARN: the last pull failed"
    doctl compute droplet delete "$BOX_ID" --force
    mv "$ENVF" "$HERE/results/box-$(date -u +%m%d%H%M).env"; echo "deleted droplet $BOX_ID" ;;
  *) sed -n 2,11p "$0" ;;
esac

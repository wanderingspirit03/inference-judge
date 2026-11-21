#!/bin/bash
# prepare_box.sh: turn a fresh gpu-amd-base MI350X droplet into the rerun's machine.  Run as root on the
# machine, from /root/rerun (the unpacked bundle).  No GPU work; safe to run again.
#   1. Docker, tmux, rsync
#   2. in the background: the reference image and the two models (amd/reference/setup.sh)
#   3. meanwhile: AMD's tuned image by the recipe of 2026-11-17 (box-build-tuned.sh), then ij-amd-tuned:paper
#   4. the agents' engine image from data/src.tar (commit 657edee), built without network as the judge did
# Logs: /root/rerun/prep/*.log.  The last line is "PREPARED" when everything is there.
set -uo pipefail
R=/root/rerun
P=$R/prep
mkdir -p "$P" "$R/out"
cd "$R"
echo "== $(date -u +%FT%TZ) prepare_box.sh" | tee -a "$P/prepare.log"
if ! command -v docker > /dev/null; then
  apt-get update -q > "$P/apt.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io python3-venv rsync tmux >> "$P/apt.log" 2>&1
  systemctl enable --now docker >> "$P/apt.log" 2>&1
fi
docker info > /dev/null 2>&1 || { echo "docker does not run; see $P/apt.log"; exit 1; }
( bash "$R/amd/reference/setup.sh" > "$P/reference-setup.log" 2>&1; echo "exit $?" >> "$P/reference-setup.log" ) &
REFPID=$!
LOG="$P/build-tuned.log" bash "$R/amd/measurements/2026-11-17/gpu-session/box-build-tuned.sh"
echo "tuned build exit $?" | tee -a "$P/prepare.log"
docker build -q -t ij-amd-tuned:paper "$R/amd/examples/amd-tuned-engine" >> "$P/prepare.log" 2>&1
wait $REFPID
tail -1 "$P/reference-setup.log" | tee -a "$P/prepare.log"
rm -rf "$P/src" && mkdir -p "$P/src" && tar -xf "$R/data/src.tar" -C "$P/src"
docker build -q --network none -t ij-agent-657edee:paper "$P/src" >> "$P/prepare.log" 2>&1
ok=1
for i in ij-atom-ref:ad256a94-lp1 ij-atom-tuned:dev973 ij-amd-tuned:paper ij-agent-657edee:paper; do
  docker image inspect -f "$i {{.Id}}" "$i" 2>/dev/null | tee -a "$P/prepare.log" || { echo "MISSING image $i" | tee -a "$P/prepare.log"; ok=0; }
done
for m in Qwen3.8-27B Qwen3.8-27B-DFlash2; do
  [ -f "/srv/models/$m/config.json" ] && du -sh "/srv/models/$m" | tee -a "$P/prepare.log" || { echo "MISSING model $m" | tee -a "$P/prepare.log"; ok=0; }
done
df -h / /srv 2>/dev/null | tee -a "$P/prepare.log"
[ $ok = 1 ] && echo PREPARED | tee -a "$P/prepare.log" || { echo "NOT PREPARED" | tee -a "$P/prepare.log"; exit 1; }

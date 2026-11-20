#!/bin/bash
# setup-box.sh [--standin] [--no-docker-launcher] [--code-access]: turn a fresh gpu-amd-base droplet into the GPU
# machine of the two-machine design.  Run as root on the machine by the
# driver, after it copied /opt/ij-amd/amd (this code) and /opt/ij-amd/keys
# (the public keys of `agents` and `judge` on the controller).  Safe to run
# again.
#
#   users    builder (the agents' commands) and judge (judged runs), both in
#            the groups ijgpu, render and video; judge also in docker,
#            builder only with --no-docker-launcher.  Their homes are
#            private to each.  builder's ssh key is the controller's agents
#            key, judge's is the controller's judge key.
#   queue    /var/lib/ij-amd/gpu, shared by both (group ijgpu, setgid);
#            each user has its own state: /var/lib/ij-amd/<user>.
#            cand runs a judged engine (build and serve), see below.
#   sudo     both may run controller/box/ij-reap as root (it checks what it
#            is asked); judge may run anything as cand, and ij-fw check.
#   docker   ON unless --no-docker-launcher (a security setting the user was
#            told about, see NOTES): builder and cand use Docker only through
#            controller/box/ij-docker (sudo), which builds images from the
#            caller's own tree with no network and runs them as the caller's
#            uid with no host mount but the model (read-only) and no added
#            devices or capabilities but the GPU's; its reference profile is
#            fixed.  builder leaves the docker group (a member is root here).
#   firewall controller/box/ij-fw apply: cand reaches loopback only, and not
#            loopback port 53 (owner match on cand's uid, IPv4 and IPv6); a
#            systemd unit applies it again at boot.  ./submit checks it before
#            every judged run and stops with an error when it is not there.
#   model    real: reference/setup.sh (ATOM image, weights download).
#            --standin: a byte-level tokenizer in /srv/models/standin and the
#            stand-in config (fake reference); no Docker, no weights.
#   code     controller/code-access.sh box: /opt/ij-amd/amd belongs to root
#            and the group ijcode (judge); builder and cand read only what
#            builder runs, the reference set-up and the examples AGENTS.md
#            names.  --code-access does only this (and the group), on a
#            machine already set up: ij-driver push-code uses it.
#
# Untested on a real droplet: everything in the real branch except
# reference/setup.sh itself, which ran by hand on 2026-11-16.
set -euo pipefail
STANDIN=0
LAUNCHER=1
ONLY_CODE=0
for a in "$@"; do
  case "$a" in
    --standin) STANDIN=1 ;;
    --no-docker-launcher) LAUNCHER=0 ;;
    --code-access) ONLY_CODE=1 ;;
    *) echo "setup-box.sh: unknown argument $a" >&2; exit 2 ;;
  esac
done
IJ=/opt/ij-amd
A=$IJ/amd
B=/var/lib/ij-amd
# One run at a time: a driver restarted during the set-up runs this again while the first run may go on.
exec 9>/run/ij-setup-box.lock
flock 9

# The driver copies the code owned by root with no access for others; nobody but root reads it until
# code-access.sh below opens what builder and cand run.
chown root:root "$IJ"; chmod 755 "$IJ"
chown -R root:root "$A"
chmod -R go-w,o-rwx "$A"
if [ "$ONLY_CODE" = 1 ]; then
  [ -f /etc/ij-amd/amd-config.json ] && [ "$(readlink -f /etc/ij-amd/amd-config.json)" = "$A/controller/standin/amd-config.standin.json" ] && STANDIN=1
  bash "$A/controller/code-access.sh" box $( [ "$STANDIN" = 1 ] && echo --standin )
  exit 0
fi

# gpu-amd-base (2026-11-17) has ROCm but no Docker; the reference runs in Docker.
if [ "$STANDIN" = 0 ] && ! command -v docker >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io python3-venv rsync
  systemctl enable --now docker
fi

groupadd -f ijgpu
for u in builder judge; do
  id "$u" >/dev/null 2>&1 || useradd -m -s /bin/bash "$u"
  for g in ijgpu docker render video; do
    [ "$g" = docker ] && [ "$u" = builder ] && [ "$LAUNCHER" = 1 ] && continue
    getent group "$g" >/dev/null && usermod -aG "$g" "$u"
  done
  chmod 700 "/home/$u"
  install -d -m 700 -o "$u" -g "$u" "/home/$u/.ssh"
done
key() { cat "$IJ/keys/$1.pub"; }
install -m 600 -o builder -g builder <(key agents) /home/builder/.ssh/authorized_keys
install -m 600 -o judge -g judge <(key judge) /home/judge/.ssh/authorized_keys
install -d -m 755 -o builder -g builder /home/builder/work

# cand: the account a judged engine's build and serve run as (submit.engine_user).
# GPU devices (render, video) but not docker, ijgpu or a login; home 700; no
# cron or at.  Only judge may become it, and only to run the engine.  It
# cannot read /home/judge or /var/lib/ij-amd/judge (both 700): the drawn
# prompts, the source the judge keeps and the reference's cache.
id cand >/dev/null 2>&1 || useradd -m -s /usr/sbin/nologin cand
for g in render video; do
  getent group "$g" >/dev/null && usermod -aG "$g" cand
done
for g in docker ijgpu; do
  getent group "$g" >/dev/null && gpasswd -d cand "$g" >/dev/null 2>&1 || true
done
chmod 700 /home/cand
for f in /etc/cron.deny /etc/at.deny; do
  touch "$f"; grep -qx cand "$f" || echo cand >> "$f"
done

install -d -m 755 -o root -g root "$B"
install -d -m 2775 -o root -g ijgpu "$B/gpu" "$B/gpu/tickets"
[ -e "$B/machine.lock" ] || install -m 664 -o root -g ijgpu /dev/null "$B/machine.lock"
install -d -m 755 -o judge -g judge "$B/judge-flag"
install -d -m 755 -o builder -g builder "$B/builder"
install -d -m 700 -o judge -g judge "$B/judge"

# the code: closed but for what builder runs (judge reads it through the group ijcode)
bash "$A/controller/code-access.sh" box $( [ "$STANDIN" = 1 ] && echo --standin )

# the launcher: builder out of the docker group (a member of it is root on the machine)
if [ "$LAUNCHER" = 1 ]; then
  getent group docker >/dev/null && { gpasswd -d builder docker >/dev/null 2>&1 || true; }
fi
cat > /etc/sudoers.d/ij-box <<EOF
builder,judge ALL=(root) NOPASSWD: $A/controller/box/ij-reap
judge ALL=(cand) NOPASSWD: ALL
judge ALL=(root) NOPASSWD: $A/controller/box/ij-fw check
EOF
if [ "$LAUNCHER" = 1 ]; then
  echo "builder,judge,cand ALL=(root) NOPASSWD: $A/controller/box/ij-docker" >> /etc/sudoers.d/ij-box
fi
chmod 440 /etc/sudoers.d/ij-box
visudo -cf /etc/sudoers.d/ij-box >/dev/null

# cand's firewall rule, now and at every boot
"$A/controller/box/ij-fw" apply
if [ -d /run/systemd/system ]; then
  cat > /etc/systemd/system/ij-fw.service <<EOF
[Unit]
Description=ij-amd: the firewall rule for cand (loopback only)
DefaultDependencies=no
Wants=network-pre.target
Before=network-pre.target
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$A/controller/box/ij-fw apply

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable ij-fw.service >/dev/null 2>&1
fi
"$A/controller/box/ij-fw" check >/dev/null

mkdir -p /etc/ij-amd
if [ "$STANDIN" = 1 ]; then
  python3 "$A/controller/standin/fake_ref.py" tokenizer /srv/models/standin
  ln -sfn "$A/controller/standin/amd-config.standin.json" /etc/ij-amd/amd-config.json
else
  rm -f /etc/ij-amd/amd-config.json
  bash "$A/reference/setup.sh"
fi
echo "setup-box: done ($( [ "$STANDIN" = 1 ] && echo stand-in || echo real ), Docker launcher $( [ "$LAUNCHER" = 1 ] && echo on || echo off ), firewall for cand on)"

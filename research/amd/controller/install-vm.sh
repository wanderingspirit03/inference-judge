#!/bin/bash
# install-vm.sh SRC: install the controller on this machine, as root.  SRC is
# a copy of research/amd (install.sh on the Mac puts it in /tmp).  Safe to
# run again: keys, the config, the hidden prompts and the repository's
# history are kept; code, sudoers, AGENTS.md, the links and the config's
# Prime Agent settings are refreshed.  The code is closed to `agents` but for
# what it runs (code-access.sh).
#
# It copies no credential from anywhere.  It generates three ssh keys here
# (agents, judge, driver), each readable only by its user.  It does not
# start the driver and does not log Prime Agent in.
set -euo pipefail
SRC="${1:?usage: install-vm.sh SRC}"
IJ=/opt/ij-amd
A=$IJ/amd
CTL=/var/lib/ij-ctl
JD=/var/lib/ij-judge
REPO=/home/agents/engine
CONF=/etc/ij-amd/controller.json

say() { echo "[install] $*"; }

# users
for u in agents judge driver; do
  id "$u" >/dev/null 2>&1 || { useradd -m -s /bin/bash "$u"; say "user $u created"; }
done
chmod 750 /home/agents /home/judge /home/driver

# code: closed to all but root and the group ijcode (judge, driver) as it
# arrives; code-access.sh then opens only what agents run (ijc, its module,
# the agent's runner).  The running driver has the group only after a restart.
groupadd -f ijcode
mkdir -p "$A"
chown root:root "$IJ"; chmod 755 "$IJ"
rsync -a --delete --exclude '__pycache__' --exclude '.pytest_cache' --chown=root:ijcode --chmod=o-rwx,g-w \
  "$SRC/" "$A/"
bash "$A/controller/code-access.sh" controller

# directories
install -d -m 755 -o driver -g driver "$CTL" "$CTL/pubkeys"
install -d -m 750 -o driver -g driver "$CTL/records"
install -d -m 700 -o driver -g driver "$CTL/ssh"
install -d -m 700 -o judge -g judge "$JD" "$JD/ssh" "$JD/subs"
install -d -m 755 -o judge -g judge /var/lib/ij-score
install -d -m 700 -o agents -g agents /home/agents/.ssh

# keys, generated here, never copied in
gen() {  # user file
  [ -f "$2" ] || { sudo -u "$1" ssh-keygen -q -t ed25519 -N "" -C "$1@ij-controller" -f "$2"; say "ssh key for $1 generated"; }
}
gen agents /home/agents/.ssh/ij_box
gen judge "$JD/ssh/ij_box"
gen driver "$CTL/ssh/driver"
install -m 644 /home/agents/.ssh/ij_box.pub "$CTL/pubkeys/agents.pub"
install -m 644 "$JD/ssh/ij_box.pub" "$CTL/pubkeys/judge.pub"

# sudo: agents hand a commit to the judge (one fixed program, the commit as a
# tar on stdin); the driver acts as agents (Prime Agent) and asks the judge
# to measure a new machine.
cat > /etc/sudoers.d/ij-amd <<EOF
agents ALL=(judge) NOPASSWD: $A/controller/judge/ij-judge submit *
driver ALL=(agents) NOPASSWD: ALL
driver ALL=(judge) NOPASSWD: $A/controller/judge/ij-judge remeasure
EOF
chmod 440 /etc/sudoers.d/ij-amd
visudo -cf /etc/sudoers.d/ij-amd >/dev/null

# config: written once; later runs only add keys that are missing
mkdir -p /etc/ij-amd
python3 "$A/controller/merge_config.py" "$A/controller/controller.installed.json" "$CONF"
chmod 644 "$CONF"
python3 "$A/controller/agents_text.py" goal > /etc/ij-amd/goal.txt
chmod 644 /etc/ij-amd/goal.txt

# the engine repository
if [ ! -d "$REPO/.git" ]; then
  sudo -u agents git init -q -b main "$REPO"
  say "engine repository created at $REPO"
fi
sudo -u agents -H git config --global user.name "agents"
sudo -u agents -H git config --global user.email "agents@ij-controller"
for c in gpu try profile ref check submit score box; do
  sudo -u agents ln -sfn "$A/controller/bin/ijc" "$REPO/$c"
done
sudo -u agents mkdir -p "$REPO/notes"
[ -e "$REPO/notes/.keep" ] || sudo -u agents touch "$REPO/notes/.keep"
python3 "$A/controller/agents_text.py" agents-md "$(python3 -c "import json; print(json.load(open('$CONF'))['judge']['amd_config'])")" \
  | sudo -u agents tee "$REPO/AGENTS.md" >/dev/null
# Only these paths are committed (`commit -- PATHS`): whatever else the agent has staged stays staged.
OURS="gpu try profile ref check submit score box notes AGENTS.md"
# shellcheck disable=SC2086
sudo -u agents git -C "$REPO" add -A $OURS
# shellcheck disable=SC2086
if ! sudo -u agents git -C "$REPO" diff --cached --quiet -- $OURS; then
  # shellcheck disable=SC2086
  if sudo -u agents git -C "$REPO" commit -q -m "The commands and AGENTS.md (written by the controller's installer)" -- $OURS; then
    say "engine repository: committed the commands and AGENTS.md"
  else
    say "engine repository: the commit failed (a merge in progress?); AGENTS.md is written but not committed"
  fi
fi

# Prime Agent for agents: the official installer, a clean HOME, telemetry off
if [ ! -x /home/agents/.local/bin/prime-agent ]; then
  sudo -u agents -H bash -c 'cd ~ && curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh' \
    > /tmp/ij-prime-install.log 2>&1 || { say "the Prime Agent installer failed: see /tmp/ij-prime-install.log"; }
fi
sudo -u agents mkdir -p /home/agents/.prime/agent
# the config's agent.settings at every install (allowedModels, the default model, traces off); other keys kept.
# Prime Agent rewrites this file while it runs: install with the agent stopped.
python3 "$A/controller/prime_settings.py" "$CONF" /home/agents/.prime/agent/settings.json agents

# the driver's service (installed, not enabled: the user starts it)
cat > /etc/systemd/system/ij-driver.service <<EOF
[Unit]
Description=ij-amd driver (GPU machine and Prime Agent)
After=network-online.target

[Service]
User=driver
ExecStart=$A/controller/driver/ij-driver run
Restart=on-failure
RestartSec=60
RestartPreventExitStatus=3 4

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload 2>/dev/null || true
say "done"

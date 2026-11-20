#!/bin/bash
# allowlist-proof.sh: show that the allowedModels spelling the installer
# writes (controller.installed.json agent.settings) is the one Prime Agent's
# daemon accepts, without the real login.  Run on the Mac.
#
# On the controller VM's Docker, in a container with no network, a fresh HOME
# per case holds only Prime Agent's installed program and its model catalog
# (no auth.json, no sessions).  The interactive program runs in tmux (its
# turns go through the daemon, which applies allowedModels; `-p` does not),
# with a dummy ANTHROPIC_API_KEY so the models are listed.  An allowed model
# gets as far as the network ("Connection error"); one the list does not
# allow is "blocked by the daemon model allowlist".  The installed settings
# also keep the first-run dialog about sharing traces away (case "fresh":
# with no settings it shows).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VM="${VM:-$HOME/.config/boat/amd-vm.sh}"
N=ij-pa-proof
S="$(python3 -c "import json; print(json.dumps(json.load(open('$HERE/../controller.installed.json'))['agent']['settings']))")"

"$VM" "sudo docker image inspect ij-test-controller >/dev/null 2>&1" || { echo "make the image first: ctl-container.sh up" >&2; exit 2; }
"$VM" "sudo docker rm -f $N >/dev/null 2>&1; sudo docker run -d --name $N --network none --label ij.test-ctl=1 ij-test-controller sleep infinity >/dev/null \
  && sudo docker exec $N useradd -m -s /bin/bash pa \
  && sudo tar -C /home/agents -cf - .local/bin/prime-agent .local/share/prime-agent .prime/agent/models/provider-model-catalog.v1.json \
     | sudo docker exec -i $N tar -xf - -C /home/pa && sudo docker exec $N chown -R pa: /home/pa"
"$VM" "sudo docker exec -i $N bash -c 'cat > /tmp/case.sh'" <<'EOF'
# case.sh LABEL SETTINGS_JSON|none MODEL
H=$(mktemp -d /tmp/pa-home.XXXX); mkdir -p $H/.prime/agent/models $H/.local $H/w
cp -r /home/pa/.local/* $H/.local/; cp /home/pa/.prime/agent/models/* $H/.prime/agent/models/
[ "$2" = none ] || printf '%s\n' "$2" > $H/.prime/agent/settings.json
tmux -L pa new-session -d -s t -x 160 -y 45 "cd $H/w && HOME=$H PRIME_AGENT_TELEMETRY=0 ANTHROPIC_API_KEY=not-a-real-key $H/.local/bin/prime-agent --provider anthropic --model $3 --thinking high 'say ok'; sleep 30"
sleep 45
out=$(tmux -L pa capture-pane -p -t t)
tmux -L pa kill-server 2>/dev/null
HOME=$H timeout 20 $H/.local/bin/prime-agent shutdown >/dev/null 2>&1
pkill -u pa -f "$H" 2>/dev/null; sleep 1; rm -rf $H
if grep -q "Share agent traces" <<<"$out"; then r="the trace dialog"
elif grep -q "blocked by the daemon model allowlist" <<<"$out"; then r="blocked by the allowlist"
elif grep -q "Connection error" <<<"$out"; then r="allowed (stopped at the network)"
else r="unclear: $(grep -v '^\s*$' <<<"$out" | tail -3 | tr '\n' ' ')"; fi
echo "$1: $r"
EOF
X() { "$VM" "sudo docker exec -u pa -e HOME=/home/pa $N bash /tmp/case.sh $(printf %q "$1") $(printf %q "$2") $3"; }
ONLY_SONNET="$(python3 -c "import json, sys; s = json.loads(sys.argv[1]); s['allowedModels'] = [m for m in s['allowedModels'] if 'sonnet' in m]; print(json.dumps(s))" "$S")"
X "installed settings, opus (expect allowed)" "$S" claude-opus-5-5
X "installed settings, sonnet (expect allowed)" "$S" claude-sonnet-5-5
X "installed settings, haiku (expect blocked)" "$S" claude-haiku-4-5
X "sonnet only, opus (expect blocked)" "$ONLY_SONNET" claude-opus-5-5
X "fresh, no settings (expect the trace dialog)" none claude-opus-5-5
"$VM" "sudo docker rm -f $N >/dev/null"

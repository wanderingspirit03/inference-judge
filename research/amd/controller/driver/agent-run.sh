#!/bin/bash
# agent-run.sh RUN REPO ARGV_JSON: one run of Prime Agent, as `agents`, in the
# driver's tmux session.  A clean environment: HOME=/home/agents (no personal
# skills), telemetry off.  The exit code goes to ~/.ij-run/RUN.exit for the
# driver's records.
RUN="$1"; REPO="$2"; ARGV_JSON="$3"
export HOME=/home/agents DO_NOT_TRACK=1 PATH="/home/agents/.local/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$HOME/.ij-run"
cd "$REPO" || exit 1
mapfile -d '' ARGV < <(python3 -c 'import json,sys; sys.stdout.write("\0".join(json.loads(sys.argv[1])))' "$ARGV_JSON")
"${ARGV[@]}"
echo "$?" > "$HOME/.ij-run/$RUN.exit"

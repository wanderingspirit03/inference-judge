#!/bin/bash
# rehearse.sh [dir]: run rerun.py end to end against the stand-in (toy model, no GPU), on this computer.
# Makes <dir> (default /tmp/rerun-rehearsal) with the layout rerun.py expects.  KILL_AFTER=<seconds> stops the
# first run after that long and starts it again, to rehearse a spot machine being taken away.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
D="${1:-/tmp/rerun-rehearsal}"
rm -rf "$D"; mkdir -p "$D/data" "$D/out"
ln -s "$REPO/research/amd" "$D/amd"
S="$D/amd/controller/standin"
python3 "$S/fake_ref.py" tokenizer "$D/model"
python3 - "$D" "$S" <<'PY'
import json, random, sys
d, s = sys.argv[1], sys.argv[2]
sys.path.insert(0, s)
import standin_tasks
cfg = json.load(open(f"{s}/amd-config.standin.json"))
cfg["model"]["dir"] = f"{d}/model"
cfg["ref"]["start"] = [f"{s}/fake-ref.sh", "start"]
cfg["ref"]["stop"] = [f"{s}/fake-ref.sh", "stop"]
cfg["ref"]["url"] = "http://127.0.0.1:8041"
json.dump(cfg, open(f"{d}/config.standin.json", "w"), indent=1)
r = random.Random(7)
chars = "etaoinshrdlu ETAOINSHRDLU,.\n0123456789"
for k in range(3):
    with open(f"{d}/data/pool-set-{k}.jsonl", "w") as f:
        for i in range(5):
            text = "".join(r.choice(chars) for _ in range(r.randrange(40, 200))) + ("~" if i == 3 else "")
            f.write(json.dumps({"id": f"set{k}-p{i}", "prompt_token_ids": list(text.encode())}) + "\n")
with open(f"{d}/data/warm.jsonl", "w") as f:
    for i in range(3):
        f.write(json.dumps({"prompt_token_ids": list("".join(r.choice(chars) for _ in range(300)).encode())}) + "\n")
with open(f"{d}/data/tasks_512.jsonl", "w") as f:
    for t in standin_tasks.tasks(40, 3):
        f.write(json.dumps(t) + "\n")
PY
export RERUN_ROOT="$D" RERUN_STANDIN=1 RERUN_MAX_TOKENS=32
if [ -n "${KILL_AFTER:-}" ]; then
  python3 "$HERE/rerun.py" > "$D/first-run.txt" 2>&1 & P=$!
  sleep "$KILL_AFTER"; kill -9 $P 2>/dev/null || true; wait $P 2>/dev/null || true
  echo "--- stopped the first run after $KILL_AFTER s; files so far:"; ls "$D/out" | tr '\n' ' '; echo
  echo "--- left running on purpose (the restart must clean up): $(pgrep -f 'fake_ref.py serve' | wc -l | tr -d ' ') fake reference(s), $(pgrep -f fake_engine.py | wc -l | tr -d ' ') fake engine(s)"
fi
python3 "$HERE/rerun.py"

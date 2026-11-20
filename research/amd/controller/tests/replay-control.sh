#!/bin/bash
# replay-control.sh: on the stand-in GPU machine, as judge (e2e-standin.sh runs
# it).  The control for the replay cheater: the SAME prompts judged twice,
# which the judge's pool never does.  Prints the two statuses; "failed
# passed" shows the replay attack works when prompts come back.
set -e
A=/opt/ij-amd/amd
d=$HOME/runs/replay-control
rm -rf "$d"; mkdir -p "$d/src"
cp "$A/examples/cheaters/replay/serve" "$A/examples/cheaters/cheat.py" "$d/src/"   # as in a commit: cand cannot read /opt
python3 - "$A/controller/standin" "$d" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import toylm
with open(sys.argv[2] + "/h.jsonl", "w") as f:
    for i in range(4):
        f.write(json.dumps({"prompt_token_ids": toylm.encode(f"Replay control prompt {i} about lanterns")}) + "\n")
with open(sys.argv[2] + "/w.jsonl", "w") as f:
    f.write(json.dumps({"prompt_token_ids": toylm.encode("warm-up rows, never judged. " * 4)}) + "\n")
EOF
# measure B's tasks: with the quality bar set (2026-11-18) a run without them fails
python3 "$A/controller/standin/standin_tasks.py" 8 9 > "$d/t.jsonl"
for k in 1 2; do
  "$A/controller/box/ij-box" judge "$d" -- submit --judge-src "$d/src" --hidden "$d/h.jsonl" --warm "$d/w.jsonl" \
    --tasks "$d/t.jsonl" --out "$d/o$k.json" > "$d/log$k" 2>&1 < <(sleep 600) || true    # ij-box stops when its stdin closes
done
for k in 1 2; do [ -f "$d/o$k.json" ] || { echo "run $k left no result; its log:"; tail -15 "$d/log$k"; }; done
python3 -c "import json, sys; print(*(json.load(open(f'{sys.argv[1]}/o{k}.json'))['status'] for k in (1, 2)))" "$d" || true
rm -rf "$d"

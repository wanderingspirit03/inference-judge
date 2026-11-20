#!/bin/bash
# Job 7: plain ATOM on the agentic pools, research/amd's ./submit --reference speed method (one request at a time,
# speed_prompts requests, speed_max_tokens answer tokens), one run per input size. Starts after the full judge.
L=/root/agentic-ref.log
A=/opt/ij-amd/amd
echo "$(date -u +%T) waiting for the judge (after-var.log: baseline exit) and the pools" >> $L
until grep -q 'baseline exit' /root/after-var.log 2>/dev/null && grep -q 'pool build rc' /root/pool-build.log 2>/dev/null; do sleep 30; done
mkdir -p /root/agentic
/root/poolvenv/bin/python - << 'EOF'
import json
rows = [json.loads(l) for l in open("/srv/data/pools/qwen38-agentic-main.jsonl")]
for s in (8000, 16000, 32000):
    pick = [r for r in rows if r["isl"] == s][:8]
    with open(f"/root/agentic/main-{s}-8.jsonl", "w") as f:
        for r in pick:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
c = json.load(open("/opt/ij-amd/amd/config.json"))
c["submit"]["speed_prompts"] = 8
json.dump(c, open("/opt/ij-amd/amd/agentic-config.json", "w"), indent=1)
EOF
export IJ_AMD_CONFIG=$A/agentic-config.json IJ_AMD_STATE=/var/lib/ij-amd/agentic IJ_REF_MAX_LEN=40000
for s in 8000 16000 32000; do
  bash $A/reference/atom-serve.sh stop > /dev/null 2>&1
  t0=$(date +%s)
  $A/submit --reference --hidden /root/agentic/main-$s-8.jsonl --out /root/agentic/ref-$s.json > /root/agentic/ref-$s.log 2>&1
  echo "$(date -u +%T) atom isl $s rc $? $(( $(date +%s) - t0 )) s: $(tail -1 /root/agentic/ref-$s.log)" >> $L
done
bash $A/reference/atom-serve.sh stop > /dev/null 2>&1
echo "$(date -u +%T) done" >> $L

#!/bin/bash
# Jobs 7 and 4 on the GPU, in order, detached. Log: /root/tuned/chain.log.
# 1. plain ATOM (the reference container, max len 40000) on the agentic pools and the 8 public prompts, measured by
#    /root/agentic_measure.py (streamed chat, ignore_eos, 220 tokens, one request at a time);
# 2. AMD's own repro of the tuned engine (megakernel arm only, for time);
# 3. the tuned engine with our ./submit method (--judge-src): answer check on the 43 placeable prompts, speed on the
#    same 8 public prompts as the reference speed runs;
# 4. the tuned engine on the agentic pools with the same client as 1, its server log kept for the acceptance rate;
# then /tmp/ij-job5-go, which starts the job 5 chain on the controller.
L=/root/tuned/chain.log; mkdir -p /root/tuned /root/tuned-work /root/agentic
A=/opt/ij-amd/amd
M=/root/agentic_measure.py
POOLS="/opt/ij-amd/amd/prompts/public.jsonl:7 /root/agentic/main-8000-9.jsonl:8 /root/agentic/main-16000-9.jsonl:8 /root/agentic/main-32000-9.jsonl:8"
log() { echo "$(date -u +%T) $*" >> $L; }
waitup() { for _ in $(seq 1 360); do curl -sf -m 3 "$1/v1/models" > /dev/null && return 0; sleep 5; done; return 1; }

/root/poolvenv/bin/python - << 'EOF'
import json
rows = [json.loads(l) for l in open("/srv/data/pools/qwen38-agentic-main.jsonl")]
for s in (8000, 16000, 32000):
    with open(f"/root/agentic/main-{s}-9.jsonl", "w") as f:
        for r in [r for r in rows if r["isl"] == s][:9]:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
EOF

# 1. plain ATOM
bash $A/reference/atom-serve.sh stop > /dev/null 2>&1
t0=$(date +%s)
IJ_REF_MAX_LEN=40000 bash $A/reference/atom-serve.sh start >> $L 2>&1
waitup http://127.0.0.1:8001 && log "atom ready after $(( $(date +%s) - t0 )) s"
python3 $M http://127.0.0.1:8001 /root/agentic/atom.json $POOLS >> $L 2>&1
log "atom agentic rc $? $(( $(date +%s) - t0 )) s"
bash $A/reference/atom-serve.sh stop > /dev/null 2>&1

# 2. AMD's repro, megakernel arm
t0=$(date +%s)
docker run --rm --name tuned-repro --network host --ipc host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v /srv/models:/models:ro -v /root/tuned-work:/work \
  -e MODEL=/models/Qwen3.8-27B -e DRAFT=/models/Qwen3.8-27B-DFlash2 -e ARMS=megakernel \
  ij-atom-tuned:dev973 bash blog_benchmark/build_and_repro.sh > /root/tuned/repro.log 2>&1
log "repro (megakernel arm) rc $? $(( $(date +%s) - t0 )) s"

# 3. our ./submit method
t0=$(date +%s)
IJ_AMD_CONFIG=$A/var-config.json IJ_AMD_STATE=/var/lib/ij-amd/tuned \
  $A/submit --judge-src /root/tuned-engine --hidden /root/var-prompts-ok.jsonl --out /root/tuned/submit-short.json \
  --commit tuned-dev973 > /root/tuned/submit-short.log 2>&1
log "submit-short rc $? $(( $(date +%s) - t0 )) s: $(tail -3 /root/tuned/submit-short.log | tr '\n' ' ')"

# 4. agentic, same client
t0=$(date +%s)
/root/tuned-engine/serve 8100 > /root/tuned/serve-agentic.log 2>&1 &
SP=$!
waitup http://127.0.0.1:8101 && log "tuned ready after $(( $(date +%s) - t0 )) s"
python3 $M http://127.0.0.1:8101 /root/agentic/tuned.json $POOLS >> $L 2>&1
log "tuned agentic rc $? $(( $(date +%s) - t0 )) s"
kill $SP; sleep 5; docker rm -f ij-engine > /dev/null 2>&1
grep -i -E "accept|draft" /root/tuned/serve-agentic.log | tail -20 > /root/tuned/acceptance.txt

touch /tmp/ij-job5-go; chmod 644 /tmp/ij-job5-go
log "done; /tmp/ij-job5-go written"

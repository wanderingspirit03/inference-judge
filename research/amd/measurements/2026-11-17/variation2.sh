#!/bin/bash
# Variation, part 2: drop prompts the reference cannot place, then 3 checks of the ATOM engine against the reference.
# Ends by writing "done" to /root/variation.log, which lets /root/after-var.sh start the full judge.
export IJ_AMD_CONFIG=/opt/ij-amd/amd/var-config.json IJ_AMD_STATE=/var/lib/ij-amd/root
A=/opt/ij-amd/amd
while pgrep -f "$A/check" >/dev/null; do sleep 5; done
cd /root/var
t0=$(date +%s)
python3 /root/varfilter.py > /root/var/filter.log 2>&1
echo "filter rc $? $(( $(date +%s) - t0 )) s: $(tail -1 /root/var/filter.log)" >> /root/variation.log
python3 - << 'EOF'
import json
p = "/opt/ij-amd/amd/var-config.json"
c = json.load(open(p))
c["check"]["prompts"] = c["submit"]["hidden_prompts"] = "/root/var-prompts-ok.jsonl"
json.dump(c, open(p, "w"), indent=1)
EOF
for i in 1 2 3; do
  t0=$(date +%s)
  $A/check $A/examples/atom-engine > /root/var/check-$i.log 2>&1
  rc=$?
  echo "check $i rc $rc $(( $(date +%s) - t0 )) s: $(grep -E 'PASS|FAIL|passed|failed' /root/var/check-$i.log | tail -1)" >> /root/variation.log
  d=$(ls -td /var/lib/ij-amd/root/check-runs/*/ | head -1); cp -r "$d" /root/var/check-run-$i
done
echo done >> /root/variation.log

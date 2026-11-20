#!/bin/bash
# Variation of the reference against itself, research/amd's method (job 3).
export IJ_AMD_CONFIG=/opt/ij-amd/amd/var-config.json IJ_AMD_STATE=/var/lib/ij-amd/root
A=/opt/ij-amd/amd
mkdir -p /root/var
cd /root/var
for i in 1 2 3; do
  bash $A/reference/atom-serve.sh stop >/dev/null 2>&1
  t0=$(date +%s)
  $A/submit --reference --hidden /root/var-prompts.jsonl --out /root/var/ref-speed-$i.json > /root/var/ref-speed-$i.log 2>&1
  echo "ref-speed $i rc $? $(( $(date +%s) - t0 )) s: $(tail -1 /root/var/ref-speed-$i.log)"
done
for i in 1 2 3; do
  t0=$(date +%s)
  $A/check $A/examples/atom-engine > /root/var/check-$i.log 2>&1
  echo "check $i rc $? $(( $(date +%s) - t0 )) s: $(grep -E 'PASS|FAIL' /root/var/check-$i.log | tail -1)"
  cp $(ls -t /var/lib/ij-amd/root/check*/*.json 2>/dev/null | head -1) /root/var/check-$i.json 2>/dev/null
done
echo done

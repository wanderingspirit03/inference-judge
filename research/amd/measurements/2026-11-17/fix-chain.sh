#!/bin/bash
# After the logprob fix (image ij-atom-ref:ad256a94-lp1, lib/ref.py and the proxy read token_logprobs):
# 1. ./check of the ATOM test engine on all 56 prompts (the 13 dropped ones included), fresh reference cache;
# 2. release job 5 (rerun from ./ref on the controller) and wait for it (at most 40 min);
# 3. the full judge baseline. Log: /root/fix/chain.log
L=/root/fix/chain.log; mkdir -p /root/fix
A=/opt/ij-amd/amd
echo "$(date -u +%T) start" >> $L
t0=$(date +%s)
IJ_AMD_CONFIG=$A/fix-config.json IJ_AMD_STATE=/var/lib/ij-amd/fix $A/check $A/examples/atom-engine > /root/fix/check56.log 2>&1
rc=$?
d=$(ls -td /var/lib/ij-amd/fix/check-runs/*/ 2>/dev/null | head -1); [ -n "$d" ] && cp -r "$d" /root/fix/check56-run
echo "$(date -u +%T) check56 rc $rc $(( $(date +%s) - t0 )) s: $(grep -E 'PASS|FAIL' /root/fix/check56.log | tail -1)" >> $L
docker rm -f ij-ref ij-engine > /dev/null 2>&1
rm -f /tmp/ij-job5b-done; touch /tmp/ij-job5b-go; chmod 644 /tmp/ij-job5b-go
echo "$(date -u +%T) job 5b released" >> $L
for _ in $(seq 1 240); do [ -e /tmp/ij-job5b-done ] && break; sleep 10; done
echo "$(date -u +%T) job 5b $( [ -e /tmp/ij-job5b-done ] && echo done || echo 'not done after 40 min, going on')" >> $L
while docker ps --format '{{.Names}}' | grep -qE '^ij-(ref|engine)$'; do sleep 10; done
t0=$(date +%s)
install -o bench -g bench -m 0640 /dev/null /opt/bench/.allow-baseline
sudo -u bench /opt/bench/run.sh baseline-atom-mi350x > /root/fix/baseline.log 2>&1
rc=$?; rm -f /opt/bench/.allow-baseline
echo "$(date -u +%T) baseline rc $rc $(( $(date +%s) - t0 )) s: $(tail -2 /root/fix/baseline.log | tr '\n' ' ' | cut -c1-300)" >> $L
echo "$(date -u +%T) done" >> $L

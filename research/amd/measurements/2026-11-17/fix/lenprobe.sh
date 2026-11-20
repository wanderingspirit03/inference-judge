#!/bin/bash
# Start the patched reference, run lenprobe.py once, stop it. Log: /root/fix/lenprobe.log
L=/root/fix/lenprobe.log
t0=$(date +%s)
echo "$(date -u +%T) lenprobe start" >> /root/fix/chain.log
/opt/ij-amd/amd/reference/atom-serve.sh start > $L 2>&1
for _ in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 5; done
/root/poolvenv/bin/python -I /root/fix/lenprobe.py http://127.0.0.1:8001 ${N:-} >> $L 2>&1
rc=$?
/opt/ij-amd/amd/reference/atom-serve.sh stop >> $L 2>&1
echo "$(date -u +%T) lenprobe rc $rc $(( $(date +%s) - t0 )) s" >> /root/fix/chain.log

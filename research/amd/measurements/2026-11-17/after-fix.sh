#!/bin/bash
# after the fix chain: the tuned engine text comparison again (fixed script), then stop
L=/root/fix/chain.log
until grep -q "^.* done$" $L; do sleep 20; done
while docker ps --format "{{.Names}}" | grep -qE "^ij-"; do sleep 10; done
t0=$(date +%s)
/root/tuned-engine/serve 8100 > /root/tuned/serve-textcmp2.log 2>&1 &
SP=$!
for _ in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8101/v1/models > /dev/null && break; sleep 5; done
/root/poolvenv/bin/python /root/tuned_textcmp.py http://127.0.0.1:8101 > /root/tuned/textcmp2.log 2>&1
echo "$(date -u +%T) textcmp2 rc $? $(( $(date +%s) - t0 )) s: $(tail -1 /root/tuned/textcmp2.log)" >> $L
kill $SP; sleep 5; docker rm -f ij-engine > /dev/null 2>&1; pkill -f "shim.py 8100" ; true

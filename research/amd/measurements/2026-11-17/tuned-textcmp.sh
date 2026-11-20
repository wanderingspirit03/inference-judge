#!/bin/bash
# after /root/tuned-chain.sh: the tuned engine's greedy answers against the reference's (no logprobs needed), then
# release the job 5 flag (/tmp/ij-job5-go was made a directory to hold the controller back; test -f fails on it).
L=/root/tuned/chain.log
until grep -q "job5-go written" $L; do sleep 15; done
t0=$(date +%s)
/root/tuned-engine/serve 8100 > /root/tuned/serve-textcmp.log 2>&1 &
SP=$!
for _ in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:8101/v1/models > /dev/null && break; sleep 5; done
/root/poolvenv/bin/python /root/tuned_textcmp.py http://127.0.0.1:8101 > /root/tuned/textcmp.log 2>&1
echo "$(date -u +%T) textcmp rc $? $(( $(date +%s) - t0 )) s: $(tail -1 /root/tuned/textcmp.log)" >> $L
kill $SP; sleep 5; docker rm -f ij-engine > /dev/null 2>&1
rmdir /tmp/ij-job5-go && touch /tmp/ij-job5-go && chmod 644 /tmp/ij-job5-go
echo "$(date -u +%T) job 5 flag released" >> $L

#!/bin/bash
# On the box: the GPU queue under contention.  Prints what happened; the
# reader checks the times.  Uses only sleep/echo under ./try (no GPU work),
# except the last part, which runs ./submit on /root/eng-good (box-submit.sh
# makes it).
set -u
IJ=/opt/ij-amd
T() { date +%s.%N | cut -c1-14; }
gpu_line() { "$IJ/gpu" | sed -n '2,4p'; }

echo "== 1. two ./try at once run one after the other"
"$IJ/try" bash -c 'echo "A start $(date +%s.%N | cut -c1-14)"; sleep 5; echo "A end   $(date +%s.%N | cut -c1-14)"' > /tmp/q1a 2>&1 &
sleep 0.5
"$IJ/try" bash -c 'echo "B start $(date +%s.%N | cut -c1-14)"; sleep 1; echo "B end   $(date +%s.%N | cut -c1-14)"' > /tmp/q1b 2>&1 &
sleep 1; echo "-- ./gpu while A holds and B waits:"; "$IJ/gpu"
wait; cat /tmp/q1a /tmp/q1b

echo "== 2. kill -9 of the ./try wrapper releases the GPU and stops its command"
"$IJ/try" bash -c 'sleep 300' > /tmp/q2 2>&1 &
W=$!; sleep 2
echo "holder before: $(gpu_line | head -1)"; kill -9 "$W"; sleep 1
echo "holder after:  $(gpu_line | head -1)"
echo "sleep 300 still running: $(pgrep -fc 'sleep 300' || true)"
t0=$(T); "$IJ/try" true 2>&1 | tail -1; echo "next ./try took $(echo "$(T) - $t0" | bc) s"

echo "== 3. the command killed: ./try returns 128+signal"
"$IJ/try" bash -c 'kill -9 $$' > /tmp/q3 2>&1; echo "rc=$?"

echo "== 4. a waiter killed leaves the queue; the next waiter goes on"
"$IJ/try" sleep 6 > /tmp/q4a 2>&1 &
sleep 0.5; "$IJ/try" true > /tmp/q4b 2>&1 & B=$!
sleep 0.5; "$IJ/try" bash -c 'echo "C ran at $(date +%s.%N | cut -c1-14)"' > /tmp/q4c 2>&1 & C=$!
sleep 1; echo "-- before killing B:"; "$IJ/gpu" | sed -n '2,6p'
kill -9 "$B"; sleep 1; echo "-- after killing B:"; "$IJ/gpu" | sed -n '2,6p'
wait "$C"; echo "C rc=$?"; cat /tmp/q4c | grep -v '^\['; grep -h "queue" /tmp/q4c | tail -1
wait

echo "== 5. nested: ./try inside ./try runs inside the same hold"
t0=$(T); "$IJ/try" "$IJ/try" bash -c 'echo "inner token ${IJ_GPU_JOB:-none}"' 2>&1; echo "rc=$? in $(echo "$(T) - $t0" | bc) s"

echo "== 6. ./submit keeps the GPU alone"
setsid bash -c 'sleep 40 < /dev/kfd' & R=$!
sleep 1; echo "-- a process outside the queue holds /dev/kfd:"; "$IJ/gpu" | sed -n '4,6p'
(cd /root/eng-good && "$IJ/submit" --note "contention test" > /tmp/q6s 2>&1) & S=$!
sleep 20
echo "-- ./try while ./submit runs:"
t0=$(T); "$IJ/try" true 2>&1 | tail -2; echo "./try finished $(echo "$(T) - $t0" | bc) s after it was started"
wait "$S"; echo "submit rc=$?"; grep -v "^  p\|^  h" /tmp/q6s | head -20
wait
echo "== done"

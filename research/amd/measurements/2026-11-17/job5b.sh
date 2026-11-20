#!/bin/bash
# Job 5 again from ./ref onward, after the logprob fix. Waits for /tmp/ij-job5b-go on the box; at the end tells the box
# with /tmp/ij-job5b-done. The hidden set is the 48 quality rows of the 56 variation prompts (none dropped).
# Logs: /root/job5b/*.log; summary /root/job5b/summary.log. The driver unit stays disabled.
D=/root/job5b; mkdir -p $D; S=$D/summary.log
as_agents() { sudo -u agents -H bash -c "$1"; }
step() {
  local name=$1 dir=$2 cmd=$3 t0 rc
  t0=$(date +%s)
  as_agents "cd $dir && $cmd" > $D/$name.log 2>&1; rc=$?
  echo "$(date -u +%T) $name rc $rc $(( $(date +%s) - t0 )) s" >> $S
}
echo "$(date -u +%T) waiting for /tmp/ij-job5b-go on the box" >> $S
until as_agents 'cd ~/engine && ./box test -f /tmp/ij-job5b-go' > /dev/null 2>&1; do sleep 30; done
echo "$(date -u +%T) go" >> $S
# the trees get the fixed test engine (serve finds the box's atom-serve.sh; shim reads token_logprobs)
as_agents 'for w in a b; do cp /opt/ij-amd/amd/examples/atom-engine/* ~/wt-$w/; git -C ~/wt-$w add -A; git -C ~/wt-$w -c user.name=agents -c user.email=agents@localhost commit -q -m "test engine after the logprob fix ($w)"; done' > $D/worktrees.log 2>&1
echo "$(date -u +%T) worktrees rc $?" >> $S
step ref  '~/wt-a' './ref --chat "Write one sentence about rivers." --max-tokens 32'
step ref-cached '~/wt-a' './ref --chat "Write one sentence about rivers." --max-tokens 32'
step check '~/wt-a' './check'
install -o judge -g judge -m 0600 /root/job5b-hidden.jsonl /var/lib/ij-judge/hidden.jsonl
t0=$(date +%s)
as_agents 'cd ~/wt-a && ./submit --note "job 5b test, tree a"' > $D/submit-a.log 2>&1 & pa=$!
sleep 5
as_agents 'cd ~/wt-b && ./submit --note "job 5b test, tree b"' > $D/submit-b.log 2>&1 & pb=$!
wait $pa; ra=$?; ta=$(( $(date +%s) - t0 ))
wait $pb; rb=$?; tb=$(( $(date +%s) - t0 ))
echo "$(date -u +%T) submit-a rc $ra $ta s; submit-b rc $rb $tb s (both started at once)" >> $S
rm -f /var/lib/ij-judge/hidden.jsonl
step score '~/wt-a' './score'
as_agents 'cd ~/engine && ./box touch /tmp/ij-job5b-done' > /dev/null 2>&1
echo "$(date -u +%T) done (hidden set removed; /tmp/ij-job5b-done on the box: rc $?)" >> $S

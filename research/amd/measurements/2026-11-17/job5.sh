#!/bin/bash
# Job 5 (controller side), run as root on the controller VM, detached. Waits until the GPU machine says the GPU is
# free for it (/tmp/ij-job5-go on the box, written after the job 2 judge and the job 7 ATOM runs), then runs the
# agents' commands from /home/agents/engine and one real ./submit of the ATOM test engine from two working trees.
# The driver unit stays disabled. Logs: /root/job5/*.log; one summary line per step in /root/job5/summary.log.
D=/root/job5; mkdir -p $D; S=$D/summary.log
as_agents() { sudo -u agents -H bash -c "$1"; }
step() {  # step NAME DIR COMMAND
  local name=$1 dir=$2 cmd=$3 t0 rc
  t0=$(date +%s)
  as_agents "cd $dir && $cmd" > $D/$name.log 2>&1; rc=$?
  echo "$(date -u +%T) $name rc $rc $(( $(date +%s) - t0 )) s" >> $S
}
echo "$(date -u +%T) waiting for /tmp/ij-job5-go on the box" >> $S
until as_agents 'cd ~/engine && ./box test -f /tmp/ij-job5-go' > /dev/null 2>&1; do sleep 60; done
echo "$(date -u +%T) go" >> $S

# two working trees of the engine repo with the ATOM test engine and the HIP probe, committed
as_agents 'cd ~/engine && for w in a b; do git worktree add -q -B try-$w ~/wt-$w HEAD; cp /opt/ij-amd/amd/examples/atom-engine/* ~/wt-$w/; cp /opt/ij-amd/amd/tests/hip_probe.cpp ~/wt-$w/; git -C ~/wt-$w add -A; git -C ~/wt-$w -c user.name=agents -c user.email=agents@localhost commit -q -m "ATOM test engine and the HIP probe ($w)"; done' > $D/worktrees.log 2>&1
echo "$(date -u +%T) worktrees rc $?" >> $S

step gpu  '~/wt-a' './gpu'
step box  '~/wt-a' './box bash -c "hostname; ls; /opt/rocm/bin/hipcc --version | head -2"'
step try  '~/wt-a' './try bash -c "/opt/rocm/bin/hipcc -O3 --offload-arch=gfx950 -o hip_probe hip_probe.cpp && ./hip_probe"'
step profile '~/wt-a' './profile bash -c "/opt/rocm/bin/hipcc -O3 --offload-arch=gfx950 -o hip_probe hip_probe.cpp && ./hip_probe"'
step ref  '~/wt-a' './ref --chat "Write one sentence about rivers." --max-tokens 32'
step ref-cached '~/wt-a' './ref --chat "Write one sentence about rivers." --max-tokens 32'
step check '~/wt-a' './check'

# the judge's hidden set for this test (removed at the end)
install -o judge -g judge -m 0600 /root/job5-hidden.jsonl /var/lib/ij-judge/hidden.jsonl
t0=$(date +%s)
as_agents 'cd ~/wt-a && ./submit --note "job 5 test, tree a"' > $D/submit-a.log 2>&1 & pa=$!
sleep 5
as_agents 'cd ~/wt-b && ./submit --note "job 5 test, tree b"' > $D/submit-b.log 2>&1 & pb=$!
wait $pa; ra=$?; ta=$(( $(date +%s) - t0 ))
wait $pb; rb=$?; tb=$(( $(date +%s) - t0 ))
echo "$(date -u +%T) submit-a rc $ra $ta s; submit-b rc $rb $tb s (both started at once)" >> $S
rm -f /var/lib/ij-judge/hidden.jsonl
step score '~/wt-a' './score'
echo "$(date -u +%T) done (the hidden test set is removed; box.json still says ready: restore /var/lib/ij-ctl/box.json.before-checks when the box goes)" >> $S

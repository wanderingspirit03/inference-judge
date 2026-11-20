#!/bin/bash
# start.sh [STEP ...]: on the Mac, after controller/install.sh and once the driver has a ready GPU machine.
# Ships the few bench/ files the session needs (bench/ is not installed on the controller or the machine) to
# /var/lib/ij-ctl/gpu-session/bench on the controller (root only), then starts run-session.sh there, detached,
# with the same arguments.  Code only: no key, token or data leaves the Mac.
#   VM=<command that runs a shell command on the VM>   default ~/.config/boat/amd-vm.sh
#   SIZES PER MEAS REFDRAWS TASK_SEEDS TASKS_PER_SEED WARM are passed on (run-session.sh has the defaults)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$(cd "$HERE/../../../../../bench" && pwd)"
VM="${VM:-$HOME/.config/boat/amd-vm.sh}"
COPYFILE_DISABLE=1 tar -C "$BENCH" --no-xattrs -czf - tokenizer.py pools/agentic_pool.py pools/judge_traces.py \
    pools/judge_tasks.py pools/pool_tasks.py \
  | "$VM" "sudo bash -c 'rm -rf /var/lib/ij-ctl/gpu-session/bench && install -d -m 700 /var/lib/ij-ctl/gpu-session/bench && tar -xzf - -C /var/lib/ij-ctl/gpu-session/bench'"
ENVS=""
for v in SIZES PER MEAS REFDRAWS TASK_SEEDS TASKS_PER_SEED WARM; do
  [ -n "${!v:-}" ] && ENVS="$ENVS $v=$(printf %q "${!v}")"
done
"$VM" "sudo env$ENVS nohup bash /opt/ij-amd/amd/measurements/2026-11-17/gpu-session/run-session.sh $* > /dev/null 2>&1 < /dev/null & sleep 2; sudo ls -t /var/lib/ij-ctl/gpu-session | head -2"
echo "started; follow it with: $VM 'sudo tail -f /var/lib/ij-ctl/gpu-session/<stamp>/session.log'"

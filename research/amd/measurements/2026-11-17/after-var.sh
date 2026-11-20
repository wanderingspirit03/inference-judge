#!/bin/bash
# Waits for the variation runs, then runs the full judge baseline (job 2) on the plain ATOM reference.
L=/root/after-var.log
echo "$(date -u +%T) waiting for /root/variation.log done" >> $L
until grep -q '^done' /root/variation.log 2>/dev/null; do sleep 20; done
echo "$(date -u +%T) variation done; stopping the reference" >> $L
bash /opt/ij-amd/amd/reference/atom-serve.sh stop >> $L 2>&1
docker rm -f ij-ref >> $L 2>&1
sleep 10
rocm-smi --showmemuse >> $L 2>&1
[ -d /opt/bench/results/baseline-atom-mi350x ] && mv /opt/bench/results/baseline-atom-mi350x /opt/bench/results/baseline-atom-mi350x.old-$(date +%s)
install -o bench -g bench -m 0640 /dev/null /opt/bench/.allow-baseline
echo "$(date -u +%T) baseline start" >> $L
sudo -u bench /opt/bench/run.sh baseline-atom-mi350x > /var/log/inference-judge/baseline-atom-mi350x.log 2>&1
rc=$?
rm -f /opt/bench/.allow-baseline
echo "$(date -u +%T) baseline exit $rc" >> $L

#!/usr/bin/env bash
# PID 1 of the local box.  Starts the box's own dockerd (Docker-in-Docker), so
# the containers bench/run.sh starts see the box's filesystem, the box's
# loopback and the box's users -- exactly as on a real machine.  Then it waits.
# run_local_box.sh drives every stage with `docker exec`.
set -euo pipefail

# cgroup v2 nesting (the same dance as docker:dind): move every process out of
# the root cgroup so dockerd may enable controllers for its children.
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  mkdir -p /sys/fs/cgroup/init
  xargs -rn1 < /sys/fs/cgroup/cgroup.procs > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || :
  sed -e 's/ / +/g' -e 's/^/+/' < /sys/fs/cgroup/cgroup.controllers \
    > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || :
fi

dockerd --host=unix:///var/run/docker.sock --storage-driver=overlay2 \
  > /var/log/dockerd.log 2>&1 &
for _ in $(seq 1 60); do
  docker info >/dev/null 2>&1 && break
  sleep 0.5
done
docker info >/dev/null 2>&1 || { echo "box_init: dockerd did not come up" >&2; tail -20 /var/log/dockerd.log >&2; }
echo "box_init: ready"
trap 'kill -TERM $(jobs -p) 2>/dev/null; exit 0' TERM INT
while :; do sleep 3600 & wait $!; done

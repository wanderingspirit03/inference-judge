#!/bin/bash
# ctl-container.sh up [OLD_REV] | sh 'COMMAND' | down: a throwaway controller
# in a container on the controller VM's Docker, for testing the installer and
# the e2e (e2e-standin.sh) without touching the installed controller.  Run on
# the Mac.  The stand-in GPU machines the e2e makes are sibling containers on
# the VM's default bridge (the host's Docker socket is mounted in).
#
#   up       builds the image, starts the container, copies in Prime Agent's
#            installed program (launcher and share dir only: no login, no
#            ~/.prime), this tree as /tmp/ij-new, and OLD_REV's research/amd
#            as /tmp/ij-old when given.  Nothing is installed yet.
#   sh       runs a command in the container as root.
#   down     removes the container and the VM's copy.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AMD="$(cd "$HERE/../.." && pwd)"
VM="${VM:-$HOME/.config/boat/amd-vm.sh}"
N=ij-test-ctl
W=/tmp/ij-test-ctl

case "${1:-}" in
  sh) shift; exec "$VM" "sudo docker exec -i $N bash -c $(printf %q "$*")" ;;
  down) exec "$VM" "sudo docker rm -f $N >/dev/null 2>&1; sudo rm -rf $W; echo removed" ;;
  up) ;;
  *) echo "usage: ctl-container.sh up [OLD_REV] | sh 'COMMAND' | down" >&2; exit 2 ;;
esac
OLD="${2:-}"

"$VM" "sudo rm -rf $W && mkdir -m 700 $W $W/new $W/img"
COPYFILE_DISABLE=1 tar -C "$AMD" --no-xattrs --exclude '__pycache__' --exclude '.DS_Store' --exclude '.pytest_cache' -czf - . \
  | "$VM" "tar -xzf - -C $W/new"
REV="$(git -C "$AMD" rev-parse HEAD)"
[ -n "$(git -C "$AMD" status --porcelain -- . 2>/dev/null)" ] && REV="$REV plus changes not committed"
echo "$REV" | "$VM" "cat > $W/new/REVISION"
if [ -n "$OLD" ]; then
  "$VM" "mkdir -m 700 $W/old"
  git -C "$AMD" archive "$OLD" -- . | "$VM" "tar -xf - -C $W/old"
  echo "$OLD" | "$VM" "cat > $W/old/REVISION"
fi

"$VM" "cat > $W/img/Dockerfile" <<'EOF'
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 sudo openssh-client rsync git tmux curl ca-certificates procps util-linux iproute2 \
    && rm -rf /var/lib/apt/lists/*
CMD ["sleep", "infinity"]
EOF
"$VM" "sudo docker build -q -t ij-test-controller $W/img >/dev/null && sudo docker rm -f $N >/dev/null 2>&1; \
  sudo docker run -d --name $N --hostname $N --label ij.test-ctl=1 \
    -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker:ro \
    -v /usr/libexec/docker/cli-plugins:/usr/libexec/docker/cli-plugins:ro ij-test-controller >/dev/null \
  && sudo docker exec $N useradd -m -s /bin/bash agents \
  && sudo tar -C /home/agents -cf - .local/bin/prime-agent .local/share/prime-agent | sudo docker exec -i $N tar -xf - -C /home/agents \
  && sudo docker exec $N chown -R agents: /home/agents/.local \
  && sudo docker cp $W/new $N:/tmp/ij-new && { [ ! -d $W/old ] || sudo docker cp $W/old $N:/tmp/ij-old; } \
  && sudo docker exec $N chmod 700 /tmp/ij-new && { [ ! -d $W/old ] || sudo docker exec $N chmod 700 /tmp/ij-old; } \
  && echo up: $N"

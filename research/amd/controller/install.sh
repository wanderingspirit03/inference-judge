#!/bin/bash
# install.sh: push research/amd to the controller VM and install it there
# (controller/install-vm.sh, as root).  Run on the Mac from anywhere.  Safe
# to run again.  Copies code only: no token, key or credential leaves this
# machine.
#   VM=<command that runs a shell command on the VM>   default ~/.config/boat/amd-vm.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AMD="$(cd "$HERE/.." && pwd)"
VM="${VM:-$HOME/.config/boat/amd-vm.sh}"
STAGE=/tmp/ij-amd-src
# The VM is reached only through $VM, so the code goes as a tar on stdin.
# The stage is private to the VM's login user (the code is closed to the agents) and removed after the install.
COPYFILE_DISABLE=1 tar -C "$AMD" --no-xattrs --exclude '__pycache__' --exclude '.DS_Store' --exclude '.pytest_cache' -czf - . \
  | "$VM" "rm -rf $STAGE && mkdir -m 700 $STAGE && tar -xzf - -C $STAGE"
# which revision this is (installed as /opt/ij-amd/amd/REVISION)
REV="$(git -C "$AMD" rev-parse HEAD 2>/dev/null || echo unknown)"
[ -n "$(git -C "$AMD" status --porcelain -- . 2>/dev/null)" ] && REV="$REV plus changes not committed"
echo "$REV" | "$VM" "cat > $STAGE/REVISION"
"$VM" "sudo bash $STAGE/controller/install-vm.sh $STAGE; rc=\$?; rm -rf $STAGE; exit \$rc"

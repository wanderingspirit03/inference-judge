#!/bin/bash
# box-card-checks.sh: the containment checks of controller/tests/e2e-standin.sh (5c) on the real GPU machine,
# as root (run-session.sh step 7), plus what only the card shows.  Changes nothing it does not put back: the
# firewall hook is removed and applied again; test containers and processes are removed.
# Prints "ok:", "FAILED:" and "seen:" lines (seen: facts for NOTES, not pass or fail); exit 1 on any FAILED.
set -u
A=/opt/ij-amd/amd
C=$A/controller
PASS=0; FAIL=0
ok()  { echo "  ok: $*"; PASS=$((PASS+1)); }
bad() { echo "  FAILED: $*"; FAIL=$((FAIL+1)); }
check() { local what="$1"; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
as() { local u="$1"; shift; sudo -u "$u" -H "$@"; }
conn='import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), 3).close(); print("connected")
except Exception as e:
    print(type(e).__name__)'

echo "== firewall"
check "the rule for cand is in place" bash -c "$C/box/ij-fw check | grep -qx ok"
check "judge may run the check through sudo" bash -c "sudo -u judge sudo -n $C/box/ij-fw check >/dev/null"
check "builder and cand may not" bash -c "! sudo -u builder sudo -n $C/box/ij-fw check 2>/dev/null && ! sudo -u cand sudo -n $C/box/ij-fw check 2>/dev/null"
check "the boot unit is enabled" systemctl is-enabled -q ij-fw.service
c=$(as cand python3 -c "$conn" 1.1.1.1 443); j=$(as judge python3 -c "$conn" 1.1.1.1 443)
check "cand cannot reach 1.1.1.1:443 ($c); judge can ($j)" bash -c '[ "$1" = ConnectionRefusedError ] && [ "$2" = connected ]' _ "$c" "$j"
c6=$(as cand python3 -c "$conn" 2606:4700:4700::1111 443)
check "cand cannot reach IPv6 off the machine ($c6)" bash -c '[ "$1" != connected ]' _ "$c6"
out=$(as cand python3 -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen()
socket.create_connection(s.getsockname(), 3).close(); print("connected")' 2>&1)
check "cand reaches its own loopback port ($out)" test "$out" = connected
# name lookups that do not go through a packet (systemd-resolved's varlink or D-Bus socket): a fact for NOTES
echo "  seen: cand's getent hosts example.com: $(as cand timeout 10 getent hosts example.com || echo "nothing (rc $?)")"
echo "  seen: /etc/resolv.conf nameserver: $(grep -m1 ^nameserver /etc/resolv.conf)"
echo "  seen: nsswitch hosts: $(grep -m1 ^hosts /etc/nsswitch.conf)"

echo "== the launcher"
check "builder is not in the docker group" bash -c "! id -nG builder | grep -qw docker"
check "cand is not in the docker group" bash -c "! id -nG cand | grep -qw docker"
for a in "-v /:/host" "--privileged" "--network host" "--pid host" "--device /dev/mem" "--cap-add SYS_ADMIN" \
         "--security-opt seccomp=unconfined" "--user 0" "--ipc host"; do
  # shellcheck disable=SC2086
  out=$(as cand sudo -n $C/box/ij-docker run $a ij-cand/engine:latest true 2>&1); rc=$?
  check "cand: ij-docker run $a is refused (rc $rc)" bash -c '[ "$1" = 2 ] && grep -q refused <<<"$2"' _ "$rc" "$out"
done
if docker image inspect ij-cand/engine:latest > /dev/null 2>&1; then
  # a container of cand's (left by the last judged build) is under cand's rule: host network, cand's uid.
  # ij-docker runs cand's containers only while judge holds the GPU queue (as in a judged run), so judge
  # takes it here for these three (2026-11-18: without the hold all three were refused).
  rm -f /tmp/ij-card-held
  sudo -u judge -H env IJ_AMD_QUEUE=/var/lib/ij-amd/gpu python3 -c "
import sys, time, os; sys.path.insert(0, '$A/lib'); import gpuqueue; from pathlib import Path
with gpuqueue.Hold(Path('/var/lib/ij-amd'), 'check', ['box-card-checks']):
    open('/tmp/ij-card-held', 'w').close()
    while os.path.exists('/tmp/ij-card-held'):
        time.sleep(1)" < /dev/null > /dev/null 2>&1 &
  holder=$!
  for _ in $(seq 600); do [ -e /tmp/ij-card-held ] && break; kill -0 $holder 2>/dev/null || break; sleep 1; done
  # the hold never came: stop the holder, so that it cannot take the queue later and keep it
  [ -e /tmp/ij-card-held ] || { bad "judge could not take the GPU queue for the container checks"; kill $holder 2>/dev/null; }
  out=$(as cand sudo -n $C/box/ij-docker run --rm ij-cand/engine:latest python3 -c "$conn" 1.1.1.1 443 2>&1 | tail -1)
  check "cand's container cannot reach 1.1.1.1:443 ($out)" test "$out" = ConnectionRefusedError
  out=$(as cand sudo -n $C/box/ij-docker run --rm ij-cand/engine:latest id -u 2>&1 | tail -1)
  check "cand's container runs as cand's uid ($out)" test "$out" = "$(id -u cand)"
  out=$(as cand sudo -n $C/box/ij-docker run --rm ij-cand/engine:latest sh -c 'ls /dev/kfd /dev/dri >/dev/null && echo gpu' 2>&1 | tail -1)
  check "cand's container sees the GPU devices ($out)" test "$out" = gpu
  rm -f /tmp/ij-card-held; wait $holder 2>/dev/null
else
  bad "no ij-cand/engine image (no judged build ran before this step)"
fi

echo "== the GPU seen by judge (batch A, item 4)"
sudo -u cand -H bash -c 'exec 3</dev/kfd; sleep 30' < /dev/null > /dev/null 2>&1 &
sleep 2
cp=$(pgrep -u cand -f 'sleep 30' | head -1)
# without IJ_AMD_REAP_SUDO judge sees only its own processes ([] on 2026-11-18); a judged run (ij-box sets
# it) asks `ij-reap --kfd-users`, which sees cand's
d=$(sudo -u judge python3 -c "
import sys; sys.path.insert(0, '$A/lib'); import gpuqueue
print(gpuqueue.kfd_users())" 2>&1)
echo "  seen: gpuqueue.kfd_users() as judge, own view only, while cand holds /dev/kfd: $d"
d=$(sudo -u judge env IJ_AMD_REAP_SUDO=1 python3 -c "
import sys; sys.path.insert(0, '$A/lib'); import gpuqueue
print(' '.join(str(u['pid']) for u in gpuqueue.kfd_users()))" 2>&1)
check "kfd_users() as judge in a judged run sees cand's process ($cp in: $d)" bash -c '[ -n "$1" ] && grep -qw "$1" <<<"$2"' _ "$cp" "$d"
pkill -u cand -f 'sleep 30'

echo "== ij-reap --machine"
sudo -u builder setsid sleep 7777 < /dev/null > /dev/null 2>&1 &
sleep 1
n0=$(pgrep -u builder -fc 'sleep 7777'); sudo -u judge sudo -n $C/box/ij-reap --machine > /dev/null 2>&1
n1=$(pgrep -u builder -fc 'sleep 7777' || true)
check "ij-reap --machine kills builder's detached process ($n0 -> $n1)" test "$n0" = 1 -a "$n1" = 0

echo "== the firewall check fails closed"
CU=$(id -u cand)
iptables -D OUTPUT -m owner --uid-owner "$CU" -j IJ-CAND
check "with the hook removed the check fails" bash -c "! $C/box/ij-fw check >/dev/null"
"$C/box/ij-fw" apply > /dev/null
check "ij-fw apply puts it back" bash -c "$C/box/ij-fw check | grep -qx ok"

echo "== $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]

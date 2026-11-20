#!/bin/bash
# code-access.sh controller|box [--standin]: who may read this code tree
# (/opt/ij-amd/amd), as root.  install-vm.sh runs it on the controller,
# setup-box.sh on the GPU machine; safe to run again.
#
# The tree belongs to root and the group ijcode, and the rest of the
# machine's accounts read only the files they run themselves (the user's
# decision of 2026-11-18: the judge's code, its tests, the cheater examples
# and the notes are closed to the agents).  Directories on the way to such a
# file can be passed through but not listed, except a directory Python
# imports from, which must be listable.
#
#   controller  ijcode: judge, driver.  Readable by all: the agents' commands
#               (controller/bin/ijc and the one module it imports,
#               controller/lib/ijctl.py) and the runner the driver starts
#               Prime Agent with, as agents (controller/driver/agent-run.sh).
#   box         ijcode: judge.  Readable by all: what builder runs itself
#               (./try, ./check, ./profile, ./ref, ./gpu through
#               controller/box/ij-box, the modules of lib/ they import, the
#               config and the public prompts), the reference set-up
#               (reference/) and the examples AGENTS.md names
#               (examples/amd-tuned-engine and examples/atom-engine, whose
#               shim it uses).  The root tools they reach through sudo
#               (ij-docker, ij-reap, ij-fw) can be run, not read.  ./check compares answers with the same
#               modules the judge uses (lib/answers.py, lib/quality.py), so
#               those stay readable.  --standin also opens controller/standin
#               (the toy model and the stand-in reference that builder and
#               cand run there).
#   Closed on both: submit, score, lib/ledger.py, controller/ (judge, driver,
#   box tools run through sudo, installers, tests), examples/cheaters and the
#   other examples, tests/, measurements/ (NOTES.md included), README.md.
set -euo pipefail
ROLE="${1:?usage: code-access.sh controller|box [--standin]}"
STANDIN=0
[ "${2:-}" = --standin ] && STANDIN=1
A="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
G=ijcode

groupadd -f "$G"
case "$ROLE" in
  controller) MEMBERS="judge driver" ;;
  box) MEMBERS="judge" ;;
  *) echo "code-access.sh: unknown role $ROLE" >&2; exit 2 ;;
esac
for u in $MEMBERS; do
  id "$u" >/dev/null 2>&1 && usermod -aG "$G" "$u"
done

chown -R "root:$G" "$A"
chmod -R u+rwX,g+rX,g-w,o-rwx "$A"
# setgid directories: a file root makes later (Python's __pycache__ when root
# imports from the tree) belongs to ijcode too, so the driver can still copy
# the whole tree to the GPU machine
find "$A" -type d -exec chmod g+s {} +

# pass: the directories from the tree's top down to $1 may be passed (o+x), not listed
pass() {
  local d
  d="$(dirname "$A/$1")"
  while [ "$d" != "$A" ] && [ "$d" != / ]; do chmod o+x "$d"; d="$(dirname "$d")"; done
  chmod o+x "$A"
}
# open PATH: readable by all (a directory with everything in it)
open() {
  [ -e "$A/$1" ] || { echo "code-access.sh: $1 is not in $A" >&2; exit 1; }
  chmod -R o+rX "$A/$1"
  pass "$1"
}
# runnable FILE: may be run through sudo, and tested with [ -x ], by all; not read (root runs it)
runnable() {
  [ -e "$A/$1" ] || { echo "code-access.sh: $1 is not in $A" >&2; exit 1; }
  chmod o+x "$A/$1"
  pass "$1"
}
# listable DIR: its names can be listed (Python imports from it); its files stay as they are
listable() {
  chmod o+rx "$A/$1"
  pass "$1"
}

if [ "$ROLE" = controller ]; then
  open controller/bin/ijc
  listable controller/lib
  open controller/lib/ijctl.py
  open controller/driver/agent-run.sh
else
  for f in try check profile ref gpu config.json prompts reference controller/box/ij-box \
           examples/amd-tuned-engine examples/atom-engine; do
    open "$f"
  done
  # the root-run tools builder and cand reach through sudo (reference/atom-serve.sh tests the launcher
  # with [ -x ] before it chooses it over a direct docker)
  for f in controller/box/ij-docker controller/box/ij-reap controller/box/ij-fw; do
    runnable "$f"
  done
  listable lib
  for f in "$A"/lib/*.py; do
    [ "$(basename "$f")" = ledger.py ] || open "lib/$(basename "$f")"
  done
  [ "$STANDIN" = 1 ] && open controller/standin
fi
echo "code-access: $A closed to all but root and $G ($MEMBERS); opened what $ROLE's other accounts run"

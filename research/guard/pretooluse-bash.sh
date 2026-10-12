#!/usr/bin/env bash
# Runs before a shell command (the Bash tool of Claude Code and Codex, bash in Pi).
#
# Rule: the accelerator and the ruler are reached through the queue
# (queue/<id>.json, which the driver's runner turns into a run.sh call) and
# nothing else.  Any command that mentions nvidia-smi or docker is blocked,
# because measuring or launching outside run.sh is how a result stops being
# believable.  The exact line
#     sudo -u bench /opt/bench/run.sh <id>
# is still let through here; sudo itself refuses it for the researcher, because
# only the driver has that rule.
#
# ONE EXCEPTION: reading the ruler's own output.
# /opt/bench/results/<id>/ is world-readable by design and the researcher is
# supposed to analyse it -- jq over
# results.json, tail on server.log, a python script over the per-turn JSONL.
# Blocking that only taught the agent to work around the hook.  So:
#
#   allowed : a read-shaped command whose every /opt/bench mention is under
#             /opt/bench/results/  (cat, jq, ls, head, tail, grep, wc, awk,
#             sort, find, python3 ... -- the list below)
#   denied  : any other /opt/bench path, and any write-shaped command
#             (>, >>, tee, rm, mv, cp, chmod, chown, sed -i, install, truncate,
#             dd, ln) that mentions /opt/bench at all
#
# Every command also passes _guard.py,
# which denies:
#   - the accelerators under any spelling: nvidia-smi or tpu-info by path or
#     split by quotes, /dev/nvidia*, /dev/accel*, /dev/vfio, and host python
#     importing torch / jax / libtpu / vllm / sglang (-c, -m, heredoc, script);
#   - write-shaped commands (redirects, tee, cp/mv/rm/ln/chmod, sed -i, editors,
#     git checkout/restore, python open(...,'w'), bash -c '...') touching the
#     frozen files (loop/, guard/, adapters/, the agent's settings folder) or
#     the driver's state dir.
# Reading all of those stays allowed, and so do date, sleep, timeout and xargs
# around readers, and run.sh output piped to head/tail/grep.
#
# THIS HOOK IS A TRIPWIRE, NOT THE ENFORCEMENT.  The real boundary is Unix
# permissions: /opt/bench is bench:bench 0751, its children 0750;
# setup/harden_research.sh makes the frozen files root-owned and read-only,
# gives the state dir to the driver user, the accelerator devices to group
# bench, and the researcher a CPU-only venv.  A hook can be bypassed by any
# process the agent starts; the permissions cannot.
set -uo pipefail
# shellcheck source=research/guard/_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

cmd=$(jqr '.tool_input.command')
[ -z "$cmd" ] && exit 0

# Trim leading/trailing whitespace for the exact-match test.
trimmed="${cmd#"${cmd%%[![:space:]]*}"}"
trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"

# The one allowed form, with the same id pattern run.sh validates.
if [[ "$trimmed" =~ ^sudo\ -u\ bench\ /opt/bench/run\.sh\ [a-z0-9][a-z0-9-]{2,40}$ ]]; then
  exit 0
fi

# The same line under timeout, or its output piped to head/tail/grep with
# plain options (no paths, no operators: the pipe stays a reader).
pipe_re='( \| (head|tail|grep)( [-A-Za-z0-9 ._:=\x27"]*)?)*'
if [[ "$trimmed" =~ ^(timeout(\ -k\ [0-9]+[smh]?)?\ [0-9]+[smh]?\ )?sudo\ -u\ bench\ /opt/bench/run\.sh\ [a-z0-9][a-z0-9-]{2,40}(\ 2\>\&1)?${pipe_re}$ ]]; then
  exit 0
fi

word_re='(^|[^A-Za-z0-9_./-])'

if [[ "$cmd" =~ ${word_re}nvidia-smi($|[^A-Za-z0-9_-]) ]]; then
  deny "'nvidia-smi' is not available to the researcher.
The accelerator is driven only by /opt/bench/run.sh, which the driver's runner
calls. Ask for what you need from the
results: run.sh saves server logs and any profile the experiment writes to /out."
fi

if [[ "$cmd" =~ ${word_re}docker($|[^A-Za-z0-9_-]) ]]; then
  deny "'docker' is not available to the researcher.
Experiments are containerised by run.sh from experiments/<id>/Dockerfile (or
inference-judge-base). Change the Dockerfile or launch.sh, then queue the
experiment: write queue/<id>.json and the driver's runner calls the ruler."
fi

# ------------------------------------------- accelerators and frozen files
hook_cwd=$(jqr '.cwd'); hook_cwd="${hook_cwd:-${CLAUDE_PROJECT_DIR:-$PWD}}"
if command -v python3 >/dev/null 2>&1; then
  guard=$(IJ_HOOK_CMD="$cmd" IJ_HOOK_CWD="$hook_cwd" IJ_PROTECTED_ROOTS="$PROTECTED" \
          python3 -I "$(dirname "${BASH_SOURCE[0]}")/_guard.py" 2>/dev/null)
else
  # No python3: a coarse, closed-leaning fallback on the raw string.
  guard=OK
  case "$cmd" in *"/dev/nvidia"*|*"/dev/accel"*|*"/dev/vfio"*|*tpu-info*|*nvidia-smi*) guard=$'ACCEL\tan accelerator' ;; esac
  case "$cmd" in *".claude"*|*".codex"*|*"loop/"*|*"guard/"*|*"adapters/"*|*"research-state"*)
    case "$cmd" in *">"*|*tee*|*" rm "*|*" mv "*|*" cp "*|*"sed -i"*|*chmod*|*" ln "*) guard=$'FROZEN\ta frozen path' ;; esac ;;
  esac
fi
gkind=${guard%%$'\t'*}
gdetail=${guard#*$'\t'}
case "$gkind" in
  OK) ;;
  ACCEL)
    deny "the accelerators are not available to the researcher ($gdetail).
The GPU/TPU is driven only by /opt/bench/run.sh, which the driver's runner calls
for you: write queue/<id>.json after 'loop/preflight.sh <id>' passes. Host
python here is CPU-only; code that imports torch/jax/vllm/sglang belongs in
experiments/<id>/ and runs inside the bench container." ;;
  FROZEN)
    deny "write-shaped command on a frozen path ($gdetail).
$FROZEN_WORDS" ;;
  *)
    deny "this hook could not inspect the command (guard said '${guard:-nothing}'),
so it refused it. Simplify the command." ;;
esac

# Nothing below concerns commands that never mention the ruler.
case "$cmd" in *"/opt/bench"*) ;; *) exit 0 ;; esac

# ------------------------------------------------------ /opt/bench is mentioned
#
# The three checks below (write-shaped, path, first word) used to be run over
# the raw string with `tr '|;&\n'` and a bare `*">"*` test, which knows nothing
# about quoting.  That denied ordinary analysis --
#     jq ".long | keys" /opt/bench/results/exp-one/results.json
# splits into `jq ".long` and `keys"`, and any python one-liner containing a
# `>` comparison inside quotes looked like a redirect.  An agent that cannot
# read its own results works around the hook, which is the worst outcome.
#
# The inspection is therefore done by a small python3 pass that walks the string
# once, tracking single quotes, double quotes and backslashes, and only treats
# | ; & newline < > ` $( ) as operators when they are UNQUOTED.  Every deny the
# old version had is still here: unquoted redirection, the file-mutating
# coreutils anywhere in the command, sed -i, any /opt/bench path outside
# results/, .. traversal, and any first word outside the reader allowlist
# (which is what keeps out sudo, bash -c and scripts of your own).
#
# python3 is on the box and on the Mac the hook tests run on.  If it is missing
# the hook fails CLOSED: the command is denied rather than waved through.
if ! command -v python3 >/dev/null 2>&1; then
  deny "python3 is missing, so this hook cannot inspect a command that mentions
/opt/bench, and it refuses rather than guess. Read the file with the Read tool."
fi

# The inspector is read into a variable rather than piped straight into
# `python3 - <<PY` inside $( ): bash 3.2 (a Mac, where these tests also run)
# mis-parses a heredoc nested in a command substitution.
INSPECT_PY=""
read -r -d '' INSPECT_PY <<'PY' || true
import os, re, shlex

cmd = os.environ.get("IJ_HOOK_CMD", "")

WRITE_CMDS = {"tee", "rm", "rmdir", "mv", "cp", "chmod", "chown", "chgrp", "ln",
              "dd", "truncate", "install", "touch", "mkdir", "shred", "unlink"}
READERS = {"cat", "jq", "ls", "head", "tail", "grep", "egrep", "fgrep", "zgrep",
           "wc", "awk", "sed", "sort", "uniq", "cut", "tr", "find", "stat",
           "file", "diff", "column", "less", "more", "od", "xxd", "md5sum",
           "sha256sum", "du", "test", "[", "echo", "printf", "python",
           "python3", "for", "done", "fi", "while", "until", "read", ":",
           "date", "sleep", "true", "basename", "dirname", "realpath",
           "readlink", "nl"}
# Wrappers that run another command: the wrapped command must be a reader.
WRAP_OPTS = {"timeout": {"-s", "-k", "--signal", "--kill-after"},
             "xargs": {"-n", "-I", "-P", "-L", "-d", "-s", "-E", "-a"}}
KEYWORDS = {"do", "then", "else", "elif", "if", "!", "{", "}", "time"}


def scan(s):
    """Split on UNQUOTED operators. Returns (segments, saw_redirection)."""
    segs, cur, redir = [], [], False
    q = None          # None, "'" or '"'
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if q == "'":
            cur.append(c)
            if c == "'":
                q = None
            i += 1
            continue
        if q == '"':
            if c == "\\" and i + 1 < n:
                cur.append(s[i:i + 2]); i += 2; continue
            cur.append(c)
            if c == '"':
                q = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            cur.append(s[i:i + 2]); i += 2; continue
        if c in "'\"":
            q = c; cur.append(c); i += 1; continue
        if c in "<>":
            redir = True
            segs.append("".join(cur)); cur = []; i += 1; continue
        if c in "|;&\n" or c in "`)":
            segs.append("".join(cur)); cur = []; i += 1; continue
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            segs.append("".join(cur)); cur = []; i += 2; continue
        cur.append(c); i += 1
    segs.append("".join(cur))
    return segs, redir


def tokens(seg):
    try:
        return shlex.split(seg, comments=False, posix=True)
    except ValueError:      # unbalanced quotes: fail closed
        return None


segs, redir = scan(cmd)

# 1. write-shaped: an unquoted redirection, a file-mutating command as a word,
#    or sed in place.
if redir:
    print("WRITE\ta redirection (> or <)"); raise SystemExit
for seg in segs:
    toks = tokens(seg)
    if toks is None:
        print("WORD\t<unparseable quoting>"); raise SystemExit
    for i, t in enumerate(toks):
        if t in WRITE_CMDS:
            print("WRITE\t" + t); raise SystemExit
        if t == "sed":
            for u in toks[i + 1:]:
                if not u.startswith("-"):
                    break
                if u == "--in-place" or re.match(r"^-[A-Za-z]*i", u):
                    print("WRITE\tsed " + u); raise SystemExit

# 2. every /opt/bench path must be under /opt/bench/results/, no traversal.
for p in re.findall(r"/opt/bench[^\s\"'`;|&()<>]*", cmd):
    if ".." in p or not (p == "/opt/bench/results"
                         or p.startswith("/opt/bench/results/")):
        print("PATH\t" + p); raise SystemExit

# 3. the first word of every segment must be a reader.
for seg in segs:
    toks = tokens(seg) or []
    while toks and (toks[0] in KEYWORDS or toks[0] in WRAP_OPTS
                    or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0])):
        w = toks.pop(0)
        if w in WRAP_OPTS:
            while toks and toks[0].startswith("-"):
                o = toks.pop(0)
                if o in WRAP_OPTS[w] and toks:
                    toks.pop(0)
            if w == "timeout" and toks:
                toks.pop(0)          # the duration
    if toks and toks[0] not in READERS:
        print("WORD\t" + toks[0]); raise SystemExit

print("OK")
PY

verdict=$(IJ_HOOK_CMD="$cmd" python3 -c "$INSPECT_PY" 2>/dev/null)

kind=${verdict%%$'\t'*}
detail=${verdict#*$'\t'}

case "$kind" in
  OK) exit 0 ;;
  WRITE)
    deny "write-shaped command mentioning /opt/bench ($detail).
/opt/bench belongs to the bench user and is the ruler; nothing the researcher
runs may write there, and the hook refuses the whole command rather than guess
which side of the redirect the ruler is on.
Read results with a pipe instead (jq ... /opt/bench/results/<id>/results.json),
or copy what you need with the Read tool." ;;
  PATH)
    deny "'$detail' is inside the ruler and is not readable output.
Only /opt/bench/results/** may be read (that is where run.sh writes results.json,
summary.txt, server.log and any profile your experiment left in /out).
The harness and the prompt pools are deliberately out of reach. To have an
experiment measured, write queue/<id>.json; the driver's runner calls the ruler." ;;
  WORD)
    deny "'$detail' is not one of the read-only commands allowed to touch
/opt/bench/results. Allowed: cat jq ls head tail grep wc awk sed sort uniq cut
tr find stat diff od python3 (and shell loops around them).
If you need something else, read the file with one of those, write your copy
somewhere under $PROJECT, and process it there." ;;
  *)
    deny "this hook could not inspect the command (python3 said '${verdict:-nothing}'),
so it refused it. Read the file with the Read tool, or simplify the command." ;;
esac

exit 0

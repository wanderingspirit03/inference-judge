# shellcheck shell=bash
# shellcheck disable=SC2034  # the variables set here are read by the adapters
# Shared by every adapter's session.sh.  Sourced, not executed.
#
# What the loop asks of an adapter (README.md in this folder):
#
#   session.sh run --model M --effort E      (the prompt arrives on stdin)
#       Run ONE researcher session in the current directory, which is the
#       research repo.  Then print ONE JSON object on stdout:
#           {"type":"result","is_error":false,"result":"<the last message>"}
#       with "is_error":true and the agent's own error text in "result" when
#       the session failed.  Exit with the agent's status, and never with 0
#       after a failure.
#   session.sh version       print the agent's version
#   session.sh auth          exit 0: logged in   1: not logged in   2: cannot tell
#
# M is "primary", "fallback" (the loop moves to it after repeated crashes), or
# a model name the operator set in IJ_PRIMARY_MODEL / IJ_FALLBACK_MODEL.

ADAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
# The tripwire every agent's hooks call: research/guard in the repo,
# /opt/inference-judge/guard on a hardened machine.
GUARD_DIR="${IJ_GUARD_DIR:-$(cd "$ADAPTER_DIR/../../guard" 2>/dev/null && pwd)}"

adapter_die() { printf '%s: %s\n' "$(basename "$ADAPTER_DIR")/session.sh" "$*" >&2; exit 64; }

# adapter_args "$@": the arguments after `run`, into MODEL and EFFORT, and the
# prompt, from stdin, into PROMPT.
adapter_args() {
  MODEL=primary EFFORT=high
  while [ $# -gt 0 ]; do
    case "$1" in
      --model)  [ $# -ge 2 ] || adapter_die "--model needs a value";  MODEL="$2";  shift 2 ;;
      --effort) [ $# -ge 2 ] || adapter_die "--effort needs a value"; EFFORT="$2"; shift 2 ;;
      *) adapter_die "unknown argument '$1' (run --model M --effort E, prompt on stdin)" ;;
    esac
  done
  PROMPT=$(cat)
  [ -n "$PROMPT" ] || adapter_die "no prompt on stdin"
}

# adapter_capture <prompt|none> <command...>: run the agent with its stdout in
# the file RAW, wait for it, and leave its status in RC.  `prompt` feeds PROMPT
# to the agent's stdin.  The driver ends a session that has run too long with
# SIGTERM; that is passed on to the agent.
adapter_capture() {
  local feed="$1" pid; shift
  SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/ij-session.XXXXXX") || adapter_die "cannot make a scratch directory"
  trap 'rm -rf "$SCRATCH"' EXIT
  RAW="$SCRATCH/stdout"
  if [ "$feed" = prompt ]; then
    printf '%s\n' "$PROMPT" > "$SCRATCH/prompt"; feed="$SCRATCH/prompt"
  else
    feed=/dev/null
  fi
  "$@" < "$feed" > "$RAW" &
  pid=$!
  trap 'kill -TERM "$pid" 2>/dev/null' TERM INT
  wait "$pid"; RC=$?
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; RC=$?; done
  trap - TERM INT
}

# adapter_no_result: the error result for an agent that printed nothing the
# adapter can read.  It carries the end of whatever the agent did print, so the
# driver still finds a quota or login message in it.
adapter_no_result() {
  jq -R -s -c '{ type: "result", is_error: true,
                 result: ("the agent printed no result" + (if length > 0 then ": " + .[-2000:] else "" end)) }' \
    "$RAW" 2>/dev/null \
    || printf '%s\n' '{"type":"result","is_error":true,"result":"the adapter could not read what the agent printed"}'
}

# adapter_reduce <jq program>: RAW is a stream of JSON events, one per line.
# The program gets them as one array ($rc is the agent's exit status) and
# prints the result object.
adapter_reduce() {
  local out
  out=$(jq -R -c -n --argjson rc "$RC" '
    [inputs | (try fromjson catch empty) | select(type == "object")]
    | if length > 0 then ('"$1"') else empty end' "$RAW" 2>/dev/null)
  if [ -n "$out" ]; then printf '%s\n' "$out"; else adapter_no_result; fi
}

# adapter_finish <result object>: print it and exit with the agent's status.
# A failed session never exits 0, whatever the agent's own status was: the
# driver counts a failure with status 0 as a success.
adapter_finish() {
  printf '%s\n' "$1"
  if [ "$RC" -eq 0 ] && printf '%s' "$1" | jq -e '.is_error == true' >/dev/null 2>&1; then RC=1; fi
  exit "$RC"
}

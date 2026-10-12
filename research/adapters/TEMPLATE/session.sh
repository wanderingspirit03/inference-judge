#!/usr/bin/env bash
# A starting point for your own agent, or your own set of agents.
#
# Copy this folder to adapters/<name>/, change the lines marked EDIT, and set
# IJ_AGENT=<name>.  ../README.md says what the loop expects of this script;
# ../codex/session.sh and ../pi/session.sh are two finished ones.
#
# As it stands it runs any program that reads a prompt on stdin, works in the
# current directory, and prints its answer as plain text.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
BIN="${IJ_AGENT_BIN:-my-agent}"                       # EDIT: your agent's command

case "${1:-}" in
  version)
    exec "$BIN" --version ;;                          # EDIT
  auth)
    # EDIT: exit 0 when the agent is logged in, 1 when it is not.  2 means
    # "cannot tell", and setup then only reports that it did not check.
    exit 2 ;;
  run)
    shift; adapter_args "$@"                          # sets MODEL, EFFORT and PROMPT
    # EDIT: start ONE session that needs no person: it must not stop to ask for
    # permission.  This is also the place to start several agents, or an
    # orchestrator of your own; the loop only reads the one object printed below
    # and the files the session leaves in the research repo.
    #
    # Have your agent send each shell command and each file edit through the
    # guard before running it ($GUARD_DIR, see ../README.md).
    adapter_capture prompt "$BIN"
    # EDIT: turn what the agent printed (the file $RAW) into the result object.
    out=$(jq -n -c --rawfile text "$RAW" --argjson rc "$RC" \
      '{type: "result", agent: "my-agent", is_error: ($rc != 0),
        result: (if $rc != 0 and ($text | length) == 0 then "the session ended without a result" else $text end)}')
    adapter_finish "$out" ;;
  *)
    adapter_die "usage: session.sh run --model M --effort E < prompt | version | auth" ;;
esac

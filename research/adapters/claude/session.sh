#!/usr/bin/env bash
# Claude Code as the researcher.  See ../README.md for what the loop asks of
# an adapter.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
BIN="${IJ_AGENT_BIN:-claude}"

# Which model plays which part.  Any other name the loop sends is passed on.
PRIMARY=opus
FALLBACK=sonnet

case "${1:-}" in
  version)
    exec "$BIN" --version ;;
  auth)
    # `claude auth status` prints {"loggedIn": true|false, ...}.  A Claude Code
    # too old to have that command prints no such object; then all that can
    # be said is whether it has ever stored anything under the home directory.
    case "$("$BIN" auth status 2>/dev/null </dev/null | jq -r '.loggedIn? | select(type == "boolean")' 2>/dev/null)" in
      true) exit 0 ;;
      false) exit 1 ;;
    esac
    [ -s "$HOME/.claude.json" ] || [ -d "$HOME/.claude" ] || exit 1
    exit 2 ;;
  run)
    shift; adapter_args "$@"
    case "$MODEL" in primary) MODEL=$PRIMARY ;; fallback) MODEL=$FALLBACK ;; esac
    # --effort covers the session; the helper agents carry `effort: high` in
    # their own files and .claude/settings.json sets effortLevel for the rest.
    CLAUDE_CODE_EFFORT_LEVEL="$EFFORT" adapter_capture none "$BIN" -p "$PROMPT" --model "$MODEL" \
      --effort "$EFFORT" --dangerously-skip-permissions --output-format json
    # Claude Code's JSON result is already the object the driver reads.
    if jq -e 'type == "object"' "$RAW" >/dev/null 2>&1; then out=$(cat "$RAW"); else out=$(adapter_no_result); fi
    adapter_finish "$out" ;;
  *)
    adapter_die "usage: session.sh run --model M --effort E < prompt | version | auth" ;;
esac

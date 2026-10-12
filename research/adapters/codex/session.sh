#!/usr/bin/env bash
# Codex CLI as the researcher.  See ../README.md for what the loop asks of an
# adapter.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
BIN="${IJ_AGENT_BIN:-codex}"

# `codex exec --json` prints one event per line.  The session's end is its
# last turn event: turn.completed, or turn.failed with the reason.  An "error"
# event on its own can be a retry that later succeeded, so it only names the
# reason when no turn finished.
REDUCE='
  . as $ev
  | ([$ev[] | select(.type == "turn.completed" or .type == "turn.failed")] | last) as $end
  | ([$ev[] | select(.type == "turn.completed") | (.usage // {})]) as $use
  | ([$ev[] | select(.type == "error") | .message] | last) as $err
  | (if $end == null then $rc != 0 else $end.type == "turn.failed" end) as $bad
  | { type: "result", agent: "codex", is_error: $bad,
      result: (if $bad then ($end.error.message? // $err // "the session ended without a result")
               else ([$ev[] | select(.type == "item.completed" and .item.type == "agent_message") | .item.text] | last // "") end),
      num_turns: ($use | length),
      usage: { input_tokens:            ([$use[] | .input_tokens // 0] | add // 0),
               cached_input_tokens:     ([$use[] | .cached_input_tokens // 0] | add // 0),
               output_tokens:           ([$use[] | .output_tokens // 0] | add // 0),
               reasoning_output_tokens: ([$use[] | .reasoning_output_tokens // 0] | add // 0) },
      session_id: ([$ev[] | select(.type == "thread.started") | .thread_id] | first) }'

case "${1:-}" in
  version)
    exec "$BIN" --version ;;
  auth)
    "$BIN" login status >/dev/null 2>&1 && exit 0
    exit 1 ;;
  run)
    shift; adapter_args "$@"
    # The Unix users are the walls (setup/harden_research.sh), so Codex's own
    # sandbox and approval prompts are off, as Claude Code's are.
    args=(exec --json --skip-git-repo-check
          --dangerously-bypass-approvals-and-sandbox --dangerously-bypass-hook-trust)
    # "primary" and "fallback" both mean the model Codex is configured to use.
    case "$MODEL" in primary|fallback) ;; *) args+=(-m "$MODEL") ;; esac
    args+=(-c "model_reasoning_effort=\"$EFFORT\"" -)
    adapter_capture prompt "$BIN" "${args[@]}"
    out=$(adapter_reduce "$REDUCE")
    # A failed session with no login is a login failure, whatever the message says.
    if [ "$(printf '%s' "$out" | jq -r '.is_error? // false' 2>/dev/null)" = true ] \
       && ! "$BIN" login status >/dev/null 2>&1; then
      out=$(printf '%s' "$out" | jq -c '.result = "not logged in: " + .result')
    fi
    adapter_finish "$out" ;;
  *)
    adapter_die "usage: session.sh run --model M --effort E < prompt | version | auth" ;;
esac

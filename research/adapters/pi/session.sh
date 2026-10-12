#!/usr/bin/env bash
# Pi as the researcher.  See ../README.md for what the loop asks of an adapter.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
BIN="${IJ_AGENT_BIN:-pi}"

# `pi --mode json` prints one event per line.  The session's end is its last
# assistant message: stopReason "error" or "aborted" is a failure, and
# errorMessage is the reason.
REDUCE='
  . as $ev
  | ([$ev[] | select(.type == "message_end" and .message.role == "assistant") | .message]) as $msgs
  | ($msgs | last) as $end
  | (if $end == null then $rc != 0 else ($end.stopReason == "error" or $end.stopReason == "aborted") end) as $bad
  | { type: "result", agent: "pi", is_error: $bad,
      result: (if $bad then ($end.errorMessage? // "the session ended without a result")
               else ([($end.content // [])[] | select(.type == "text") | .text] | join("")) end),
      num_turns: ([$ev[] | select(.type == "turn_end")] | length),
      total_cost_usd: ([$msgs[] | .usage.cost.total // 0] | add // 0),
      usage: { input_tokens:                ([$msgs[] | .usage.input // 0] | add // 0),
               output_tokens:               ([$msgs[] | .usage.output // 0] | add // 0),
               cache_read_input_tokens:     ([$msgs[] | .usage.cacheRead // 0] | add // 0),
               cache_creation_input_tokens: ([$msgs[] | .usage.cacheWrite // 0] | add // 0) },
      session_id: ([$ev[] | select(.type == "session") | .id] | first) }'

# Pi checks a login per model, so it can only be asked when a model is named.
logged_in() {   # <model>; 0 ready, 1 not ready or invalid, 2 cannot tell
  case "$1" in primary|fallback|"") return 2 ;; esac
  "$BIN" auth check --model "$1" >/dev/null 2>&1 && return 0
  return 1
}

case "${1:-}" in
  version)
    exec "$BIN" --version ;;
  auth)
    logged_in "${IJ_PRIMARY_MODEL:-primary}"; exit $? ;;
  run)
    shift; adapter_args "$@"
    # guard.ts connects Pi's tool calls to the guard.  It is named here, so it
    # loads without a trust question; --no-approve keeps Pi from loading
    # extensions the researcher may have written into the research repo.
    args=(--mode json --no-approve -e "$ADAPTER_DIR/guard.ts")
    # "primary" and "fallback" both mean the model Pi is configured to use.
    case "$MODEL" in primary|fallback) ;; *) args+=(--model "$MODEL") ;; esac
    args+=(--thinking "$EFFORT" "$PROMPT")
    IJ_GUARD_DIR="$GUARD_DIR" adapter_capture none "$BIN" "${args[@]}"
    out=$(adapter_reduce "$REDUCE")
    # A failed session with no login is a login failure, whatever the message says.
    if [ "$(printf '%s' "$out" | jq -r '.is_error? // false' 2>/dev/null)" = true ]; then
      logged_in "$MODEL"
      [ $? -eq 1 ] && out=$(printf '%s' "$out" | jq -c '.result = "not logged in: " + .result')
    fi
    adapter_finish "$out" ;;
  *)
    adapter_die "usage: session.sh run --model M --effort E < prompt | version | auth" ;;
esac

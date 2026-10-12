#!/usr/bin/env bash
# The literature pass: run once at hour 0 (before the loop starts) and then
# daily from cron. It reads, it does not run experiments; its only output is new
# entries in HYPOTHESES.md and a commit.
#
#     ./loop/literature.sh            # run it
#     ./loop/literature.sh --dry-run  # print the prompt and exit
#     ./loop/run.sh --install-cron    # daily at 00:00 UTC
set -uo pipefail

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${IJ_LITERATURE_LOG:-$RESEARCH_DIR/loop/literature.log}"
# The agent is reached the way the driver reaches it: through its adapter.
ADAPTER="$RESEARCH_DIR/adapters/${IJ_AGENT:-claude}/session.sh"
MODEL="${IJ_MODEL:-primary}"
EFFORT="${IJ_EFFORT:-high}"
TIMEOUT_S="${IJ_LITERATURE_TIMEOUT:-3600}"
RETRY_SLEEP="${IJ_RATE_LIMIT_SLEEP:-900}"

read -r -d '' PROMPT <<'PROMPT_END' || true
You are the inference researcher (see AGENTS.md). This is the literature pass, not an
experiment. Read the target first (/etc/inference-judge/target.yaml). Survey, with helper agents
if you have them: the target engine's release notes and merged PRs from the last 60 days; any
serving write-ups for the target's model; papers on fast inference for this kind of model and
accelerator. For each finding that could improve the target's metric, append to HYPOTHESES.md:
tag, one-line hypothesis, mechanism, link, estimated cost to try, and whether the engine already
ships it. Then list the five ideas you believe nobody has tried for this model on this
accelerator and why. Do not run experiments. Commit.
PROMPT_END

case "${1:-}" in
  -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
  --dry-run) printf '%s\n' "$PROMPT"; echo; echo "(would run: $ADAPTER run --model $MODEL --effort $EFFORT < the prompt above, in $RESEARCH_DIR)"; exit 0 ;;
  "") ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

[ -x "$ADAPTER" ] || { echo "no adapter at $ADAPTER (IJ_AGENT=${IJ_AGENT:-claude})" >&2; exit 1; }
TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"
cd "$RESEARCH_DIR" || exit 1
printf '%s literature: start\n' "$(date -u +%FT%TZ)" >> "$LOG"

attempt() {
  if [ -n "$TIMEOUT_BIN" ]; then
    printf '%s\n' "$PROMPT" | "$TIMEOUT_BIN" "$TIMEOUT_S" "$ADAPTER" run --model "$MODEL" --effort "$EFFORT"
  else
    printf '%s\n' "$PROMPT" | "$ADAPTER" run --model "$MODEL" --effort "$EFFORT"
  fi
}

out=$(attempt 2>&1); rc=$?
if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -qiE 'rate.?limit|usage limit|429'; then
  printf '%s literature: rate limited, retrying in %ss\n' "$(date -u +%FT%TZ)" "$RETRY_SLEEP" >> "$LOG"
  sleep "$RETRY_SLEEP"
  out=$(attempt 2>&1); rc=$?
fi

printf '%s literature: end rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG"
printf '%s\n' "$out" | tail -c 4000 >> "$LOG"
# The pass is told to commit; commit anything it left uncommitted anyway.
git add -A >/dev/null 2>&1 || true
git diff --cached --quiet 2>/dev/null || git commit -qm "literature pass $(date -u +%F)" >/dev/null 2>&1 || true
exit "$rc"

#!/usr/bin/env bash
# Pre-flight for one experiment (DESIGN 2.7).  Run it before you
# queue anything; the runner runs it again before it spends a ruler slot.
#
#     loop/preflight.sh <id>
#
# Checks, all on the CPU, in seconds:
#   - the id is well formed, not reserved (baseline-*) and not used before
#     (no results dir, no verdict in the driver's runs.jsonl);
#   - experiments/<id>/launch.sh exists, passes `bash -n` and
#     `shellcheck -S error`;
#   - every experiments/<id>/patches/*.diff|*.patch applies (`git apply
#     --check`) to at least one engine checkout in IJ_PATCH_TREES;
#   - experiments/<id>/deps.txt, if present, parses as pip requirements.
#
# A server that never comes up, or a reserved id, would otherwise cost a slot
# on the judge. Every one of those is caught here or by the bench's own smoke
# step, and a failure here costs no slot.
#
# The LAST line is the verdict, for the researcher and for the runner:
#     PREFLIGHT_OK <id>
#     PREFLIGHT_FAILED <id>: <reason>; <reason>
# Exit 0 when OK, 1 when FAILED, 2 on bad usage.
set -uo pipefail

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="${IJ_RESEARCH_DIR:-$(dirname "$LOOP_DIR")}"
RESULTS_DIR="${IJ_RESULTS_DIR:-/opt/bench/results}"
STATE_DIR="${IJ_STATE_DIR:-$HOME/research-state}"
# The engine trees the target lists (engine.sources), as bootstrap wrote them;
# a copy of engine_source_roots in guard/_lib.sh.  No file: no tree.
engine_trees() {
  local file="${IJ_ENGINE_SOURCES_FILE:-/etc/inference-judge/engine-sources}" n out=""
  [ -r "$file" ] || return 0
  for n in $(grep -E '^[a-z0-9][a-z0-9_-]*$' "$file" 2>/dev/null); do out="${out:+$out:}$HOME/$n-src"; done
  printf '%s' "$out"
}
PATCH_TREES="${IJ_PATCH_TREES:-$(engine_trees)}"
PY="${IJ_PY:-python3}"
# bench/run.sh accepts ^[a-z0-9][a-z0-9-]{2,40}$.
# The same rule here makes a long id fail now, not at the bench.
ID_RE='^[a-z0-9][a-z0-9-]{2,40}$'

[ $# -eq 1 ] || { echo "usage: loop/preflight.sh <id>" >&2; exit 2; }
ID="$1"
FAILS=()
fail() {   # one line per reason, whatever the tool printed
  local r="${1//$'\n'/ }"
  FAILS+=("$r"); printf 'FAIL: %s\n' "$r"
}
ok()   { printf 'ok:   %s\n' "$1"; }

verdict() {
  if [ "${#FAILS[@]}" -eq 0 ]; then
    printf 'PREFLIGHT_OK %s\n' "$ID"; exit 0
  fi
  local IFS=';'
  printf 'PREFLIGHT_FAILED %s: %s\n' "$ID" "${FAILS[*]}"; exit 1
}

# ------------------------------------------------------------------ the id --
if ! [[ "$ID" =~ $ID_RE ]]; then
  ID="${ID//[^a-z0-9-]/?}"
  fail "id does not match $ID_RE"; verdict
fi
case "$ID" in baseline-*) fail "baseline-* ids are reserved for the operator" ;; esac
[ -e "$RESULTS_DIR/$ID" ] && fail "id already used: $RESULTS_DIR/$ID exists"
if [ -f "$STATE_DIR/runs.jsonl" ] && command -v jq >/dev/null 2>&1; then
  used=$(jq -r --arg id "$ID" 'select(.id == $id and .verdict != "PREFLIGHT_FAILED") | .verdict' \
         "$STATE_DIR/runs.jsonl" 2>/dev/null | head -1)
  [ -n "$used" ] && fail "id already used: runs.jsonl has a $used verdict for it"
fi
[ "${#FAILS[@]}" -eq 0 ] && ok "id $ID is free"

# --------------------------------------------------------------- launch.sh --
EXP="$RESEARCH_DIR/experiments/$ID"
if [ ! -f "$EXP/launch.sh" ]; then
  fail "missing experiments/$ID/launch.sh"
else
  if err=$(bash -n "$EXP/launch.sh" 2>&1); then
    ok "launch.sh parses (bash -n)"
  else
    fail "launch.sh does not parse: $(printf '%s' "$err" | head -1)"
  fi
  if command -v shellcheck >/dev/null 2>&1; then
    if err=$(shellcheck -S error -f gcc "$EXP/launch.sh" 2>&1); then
      ok "launch.sh passes shellcheck -S error"
    else
      fail "shellcheck error: $(printf '%s' "$err" | head -1 | sed "s|$EXP/||")"
    fi
  else
    printf 'warn: shellcheck is not installed; error-level lint skipped\n'
  fi
fi

# ----------------------------------------------------------------- patches --
shopt -s nullglob
for p in "$EXP"/patches/*.diff "$EXP"/patches/*.patch; do
  applied=""
  IFS=: read -r -a trees <<< "$PATCH_TREES"
  for t in ${trees[@]+"${trees[@]}"}; do
    [ -d "$t" ] || continue
    if git -C "$t" apply --check "$p" >/dev/null 2>&1; then applied="$t"; break; fi
  done
  if [ -n "$applied" ]; then
    ok "patch $(basename "$p") applies to $applied"
  else
    fail "patch $(basename "$p") applies to none of ${PATCH_TREES:-the engine trees: the target lists none}"
  fi
done
shopt -u nullglob

# ----------------------------------------------------------------- deps.txt --
if [ -f "$EXP/deps.txt" ]; then
  DEPS_PY='
import re, sys
try:
    from packaging.requirements import Requirement, InvalidRequirement
except ImportError:
    Requirement = None
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._ -]+\])?\s*((==|>=|<=|~=|!=|>|<|===)\s*[A-Za-z0-9.*+!_-]+\s*,?\s*)*(;.*)?$")
OPTS = ("--extra-index-url", "--index-url", "-i", "--find-links", "-f", "--pre", "--no-deps")
bad = []
for n, raw in enumerate(open(sys.argv[1], encoding="utf-8", errors="replace"), 1):
    line = raw.split(" #")[0].strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("-"):
        if line.split()[0].split("=")[0] not in OPTS:
            bad.append(f"line {n}: option {line.split()[0]!r} is not allowed")
        continue
    if re.match(r"^(git\+|https?://)", line) or " @ " in line:
        continue
    if Requirement is not None:
        try:
            Requirement(line)
        except InvalidRequirement as e:
            bad.append(f"line {n}: " + " ".join(str(e).split())[:120])
    elif not NAME.match(line):
        bad.append(f"line {n}: not a requirement: {line[:60]!r}")
print("; ".join(bad))
'
  if ! command -v "$PY" >/dev/null 2>&1; then
    fail "deps.txt present but $PY is missing, cannot parse it"
  else
    err=$("$PY" -I -c "$DEPS_PY" "$EXP/deps.txt" 2>&1)
    if [ -z "$err" ]; then ok "deps.txt parses"; else fail "deps.txt: $err"; fi
  fi
fi

verdict

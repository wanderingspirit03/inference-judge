# shellcheck shell=bash
# shellcheck disable=SC2034  # the variables set here are read by the scripts that source this
# Shared helpers for the inference-judge guard.  Sourced, not executed.
#
# A guard script receives one JSON object on stdin, e.g.
#   {"cwd":"/home/researcher/research","tool_name":"Write",
#    "tool_input":{"file_path":"...","content":"..."}}
# and blocks the tool call by exiting 2 with an explanation on stderr.  Claude
# Code and Codex call the scripts as hooks; Pi calls them from an extension.
# A hook that exits with anything else is a hook error, and the tool call may
# still run, so these scripts must only ever exit 0 or 2.

HOOK_INPUT="${HOOK_INPUT-$(cat)}"

jqr() { printf '%s' "$HOOK_INPUT" | jq -r "$1 // empty" 2>/dev/null; }

deny() { printf 'BLOCKED by inference-judge hook: %s\n' "$1" >&2; exit 2; }

# What setup knows and an agent's own environment may not say: plain KEY=value
# lines in a file that setup/harden_research.sh writes and only root can
# change.  Only IJ_ names are taken, and a variable already in the
# environment wins.
GUARD_ENV="${IJ_GUARD_ENV:-/etc/inference-judge/guard.env}"
if [ -r "$GUARD_ENV" ]; then
  while IFS='=' read -r _k _v || [ -n "$_k" ]; do
    [[ "$_k" =~ ^IJ_[A-Z0-9_]+$ ]] || continue
    [ -n "${!_k:-}" ] || export "$_k=$_v"
  done < "$GUARD_ENV"
  unset _k _v
fi

# The research repo: named by setup, else by Claude Code, else the directory
# this guard sits in.  The last is right for a guard run from the repo; for the
# installed copy it names no research repo, and every edit is then refused.
PROJECT="${IJ_RESEARCH_DIR:-${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

# Frozen for the run: the loop, the guard, the adapters, the agent's own
# settings folder, and the driver's state dir (wherever this setup keeps it).
PROTECTED="${IJ_PROTECTED_ROOTS:-$PROJECT/loop:$PROJECT/guard:$PROJECT/adapters:$PROJECT/.claude:$PROJECT/.codex:/home/driver/research-state:$(dirname "$PROJECT")/research-state:$HOME/research-state${IJ_STATE_DIR:+:$IJ_STATE_DIR}}"
FROZEN_WORDS="loop/ (driver, runner, preflight), guard/, adapters/ and the agent's
settings folder (.claude/ or .codex/) are fixed for the whole run, and the
driver's state dir (BRIEF.md, events.jsonl, runs.jsonl) is written only by the
driver. Read them freely. If one of them is wrong, say so in LAB_NOTEBOOK.md
and NEXT.md; the operator will change it."

# Normalise a path: make it absolute against $2, then resolve . and .. (and
# symlinks, where the platform can).  GNU realpath -m on the box; python3
# elsewhere so the hooks are testable on a Mac.
normalise_path() {
  local p="$1" base="${2:-$PWD}"
  case "$p" in /*) ;; *) p="$base/$p" ;; esac
  if realpath -m / >/dev/null 2>&1; then
    realpath -m -- "$p"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import os,sys; print(os.path.normpath(sys.argv[1]))' "$p"
  else
    # No normaliser available: fail closed by keeping the raw path, which still
    # matches the literal roots below for well-formed absolute paths.
    printf '%s\n' "$p"
  fi
}

# True when $1 is inside (or equal to) one of the colon-separated roots in $2.
path_within() {
  local path="$1" roots="$2" root
  local IFS=:
  for root in $roots; do
    [ -z "$root" ] && continue
    [ "$path" = "$root" ] && return 0
    case "$path" in "$root"/*) return 0 ;; esac
  done
  return 1
}

# The engine source trees the researcher may edit: one name per line in
# $IJ_ENGINE_SOURCES_FILE (default /etc/inference-judge/engine-sources), written root-only
# by setup/bootstrap.sh from the target's engine.sources.  The tree <name> lives at
# <home>/<name>-src, next to the research repo.  No file means no tree: nothing
# outside the research repo may be edited.  Prints the colon-separated roots.  loop/preflight.sh carries a copy of
# this reader: the two directories are installed apart and cannot share a file.
engine_source_roots() {
  local home="$1" file="${IJ_ENGINE_SOURCES_FILE:-/etc/inference-judge/engine-sources}" names n out=""
  if [ -r "$file" ]; then names=$(grep -E '^[a-z0-9][a-z0-9_-]*$' "$file" 2>/dev/null)
  else names=""; fi
  for n in $names; do out="${out:+$out:}$home/$n-src"; done
  printf '%s' "$out"
}

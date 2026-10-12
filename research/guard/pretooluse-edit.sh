#!/usr/bin/env bash
# Runs before a file edit: Edit | Write | MultiEdit | NotebookEdit (Claude
# Code), apply_patch (Codex), write | edit (Pi).
#
# Rule: the researcher may write only inside its own research repo and the
# engine checkouts the target lists (engine.sources; none unless it lists them).
# Everything else -- /opt/bench above all, but also /etc, /opt/models and other
# users' homes -- is denied.  Reading is not restricted; only writes are.
#
# Inside the research repo, loop/ (driver, runner, preflight), guard/,
# adapters/ and the agent's settings folder are frozen for the run, and the
# driver's state dir is the driver's.  Tripwire only: the boundary is
# setup/harden_research.sh, which makes them root- or driver-owned.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

ALLOWED_ROOTS="${IJ_ALLOWED_WRITE_ROOTS:-$PROJECT:$(engine_source_roots "$(dirname "$PROJECT")")}"

tool=$(jqr '.tool_name')
cwd=$(jqr '.cwd'); cwd="${cwd:-$PWD}"

if [ "$tool" = apply_patch ]; then
  # Codex sends the whole patch as tool_input.command.  The files it changes
  # are its "*** Add File: ", "*** Update File: ", "*** Delete File: " and
  # "*** Move to: " lines; lines of file content start with +, - or a space.
  raws=$(jqr '.tool_input.command' | tr -d '\r' \
         | sed -n -E 's/^\*\*\* (Add File|Update File|Delete File|Move to): (.*[^ ]) *$/\2/p')
  # A patch this script cannot read is refused: allowing it would allow any file.
  [ -z "$raws" ] && deny "apply_patch: the guard found no '*** Add File:', '*** Update File:' or
'*** Delete File:' line in the patch, so it cannot tell which files would change."
else
  # Edit/Write/MultiEdit use file_path; NotebookEdit uses notebook_path; Pi uses path.
  raws=$(jqr '.tool_input.file_path')
  [ -z "$raws" ] && raws=$(jqr '.tool_input.notebook_path')
  [ -z "$raws" ] && raws=$(jqr '.tool_input.path')
  # No path in the payload: nothing to judge, let the agent's own tool validate it.
  [ -z "$raws" ] && exit 0
fi

while IFS= read -r raw; do
  [ -z "$raw" ] && continue
  path=$(normalise_path "$raw" "$cwd")

  if path_within "$path" "$PROTECTED"; then
    deny "$tool to '$path': this file is frozen for the run.
$FROZEN_WORDS"
  fi

  path_within "$path" "$ALLOWED_ROOTS" && continue

  deny "$tool to '$path' is outside the writable roots ($ALLOWED_ROOTS).
/opt/bench is the ruler and belongs to another user: queue experiments with
queue/<id>.json (the driver's runner calls the ruler) instead of editing the harness.
Put experiment code under $PROJECT/experiments/<id>/."
done <<EOF
$raws
EOF
exit 0

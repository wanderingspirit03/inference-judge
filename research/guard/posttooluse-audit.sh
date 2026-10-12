#!/usr/bin/env bash
# Runs after every tool call: append one line per call to audit.log.
# This script must never block, so it always exits 0.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

LOG="${IJ_AUDIT_LOG:-$PROJECT/audit.log}"

tool=$(jqr '.tool_name')
summary=$(printf '%s' "$HOOK_INPUT" | jq -c '.tool_input // {}' 2>/dev/null | tr '\n\t' '  ')
summary=${summary:0:200}

printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tool:-unknown}" "$summary" >> "$LOG" 2>/dev/null || true
exit 0

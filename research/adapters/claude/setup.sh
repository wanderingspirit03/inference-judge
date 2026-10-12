#!/usr/bin/env bash
# What setup needs to know about Claude Code.  ../README.md lists the verbs.
set -euo pipefail

case "${1:-}" in
  install)
    npm install -g @anthropic-ai/claude-code ;;
  login)
    echo "claude        (log in when it asks, then /exit)" ;;
  managed-path)
    echo /etc/claude-code/managed-settings.json ;;
  managed-config)
    # Root-owned settings that Claude Code reads before any other.  Only these
    # hooks run (allowManagedHooksOnly), and the deny rules hold in
    # bypassPermissions mode too.  Both are tripwires; the boundary is file
    # ownership.
    : "${INSTALL_DIR:?}" "${RESEARCH_DIR:?}" "${STATE_DIR:?}" "${RESULTS_DIR:?}" "${FROZEN:?}"
    deny=""
    IFS=: read -r -a frozen <<< "$FROZEN"
    for p in "${frozen[@]}"; do
      [ -n "$p" ] || continue
      deny="${deny:+$deny,
      }\"Edit(/$p/**)\""
    done
    cat <<JSON
{
  "\$comment": "Installed by setup/harden_research.sh. Only these hooks run (allowManagedHooksOnly); they are root-owned copies in $INSTALL_DIR/guard. The deny rules hold in bypassPermissions mode too. Both are tripwires; the boundary is file ownership.",
  "allowManagedHooksOnly": true,
  "hooks": {
    "PreToolUse": [
      {"matcher": "Edit|Write|MultiEdit|NotebookEdit",
       "hooks": [{"type": "command", "command": "$INSTALL_DIR/guard/pretooluse-edit.sh", "timeout": 15}]},
      {"matcher": "Bash",
       "hooks": [{"type": "command", "command": "$INSTALL_DIR/guard/pretooluse-bash.sh", "timeout": 15}]}
    ],
    "PostToolUse": [
      {"matcher": "*",
       "hooks": [{"type": "command", "command": "$INSTALL_DIR/guard/posttooluse-audit.sh", "timeout": 15}]}
    ]
  },
  "permissions": {
    "deny": [
      $deny
    ]
  },
  "env": {"IJ_PROTECTED_ROOTS": "$FROZEN",
          "IJ_STATE_DIR": "$STATE_DIR", "IJ_RESEARCH_DIR": "$RESEARCH_DIR",
          "IJ_RESULTS_DIR": "$RESULTS_DIR"}
}
JSON
    ;;
  *)
    echo "usage: setup.sh install | login | managed-path | managed-config" >&2; exit 64 ;;
esac

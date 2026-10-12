#!/usr/bin/env bash
# What setup needs to know about the Codex CLI.  ../README.md lists the verbs.
set -euo pipefail

case "${1:-}" in
  install)
    npm install -g @openai/codex ;;
  login)
    echo "codex login   (or: codex login --device-auth on a machine with no browser)" ;;
  managed-path)
    echo /etc/codex/requirements.toml ;;
  managed-config)
    # Root-owned requirements that Codex reads before any other configuration.
    # Hooks are switched on, only the hooks listed here run, and they are the
    # root-owned copies of the guard.  A tripwire; the boundary is file
    # ownership.
    : "${INSTALL_DIR:?}"
    cat <<TOML
# Installed by setup/harden_research.sh.  Only these hooks run
# (allow_managed_hooks_only); they are root-owned copies in $INSTALL_DIR/guard.
allow_managed_hooks_only = true

[features]
hooks = true

[hooks]
managed_dir = "$INSTALL_DIR"

[[hooks.PreToolUse]]
matcher = "^apply_patch\$"

[[hooks.PreToolUse.hooks]]
type = "command"
command = "$INSTALL_DIR/guard/pretooluse-edit.sh"
timeout = 15

[[hooks.PreToolUse]]
matcher = "^Bash\$"

[[hooks.PreToolUse.hooks]]
type = "command"
command = "$INSTALL_DIR/guard/pretooluse-bash.sh"
timeout = 15

[[hooks.PostToolUse]]

[[hooks.PostToolUse.hooks]]
type = "command"
command = "$INSTALL_DIR/guard/posttooluse-audit.sh"
timeout = 15
TOML
    ;;
  *)
    echo "usage: setup.sh install | login | managed-path | managed-config" >&2; exit 64 ;;
esac

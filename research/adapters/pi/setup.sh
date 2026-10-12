#!/usr/bin/env bash
# What setup needs to know about Pi.  ../README.md lists the verbs.
set -euo pipefail

case "${1:-}" in
  install)
    npm install -g @earendil-works/pi-coding-agent ;;
  login)
    echo "pi            (type /login, pick a provider, then /quit; or put the provider's API key in the researcher's environment)" ;;
  managed-path|managed-config)
    # Pi has no root-owned settings file.  The guard is loaded by session.sh,
    # which is itself root-owned on a hardened machine.
    : ;;
  *)
    echo "usage: setup.sh install | login | managed-path | managed-config" >&2; exit 64 ;;
esac

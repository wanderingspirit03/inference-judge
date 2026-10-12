"""6. Guard rails of the research agent: the JSON an agent's hook sends before
a shell command, fed to this tree's guard (research/guard).

The agent must not touch the accelerator while a scored run may be using it
(that would disturb the measurement) and must not rewrite its own settings.
Ordinary commands must still go through, or the guard rail would be passing
by refusing everything.

Exit 2 is a deny, exit 0 an allow (research/guard/_lib.sh); anything else is
a hook error, which an agent may treat as "run the tool anyway".
"""
from __future__ import annotations

import json
import os

from core import END_TO_END, Outcome
from core import run as sh

HOOK = "research/guard/pretooluse-bash.sh"
RESEARCH = "/home/researcher/research"      # the research repo the payloads name
MUST_DENY = [
    "/usr/bin/nvidia-smi",
    "python3 -c 'import torch; torch.cuda.init()'",
    "python3 -c 'import jax'",
    "ls /dev/accel0",
    f"sed -i 's/deny/allow/' {RESEARCH}/.claude/settings.json",
]
MUST_ALLOW = [
    "date -u",
    "sleep 5",
    "xargs -n1 echo < f",
    "timeout 5 true",
    "cat LEADERBOARD.md | head",
]


def decide(hook, command: str) -> str:
    payload = {"session_id": "scorecard", "hook_event_name": "PreToolUse",
               "cwd": RESEARCH, "tool_name": "Bash",
               "tool_input": {"command": command}}
    # the guard is told where the research repo is; nothing of this machine's leaks in
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp",
           "IJ_RESEARCH_DIR": RESEARCH, "IJ_GUARD_ENV": "/nonexistent/guard.env"}
    p = sh(["bash", str(hook)], timeout=20, env=env, input=json.dumps(payload))
    return {0: "allow", 2: "deny"}.get(p.returncode, f"error rc={p.returncode}")


def run(judge, ctx) -> list:
    hook = judge.script(HOOK)
    deny = {c: decide(hook, c) for c in MUST_DENY}
    allow = {c: decide(hook, c) for c in MUST_ALLOW}
    nd = sum(v == "deny" for v in deny.values())
    na = sum(v == "allow" for v in allow.values())
    leaks = [c for c, v in deny.items() if v != "deny"]
    blocks = [c for c, v in allow.items() if v != "allow"]
    return [
        Outcome("6a", f"the agent running {len(MUST_DENY)} commands on the accelerator or its settings", f"deny all {len(MUST_DENY)}",
                f"{nd}/{len(MUST_DENY)} denied" + (f"; let through: {leaks[0][:30]}..."
                                                   if leaks else ""),
                nd == len(MUST_DENY), END_TO_END, deny),
        Outcome("6b", f"the agent running {len(MUST_ALLOW)} harmless commands (the control)", f"allow all {len(MUST_ALLOW)}",
                f"{na}/{len(MUST_ALLOW)} allowed" + (f"; blocked: {blocks[0][:30]}"
                                                     if blocks else ""),
                na == len(MUST_ALLOW), END_TO_END, allow),
    ]

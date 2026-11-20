#!/usr/bin/env python3
"""prime_settings.py CONFIG SETTINGS_JSON [OWNER]: write the controller's
Prime Agent settings (config agent.settings) into Prime Agent's settings.json,
as root, at every install.

Every key of agent.settings is set to the config's value (a dict such as
agentTraces is set key by key); every other key in the file (what Prime
Agent writes itself: recentModels, a login's choices) is kept.  The file is
replaced in one step (a new file, then rename), owned by OWNER (default
agents), mode 600.

Some keys are limits Prime Agent enforces: allowedModels (an array of
"provider/model-id" selectors; Prime Agent refuses every other model and
fails closed on a value that is not an array of strings), defaultProvider,
defaultModel, defaultThinkingLevel.  This checks their shape before it
writes, so a wrong type never reaches the file.  agentTraces.enabled false
and onboardingShown true answer the first-run question about sharing traces
("Not now")."""
import json
import os
import pwd
import sys
import tempfile
from pathlib import Path

LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def check(s: dict) -> None:
    a = s.get("allowedModels")
    if a is not None and not (isinstance(a, list) and a and all(isinstance(x, str) and "/" in x for x in a)):
        raise SystemExit("agent.settings.allowedModels must be a non-empty array of \"provider/model-id\" strings")
    if a and s.get("defaultProvider") and s.get("defaultModel") \
            and f"{s['defaultProvider']}/{s['defaultModel']}" not in a:
        raise SystemExit("agent.settings: defaultProvider/defaultModel is not in allowedModels")
    if s.get("defaultThinkingLevel") is not None and s["defaultThinkingLevel"] not in LEVELS:
        raise SystemExit(f"agent.settings.defaultThinkingLevel must be one of {', '.join(LEVELS)}")


def merge(cur: dict, want: dict) -> dict:
    out = dict(cur)
    for k, v in want.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def main(a) -> int:
    if len(a) not in (2, 3):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    want = {k: v for k, v in json.loads(Path(a[0]).read_text())["agent"]["settings"].items() if not k.startswith("_")}
    check(want)
    t = Path(a[1])
    cur = {}
    if t.exists():
        try:
            cur = json.loads(t.read_text() or "{}")
        except ValueError:
            raise SystemExit(f"{t} is not JSON; not changed (fix or move it, then install again)") from None
        if not isinstance(cur, dict):
            raise SystemExit(f"{t} is not a JSON object; not changed")
    new = merge(cur, want)
    if new == cur and t.exists():
        print(f"[install] {t}: the controller's settings are in place")
        return 0
    pw = pwd.getpwnam(a[2] if len(a) == 3 else "agents")
    t.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".settings.", dir=t.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(new, indent=2) + "\n")
        os.chmod(tmp, 0o600)
        if os.geteuid() == 0:
            os.chown(tmp, pw.pw_uid, pw.pw_gid)
        os.replace(tmp, t)
    except BaseException:
        os.unlink(tmp)
        raise
    changed = sorted(k for k in want if cur.get(k) != new.get(k))
    print(f"[install] {t}: set {', '.join(changed) or 'nothing new'}; other keys kept")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

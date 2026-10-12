"""7. The research loop around the judge.

(a) The driver (research/loop/run.sh) with an agent that crashes every time
    (prints "boom", exits 1), just after a 24 h run has started. It must not
    give up the run: back off and keep trying, or pause with an alert. It is
    stopped after CAP_S seconds either way. The driver reaches the agent
    through the default adapter (research/adapters/claude/session.sh). The
    real CLI is never reachable: the stub is IJ_AGENT_BIN, first on PATH, and
    PATH is otherwise /usr/bin:/bin plus a jq link.
(b) A queued experiment whose only patch does not apply to the engine source.
    The real runner (research/loop/runner.sh --once) must find that out in
    its pre-flight and never start the ruler for it. The ruler is a stub that
    records being called. The control is the same experiment with a patch
    that applies, which must reach the ruler.
"""
from __future__ import annotations

import json
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from core import END_TO_END, Outcome, python
from core import run as sh

CAP_S = 25
STUB = """#!/bin/bash
/usr/bin/python3 -c 'import json,sys,time; open(sys.argv[1],"a").write(json.dumps({"t": time.time(), "argv": sys.argv[2:]})+"\\n")' %s "$@"
echo "boom" >&2
exit 1
"""


def crash_loop(judge, ctx) -> Outcome:
    driver = judge.script("research/loop/run.sh")
    tmp = Path(tempfile.mkdtemp(prefix="crash-", dir=ctx.work))
    research = tmp / "research"
    shutil.copytree(driver.parent.parent, research,
                    ignore=shutil.ignore_patterns("experiments", "__pycache__", ".git"))
    (research / "experiments").mkdir()
    st = research / "loop" / "state.json"
    state = json.loads(st.read_text()) if st.exists() else {}
    now = int(time.time())
    state.update({"clock_start_epoch": now,
                  "clock_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))})
    st.write_text(json.dumps(state, indent=2))
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=s", "-c", "user.email=s@s", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=research, capture_output=True)
    stub_dir = tmp / "bin"
    stub_dir.mkdir()
    calls = tmp / "agent-calls.jsonl"
    stub = stub_dir / "claude"
    stub.write_text(STUB % calls)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    jq = shutil.which("jq")
    if jq:
        (stub_dir / "jq").symlink_to(jq)
    state_dir, results = tmp / "research-state", tmp / "results"
    results.mkdir()
    (tmp / "home").mkdir()
    env = {"PATH": f"{stub_dir}:/usr/bin:/bin", "HOME": str(tmp / "home"), "LANG": "C",
           "TMPDIR": str(tmp), "IJ_AGENT": "claude", "IJ_AGENT_BIN": str(stub), "IJ_PY": python(),
           "IJ_RESULTS_DIR": str(results), "IJ_DRIVER_LOG": str(tmp / "driver.log"),
           "IJ_STATE_DIR": str(state_dir), "IJ_RESEARCH_DIR": str(research),
           "IJ_ALERT": str(tmp / "alert.txt"),
           # short waits so the policy is visible inside CAP_S; the policy is the code's own
           "IJ_ITER_TIMEOUT": "60", "IJ_ITER_TIMEOUT_S": "60",
           "IJ_BACKOFF_BASE_S": "1", "IJ_POLL_S": "1", "IJ_TIMER_S": "1",
           "IJ_RATE_LIMIT_SLEEP": "1", "IJ_NOPROGRESS_SLEEP": "1",
           "IJ_NOPROGRESS_SLEEP_LONG": "1", "IJ_GIT_COMMIT": "0"}
    t0 = time.time()
    p = sh(["/bin/bash", str(research / "loop" / "run.sh")], timeout=CAP_S, env=env,
           cwd=research)
    took = time.time() - t0
    cs = [json.loads(x) for x in calls.read_text().splitlines()] if calls.exists() else []
    models = sorted({c["argv"][c["argv"].index("--model") + 1] for c in cs
                     if "--model" in c["argv"][:-1]})
    alerts = [str(f.relative_to(tmp)) for f in
              (research / "ALERTS.md", state_dir / "ALERTS.md", tmp / "alert.txt") if f.exists()]
    alive = p.returncode == 124
    obs = ((f"still running after {CAP_S} s" if alive
            else f"exited rc={p.returncode} after {took:.0f} s")
           + f"; {len(cs)} agent starts"
           + ("; ALERT raised" if alerts else ""))
    return Outcome("7a", "the agent process crashing on every start of a 24 h run",
                   "keep the run alive (back off, or raise alert)", obs, alive and (len(cs) > 3 or bool(alerts)), END_TO_END,
                   {"rc": p.returncode, "calls": len(cs), "models": models, "alerts": alerts,
                    "cap_s": CAP_S, "stderr": p.stderr.strip().splitlines()[-3:],
                    "stdout": p.stdout.strip().splitlines()[-4:]})


GOOD = "line one\nline two\nline three\n"
PATCH = """--- a/engine.py
+++ b/engine.py
@@ -1,3 +1,3 @@
 %s
-line two
+line 2
 line three
"""
LAUNCH = """#!/usr/bin/env bash
set -euo pipefail
cd /src/engine && git apply /exp/patches/fix.patch
exec python3 -m engine.serve --port 8000
"""


def _admit(judge, tmp: Path, engine: Path, exp_id: str, context_line: str) -> dict:
    exp_dir = tmp / exp_id
    (exp_dir / "patches").mkdir(parents=True)
    (exp_dir / "patches" / "fix.patch").write_text(PATCH % context_line)
    (exp_dir / "launch.sh").write_text(LAUNCH)
    return judge.admit_experiment(exp_dir, exp_id, engine)


def broken_patch(judge, ctx) -> Outcome:
    tmp = Path(tempfile.mkdtemp(prefix="patch-", dir=ctx.work))
    engine = tmp / "engine-src"
    engine.mkdir()
    (engine / "engine.py").write_text(GOOD)
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=s", "-c", "user.email=s@s", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=engine, capture_output=True)
    r = _admit(judge, tmp, engine, "exp-broken-patch", "this line is not in the file")
    control = _admit(judge, tmp, engine, "exp-good-patch", "line one")
    obs = (("judge STARTED" if r["slot_spent"] else "judge never started")
           + f"; {r['verdict']}")
    obs += ("; good patch: judge ran" if control["slot_spent"] else
            f"; but the control (good patch) never ran: {control['verdict']}")
    return Outcome("7b", "a queued experiment whose patch does not apply to the engine",
                   "reject it before the judge starts", obs,
                   (not r["slot_spent"]) and r["verdict"] == "PREFLIGHT_FAILED"
                   and control["slot_spent"], END_TO_END,
                   {"broken_patch": r, "control_good_patch": control})


def run(judge, ctx) -> list:
    return [crash_loop(judge, ctx), broken_patch(judge, ctx)]

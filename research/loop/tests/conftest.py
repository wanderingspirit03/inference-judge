"""Shared fixtures for the loop tests.

The two true externals are replaced by stubs on the same seams production
uses:

  IJ_BENCH_CMD -> fake-run.sh: records its argv, writes results.json
  the agent    -> `claude`, `codex` and `pi` stubs, first on PATH, that print
                  what each tool documents; the real adapters
                  (research/adapters/<name>/session.sh) run them

No test calls a real agent or the real /opt/bench.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

LOOP = Path(__file__).resolve().parent.parent
ADAPTERS = LOOP.parent / "adapters"
GUARD = LOOP.parent / "guard"
sys.path.insert(0, str(LOOP))

FAKE_BENCH = r"""#!/usr/bin/env bash
# Stub of /opt/bench/run.sh.  Records argv NUL-separated, one file per call.
set -u
n=$(ls "$FAKE_CALLS" 2>/dev/null | wc -l | tr -d ' ')
printf '%s\0' "$@" > "$FAKE_CALLS/$n.argv"
id="${!#}"
[ "${FAKE_BENCH_NO_RESULT:-0}" = 1 ] && { echo "bench: no result"; exit 3; }
mkdir -p "$FAKE_RESULTS/$id"
v="${FAKE_VERDICT:-REJECTED}"
[ "${1:-}" = "--instrument" ] && v=INSTRUMENT
printf '{"id":"%s","run_id":"%s-r1","verdict":"%s","duration_s":12,"stats":{"delta":0.01,"mde":0.07,"sigma_L":0.03},"reasons":["stub"]}\n' \
  "$id" "$id" "$v" > "$FAKE_RESULTS/$id/results.json"
echo "bench: $v"
"""

FAKE_AGENT_HEAD = r'''#!/usr/bin/env python3
"""Stub of an agent CLI.  Logs each session it is asked to run; behaviour from
$FAKE_AGENT_SCRIPT, one outcome per line, consumed in order (the last line
repeats)."""
import json, os, sys, time
argv = sys.argv[1:]
LOGGED_IN = os.environ.get("FAKE_AGENT_LOGIN", "1") == "1"
def session(**more):
    """Log the call, run $FAKE_AGENT_HOOK, return the scripted outcome."""
    with open(os.environ["FAKE_AGENT_LOG"], "a") as f:
        env = {k: v for k, v in os.environ.items() if k.startswith(("IJ_", "CLAUDE_"))}
        f.write(json.dumps({"argv": argv, "cwd": os.getcwd(), "t": time.time(), "pid": os.getpid(),
                            "env": env, **more}) + "\n")
    script = os.environ.get("FAKE_AGENT_SCRIPT", "")
    lines = open(script).read().split() if script and os.path.exists(script) else []
    if len(lines) > 1:
        open(script, "w").write("\n".join(lines[1:]) + "\n")
    hook = os.environ.get("FAKE_AGENT_HOOK")
    if hook:
        os.system(hook)
    return lines[0] if lines else "ok"
def ev(**k):
    print(json.dumps(k), flush=True)
'''

FAKE_CLAUDE = FAKE_AGENT_HEAD + r'''
if argv[:1] == ["--version"]:
    print("0.0.0 (stub claude)"); sys.exit(0)
if argv[:2] == ["auth", "status"]:
    if os.environ.get("FAKE_CLAUDE_OLD"):      # a Claude Code from before `claude auth`
        sys.stderr.write("error: unknown command 'auth'\n"); sys.exit(1)
    print(json.dumps({"loggedIn": LOGGED_IN, "authMethod": "claude.ai" if LOGGED_IN else "none"}))
    sys.exit(0 if LOGGED_IN else 1)
mode = session()
if mode == "ok":
    ev(type="result", is_error=False, result="done", num_turns=3, total_cost_usd=0.5,
       usage={"input_tokens": 10, "output_tokens": 5})
elif mode == "silent":
    sys.exit(0)
elif mode == "crash":
    sys.stderr.write("Segmentation fault\n"); sys.exit(1)
elif mode == "garbage":           # not the documented output at all
    print("error: unknown option '--output-format'"); sys.exit(2)
elif mode == "softfail":          # reports a failure and still exits 0
    ev(type="result", is_error=True, subtype="error_during_execution", result="Edit tool failed: file not found")
elif mode == "limit":
    ev(type="result", is_error=True, result="Claude usage limit reached. Try again later."); sys.exit(1)
elif mode == "auth":
    ev(type="result", is_error=True, result="OAuth token has expired. Please run /login. Try again after logging in.")
    sys.exit(1)
elif mode.startswith("sleep"):
    time.sleep(float(mode[5:] or 5))
    ev(type="result", is_error=False, result="slept")
'''

# The events `codex exec --json` documents: thread.started, turn.started,
# item.started / item.completed, then turn.completed (with usage) or turn.failed.
FAKE_CODEX = FAKE_AGENT_HEAD + r'''
if argv[:1] == ["--version"]:
    print("codex-cli 0.0.0 (stub)"); sys.exit(0)
if argv[:2] == ["login", "status"]:
    print("Logged in" if LOGGED_IN else "Not logged in"); sys.exit(0 if LOGGED_IN else 1)
mode = session(prompt=sys.stdin.read() if argv[-1:] == ["-"] else "")
USAGE = {"input_tokens": 24763, "cached_input_tokens": 24448, "output_tokens": 122, "reasoning_output_tokens": 7}
if mode == "silent":
    sys.exit(0)
if mode == "crash":
    sys.stderr.write("thread 'main' panicked\n"); sys.exit(101)
if mode == "garbage":             # not the documented output at all
    print("error: unexpected argument '--json' found"); sys.exit(2)
ev(type="thread.started", thread_id="0199a213-81c0-7800-8aa1-bbab2a035a53")
ev(type="turn.started")
if mode in ("ok", "retry"):
    if mode == "retry":
        ev(type="error", message="Reconnecting... 1/5")
    ev(type="item.started", item={"id": "item_1", "type": "command_execution", "command": "bash -lc ls", "status": "in_progress"})
    ev(type="item.completed", item={"id": "item_2", "type": "agent_message", "text": "first note"})
    ev(type="item.completed", item={"id": "item_3", "type": "agent_message", "text": "queued one experiment"})
    ev(type="turn.completed", usage=USAGE)
elif mode in ("limit", "auth", "fail"):
    why = {"limit": "You've hit your usage limit. Try again in 3 hours 12 minutes.",
           "auth": "unexpected status 401 Unauthorized", "fail": "stream disconnected before completion"}[mode]
    ev(type="error", message=why)
    ev(type="turn.failed", error={"message": why}); sys.exit(1)
elif mode == "softfail":          # reports a failure and still exits 0
    ev(type="turn.failed", error={"message": "stream disconnected before completion"})
elif mode.startswith("sleep"):
    time.sleep(float(mode[5:] or 5))
    ev(type="item.completed", item={"id": "item_1", "type": "agent_message", "text": "slept"})
    ev(type="turn.completed", usage=USAGE)
'''

# The events `pi --mode json` documents: a session header, then agent, turn and
# message events; an assistant message_end carries the text, usage and stopReason.
FAKE_PI = FAKE_AGENT_HEAD + r'''
if argv[:1] == ["--version"]:
    print("0.0.0 (stub pi)"); sys.exit(0)
if argv[:2] == ["auth", "check"]:
    print("ready" if LOGGED_IN else "not_ready"); sys.exit(0 if LOGGED_IN else 1)
mode = session(prompt=argv[-1] if argv else "", stdin=sys.stdin.read())
def usage(i, o, cost):
    return {"input": i, "output": o, "cacheRead": 5, "cacheWrite": 2, "totalTokens": i + o + 7,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": cost}}
def assistant(content, stop, **extra):
    m = {"role": "assistant", "content": content, "api": "x", "provider": "p", "model": "m",
         "usage": usage(100, 20, 0.25), "stopReason": stop, "timestamp": 1733234401000, **extra}
    ev(type="message_end", message=m)
    ev(type="turn_end", message=m, toolResults=[])
if mode == "silent":
    sys.exit(0)
if mode == "crash":
    sys.stderr.write("TypeError: Cannot read properties of undefined\n"); sys.exit(1)
if mode == "garbage":
    print("Unknown option: --mode"); sys.exit(2)
ev(type="session", version=3, id="5f0c7a52-0000-4000-8000-000000000001", timestamp="2026-01-01T00:00:00.000Z", cwd=os.getcwd())
ev(type="agent_start"); ev(type="turn_start")
ev(type="message_end", message={"role": "user", "content": "the prompt", "timestamp": 1733234401000})
if mode in ("ok", "retry"):
    if mode == "retry":
        ev(type="auto_retry_start", attempt=1, maxAttempts=3, delayMs=2000, errorMessage="overloaded")
    assistant([{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls"}}], "toolUse")
    ev(type="turn_start")
    ev(type="message_update", assistantMessageEvent={"type": "text_delta", "contentIndex": 0, "delta": "queued"})
    assistant([{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "queued one "},
               {"type": "text", "text": "experiment"}], "stop")
    ev(type="agent_end", messages=[], willRetry=False); ev(type="agent_settled")
elif mode in ("limit", "auth", "fail"):
    why = {"limit": "429 You exceeded your current quota, rate limit reached",
           "auth": "401 Unauthorized: invalid api key", "fail": "terminated"}[mode]
    assistant([], "error", errorMessage=why)
    ev(type="agent_end", messages=[], willRetry=False); ev(type="agent_settled")
    sys.exit(1)
elif mode == "softfail":
    assistant([], "error", errorMessage="terminated")
    ev(type="agent_end", messages=[], willRetry=False); ev(type="agent_settled")
elif mode.startswith("sleep"):
    time.sleep(float(mode[5:] or 5))
    assistant([{"type": "text", "text": "slept"}], "stop")
    ev(type="agent_settled")
'''

VALID_LAUNCH = "#!/usr/bin/env bash\nset -euo pipefail\necho start\n"


def _x(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class Lab:
    """A throwaway research repo + state dir + results dir + stubs."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.research = tmp / "research"
        self.queue = self.research / "queue"
        self.experiments = self.research / "experiments"
        self.state = tmp / "state"
        self.results = tmp / "results"
        self.calls = tmp / "bench_calls"
        self.bin = tmp / "bin"
        self.spool = tmp / "jobs"
        self.frozen = tmp / "frozen"
        for d in (self.queue, self.experiments, self.state, self.results, self.calls,
                  self.bin, self.spool, self.frozen):
            d.mkdir(parents=True, exist_ok=True)
        (self.frozen / "hook.sh").write_text("echo frozen\n")
        self.hypotheses = self.research / "HYPOTHESES.jsonl"
        self.hypotheses.write_text("")
        self.fake_bench = _x(self.bin / "fake-run.sh", FAKE_BENCH)
        for name, stub in (("claude", FAKE_CLAUDE), ("codex", FAKE_CODEX), ("pi", FAKE_PI)):
            _x(self.bin / name, stub)
        self.agent_log = tmp / "agent_calls.jsonl"
        self.agent_script = tmp / "agent_script.txt"
        self.alerts = tmp / "alerts.log"
        self.legacy = tmp / "legacy_state.json"

    # -- environment ------------------------------------------------------
    def env(self, **extra) -> dict:
        e = dict(os.environ)
        for k in list(e):
            if k.startswith("IJ_"):
                del e[k]
        e.update({
            "PATH": f"{self.bin}:{e.get('PATH', '/usr/bin:/bin')}",
            "IJ_PY": sys.executable,
            "IJ_STATE_DIR": str(self.state),
            "IJ_RESEARCH_DIR": str(self.research),
            "IJ_RESULTS_DIR": str(self.results),
            "IJ_BENCH_CMD": str(self.fake_bench),
            "IJ_RESEARCHER_SUDO": "",
            "IJ_JOBS_SPOOL": str(self.spool),
            "IJ_FROZEN_PATHS": str(self.frozen),
            "IJ_LEGACY_STATE": str(self.legacy),
            "IJ_ALERT": str(self.alerts),
            "IJ_PATCH_TREES": str(self.tmp / "no-trees"),
            "IJ_GIT_COMMIT": "0",
            "IJ_POLL_S": "0.02",
            "IJ_MIN_GAP_S": "0",
            "IJ_BACKOFF_BASE_S": "0",
            "IJ_RUNNER_POLL_S": "0.05",
            "FAKE_CALLS": str(self.calls),
            "FAKE_RESULTS": str(self.results),
            "FAKE_AGENT_LOG": str(self.agent_log),
            "FAKE_AGENT_SCRIPT": str(self.agent_script),
        })
        e.update({k: str(v) for k, v in extra.items()})
        return e

    # -- research-side helpers -------------------------------------------
    def experiment(self, exp_id: str, launch: str = VALID_LAUNCH) -> Path:
        d = self.experiments / exp_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "launch.sh").write_text(launch)
        return d

    def predict(self, exp_id: str, prediction=-5.0, **kw) -> None:
        with open(self.hypotheses, "a") as f:
            f.write(json.dumps({"id": f"H-{exp_id}", "title": f"t {exp_id}", "experiment": exp_id,
                                "prediction": prediction, "status": "open", **kw}) + "\n")

    def enqueue(self, exp_id: str, kind: str = "candidate", name: str | None = None, **extra) -> Path:
        p = self.queue / (name or f"{exp_id}.json")
        p.write_text(json.dumps({"id": exp_id, "kind": kind, **extra}))
        return p

    def ready(self, exp_id: str, kind: str = "candidate") -> Path:
        """An experiment that passes pre-flight and the queue checks."""
        self.experiment(exp_id)
        if kind == "candidate":
            self.predict(exp_id)
        return self.enqueue(exp_id, kind)

    # -- observation helpers ---------------------------------------------
    def bench_calls(self) -> list[list[str]]:
        out = []
        for f in sorted(self.calls.glob("*.argv"), key=lambda p: int(p.stem)):
            out.append([a for a in f.read_bytes().decode().split("\0") if a != ""])
        return out

    def agent_calls(self) -> list[dict]:
        """One entry per session the agent stub was asked to run."""
        if not self.agent_log.exists():
            return []
        return [json.loads(l) for l in self.agent_log.read_text().splitlines() if l.strip()]

    def events(self) -> list[dict]:
        import loopstate
        return loopstate.load_events(self.state / "events.jsonl")

    def state_now(self):
        import loopstate
        return loopstate.replay(self.events())

    def script(self, *modes: str) -> None:
        self.agent_script.write_text("\n".join(modes) + "\n")

    # -- running the real scripts ----------------------------------------
    def run(self, argv: list[str], timeout: float = 60, **env) -> subprocess.CompletedProcess:
        return subprocess.run(argv, env=self.env(**env), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)

    def lstate(self, *args: str, **env) -> subprocess.CompletedProcess:
        return self.run([sys.executable, "-I", str(LOOP / "loopstate.py"), *args], **env)

    def runner_once(self, **env) -> subprocess.CompletedProcess:
        return self.run([str(LOOP / "runner.sh"), "--once"], **env)

    def driver(self, *args: str, timeout: float = 60, **env) -> subprocess.CompletedProcess:
        return self.run(["bash", str(LOOP / "run.sh"), *args], timeout=timeout, **env)

    def start_clock(self, hours: float = 24.0, **kw) -> None:
        args = ["start-clock", "--accel-hours", str(hours)]
        if "deadline" in kw:
            args += ["--deadline", str(kw["deadline"])]
        else:
            args += ["--wall-hours", str(hours)]
        r = self.lstate(*args)
        assert r.returncode == 0, r.stderr


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    return Lab(tmp_path)


@pytest.fixture
def T0() -> float:
    return 1_800_000_000.0   # a fixed "now" for pure-function tests


def pid_alive(pid: int) -> bool:
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != "" and not r.stdout.strip().startswith("Z")


def wait_until(pred, timeout: float = 10.0, step: float = 0.05) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return pred()

"""loop/run.sh end to end with stub agents on PATH (conftest.py), reached
through the real adapters.

The driver must never exit on crashes: back off, fall back to the second
model, then pause and ALERT while the ruler keeps measuring.  Auth failures
pause at once; quota limits are never crashes.  It exits 10 when the budget
is spent, after reaping the researcher's detached jobs."""
from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import threading
import time

import pytest

from conftest import ADAPTERS, LOOP, Lab, pid_alive, wait_until


def sessions(lab: Lab) -> list[dict]:
    return [e for e in lab.events() if e["type"] == "session_end"]


def models(lab: Lab) -> list[str]:
    """The model each claude session was started with."""
    out = []
    for c in lab.agent_calls():
        a = c["argv"]
        out.append(a[a.index("--model") + 1])
    return out


@pytest.fixture
def clocked(lab: Lab) -> Lab:
    lab.start_clock(24)
    return lab


def test_session_prompt_carries_the_brief(clocked: Lab):
    clocked.script("ok")
    r = clocked.driver("--once")
    assert r.returncode == 0, r.stdout + r.stderr
    [call] = clocked.agent_calls()
    prompt = call["argv"][call["argv"].index("-p") + 1]
    assert f"Read {clocked.state / 'BRIEF.md'} first" in prompt and "AGENTS.md" in prompt
    assert "UTC now: 20" in prompt and "Session trigger: first" in prompt
    assert call["cwd"] == str(clocked.research.resolve()) or call["cwd"] == str(clocked.research)
    assert "--model" in call["argv"] and "opus" in call["argv"]
    brief = (clocked.state / "BRIEF.md").read_text()
    assert "UTC now:" in brief and "Queue (" in brief
    [s] = sessions(clocked)
    assert s["class"] == "ok" and s["usage"]["num_turns"] == 3
    assert (clocked.state / "process.jsonl").exists()


def test_three_crashes_fall_back_then_pause_and_alert_never_exit_1(clocked: Lab):
    clocked.script("crash")
    r = clocked.driver("--max-loops", "12", IJ_FALLBACK_SESSIONS=2)
    assert r.returncode == 0, r.stdout + r.stderr          # an earlier driver exited 1 here
    assert models(clocked) == ["opus"] * 3 + ["sonnet"] * 3
    assert [s["class"] for s in sessions(clocked)] == ["crash"] * 6
    st = clocked.state_now()
    assert st.paused["reason"] == "crashes"
    alerts = clocked.alerts.read_text()
    assert "running the next 2 on the fallback model" in alerts and "paused" in alerts
    # the operator resumes; sessions start again on the primary model
    clocked.script("ok")
    assert clocked.driver("--resume").returncode == 0
    clocked.driver("--once")
    assert models(clocked)[-1] == "opus"


def test_resume_file_clears_a_pause(clocked: Lab):
    clocked.script("crash")
    clocked.driver("--max-loops", "10", IJ_FALLBACK_SESSIONS=0)
    assert clocked.state_now().paused["reason"] == "crashes"
    (clocked.state / "RESUME").touch()
    clocked.script("ok")
    clocked.driver("--once")
    assert clocked.state_now().paused is None and not (clocked.state / "RESUME").exists()


def test_auth_failure_pauses_without_counting_a_crash(clocked: Lab):
    clocked.script("auth")
    r = clocked.driver("--max-loops", "6")
    assert r.returncode == 0
    assert len(clocked.agent_calls()) == 1
    st = clocked.state_now()
    assert st.paused["reason"] == "auth" and st.crash_streak == 0
    alerts = clocked.alerts.read_text()
    assert "authentication failed" in alerts and "Claude" not in alerts


def test_auth_probe_unpauses_when_login_is_fixed(clocked: Lab):
    clocked.script("auth", "ok")
    clocked.driver("--max-loops", "8", IJ_AUTH_PROBE_S=0.1)
    assert [s["class"] for s in sessions(clocked)][:2] == ["auth", "ok"]
    assert clocked.state_now().paused is None


def test_quota_limit_is_not_a_crash(clocked: Lab):
    clocked.script("limit")
    r = clocked.driver("--max-loops", "5", IJ_RATE_LIMIT_SLEEP=60)
    assert r.returncode == 0
    assert len(clocked.agent_calls()) == 1
    [s] = sessions(clocked)
    assert s["class"] == "limit" and s["sleep_s"] >= 60
    assert clocked.state_now().crash_streak == 0


def test_session_timeout_is_capped_and_not_a_crash(clocked: Lab):
    clocked.script("sleep30")
    t0 = time.time()
    clocked.driver("--once", IJ_ITER_TIMEOUT_S=1, IJ_KILL_GRACE=1, timeout=30)
    assert time.time() - t0 < 20
    [s] = sessions(clocked)
    assert s["class"] == "timeout" and clocked.state_now().crash_streak == 0


OTHER_AGENTS = ["codex", "pi"]


@pytest.mark.parametrize("agent", OTHER_AGENTS)
def test_the_driver_runs_a_session_through_another_agent(clocked: Lab, agent):
    clocked.script("ok")
    r = clocked.driver("--once", IJ_AGENT=agent)
    assert r.returncode == 0, r.stdout + r.stderr
    [call] = clocked.agent_calls()
    assert f"Read {clocked.state / 'BRIEF.md'} first" in call["prompt"]
    assert call["cwd"] in (str(clocked.research), str(clocked.research.resolve()))
    [s] = sessions(clocked)
    assert s["class"] == "ok" and s["usage"]["num_turns"] >= 1 and s["usage"]["output_tokens"] > 0
    [result] = (clocked.state / "iterations").glob("*.json")
    out = json.loads(result.read_text())
    assert out["agent"] == agent and out["result"] == "queued one experiment"


@pytest.mark.parametrize("agent", OTHER_AGENTS)
@pytest.mark.parametrize("mode, cls", [("limit", "limit"), ("auth", "auth"), ("fail", "crash"),
                                       ("softfail", "crash"), ("crash", "crash"), ("garbage", "crash")])
def test_the_driver_tells_how_another_agent_s_session_ended(clocked: Lab, agent, mode, cls):
    clocked.script(mode, "ok")
    clocked.driver("--max-loops", "1", IJ_AGENT=agent, IJ_RATE_LIMIT_SLEEP=60)
    assert [s["class"] for s in sessions(clocked)] == [cls]


@pytest.mark.parametrize("agent", OTHER_AGENTS)
def test_a_logged_out_agent_pauses_the_loop_whatever_its_error_says(clocked: Lab, agent):
    clocked.script("fail")
    clocked.driver("--max-loops", "3", IJ_AGENT=agent, IJ_PRIMARY_MODEL="vendor/some-model", FAKE_AGENT_LOGIN=0)
    assert [s["class"] for s in sessions(clocked)] == ["auth"]
    assert clocked.state_now().paused["reason"] == "auth"


@pytest.mark.parametrize("agent", OTHER_AGENTS)
def test_another_agent_s_session_is_capped_too(clocked: Lab, agent):
    clocked.script("sleep30")
    t0 = time.time()
    clocked.driver("--once", IJ_AGENT=agent, IJ_ITER_TIMEOUT_S=1, IJ_KILL_GRACE=1, timeout=30)
    assert time.time() - t0 < 20
    [s] = sessions(clocked)
    assert s["class"] == "timeout"
    assert wait_until(lambda: not pid_alive(clocked.agent_calls()[0]["pid"]))


def test_an_agent_without_an_adapter_stops_the_driver_at_once(clocked: Lab):
    for args in (["--once"], ["--dry-run"]):
        r = clocked.driver(*args, IJ_AGENT="nosuchagent")
        assert r.returncode != 0 and "nosuchagent" in r.stderr and "codex" in r.stderr
    assert clocked.agent_calls() == [] and sessions(clocked) == []


def test_the_agent_s_command_can_be_named(clocked: Lab):
    """IJ_AGENT_BIN: the agent's command when it is not on the researcher's PATH."""
    other = clocked.bin / "claude-elsewhere"
    (clocked.bin / "claude").rename(other)
    clocked.script("ok")
    r = clocked.driver("--once", IJ_AGENT_BIN=other)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(clocked.agent_calls()) == 1 and sessions(clocked)[0]["class"] == "ok"


def test_the_literature_pass_goes_through_the_adapter_too(lab: Lab):
    """literature.sh works on the repo it sits in (and commits there), so this runs a copy."""
    shutil.copytree(LOOP, lab.research / "loop", ignore=shutil.ignore_patterns("tests", "__pycache__", "*.log"))
    shutil.copytree(ADAPTERS, lab.research / "adapters")
    subprocess.run(["git", "init", "-q"], cwd=lab.research, check=True)
    script = lab.research / "loop" / "literature.sh"
    dry = lab.run(["bash", str(script), "--dry-run"])
    assert dry.returncode == 0 and "AGENTS.md" in dry.stdout
    assert "adapters/claude/session.sh run --model primary --effort high" in dry.stdout
    assert lab.agent_calls() == []
    for agent in ("claude", "codex"):
        r = lab.run(["bash", str(script)], IJ_AGENT=agent)
        assert r.returncode == 0, r.stdout + r.stderr
    claude, codex = lab.agent_calls()
    assert "literature pass" in claude["argv"][claude["argv"].index("-p") + 1]
    assert "literature pass" in codex["prompt"]
    assert claude["cwd"] in (str(lab.research), str(lab.research.resolve()))
    lab.script("limit", "ok")                       # one quota limit: wait, then try once more
    r = lab.run(["bash", str(script)], IJ_AGENT="pi", IJ_RATE_LIMIT_SLEEP=0)
    assert r.returncode == 0 and len(lab.agent_calls()) == 4
    assert lab.run(["bash", str(script)], IJ_AGENT="nosuchagent").returncode != 0


def test_finish_reaps_jobs_writes_summary_stub_and_exits_10(lab: Lab):
    r = lab.run(["bash", str(LOOP / "bg.sh"), "sweep", "sleep", "300"])
    assert r.returncode == 0, r.stderr
    pid = int(r.stdout.split()[0])
    assert pid_alive(pid)
    rec = json.loads((lab.spool / f"{pid}.json").read_text())
    assert rec["pgid"] == pid and rec["name"] == "sweep"
    lab.start_clock(24, deadline=time.time() + 3)
    lab.script("ok")
    r = lab.driver(timeout=60)
    assert r.returncode == 10, r.stdout + r.stderr
    assert wait_until(lambda: not pid_alive(pid))
    assert (lab.state / "SUMMARY.driver.md").exists()
    ev = lab.events()
    assert any(e["type"] == "job" and e["pgid"] == pid for e in ev)
    assert any(e["type"] == "job_reaped" and e["pgid"] == pid for e in ev)
    assert ev[-1]["type"] in ("finish", "alert") and lab.state_now().finished == "wall deadline reached"
    assert "run finished" in lab.alerts.read_text()
    # a restarted driver (systemd) stops again at once
    assert lab.driver(timeout=30).returncode == 10


def test_frozen_files_changing_pauses_sessions(clocked: Lab):
    clocked.script("ok")
    clocked.driver("--once")
    (clocked.frozen / "hook.sh").write_text("exit 0  # disabled\n")
    r = clocked.lstate("next")
    assert r.stdout.startswith("sleep") and "paused" in r.stdout
    assert "frozen files" in clocked.alerts.read_text()
    assert clocked.driver("--resume").returncode == 0
    assert clocked.state_now().paused is None


def test_clock_is_seeded_once_from_bootstraps_state_json(lab: Lab):
    t0 = int(time.time()) - 100
    lab.legacy.write_text(json.dumps({"clock_start_epoch": t0, "budget_hours": 24}))
    r = lab.lstate("next")
    assert r.stdout.startswith("session"), r.stdout + r.stderr
    st = lab.state_now()
    assert st.clock_start == t0 and st.wall_deadline == t0 + 24 * 3600
    lab.lstate("next")
    assert len([e for e in lab.events() if e["type"] == "run_config"]) == 1


def test_no_clock_no_session(lab: Lab):
    r = lab.lstate("next")
    assert r.stdout.startswith("sleep") and "no clock" in r.stdout


def test_dry_run_exits_0_and_touches_nothing(clocked: Lab):
    before = (clocked.state / "events.jsonl").read_text()
    r = clocked.driver("--dry-run", PATH="/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WRONG" not in r.stdout and "classifier" in r.stdout
    assert "expired login" in r.stdout and "ok (auth)" in r.stdout
    assert "--- BRIEF.md" in r.stdout
    assert "adapters/claude/session.sh run --model primary --effort high" in r.stdout
    assert (clocked.state / "events.jsonl").read_text() == before
    assert clocked.agent_calls() == []


def test_dry_run_shows_the_model_this_environment_names(clocked: Lab):
    """The policy in force is the one the last driver start logged.  A dry run
    previews a driver started now, so a model named in its environment shows,
    and one that is not named stays as the last start logged it."""
    path = "/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin"
    r = clocked.driver("--dry-run", PATH=path, IJ_PRIMARY_MODEL="vendor/some-model")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "session.sh run --model vendor/some-model --effort high" in r.stdout
    assert clocked.lstate("start", IJ_PRIMARY_MODEL="logged/model").returncode == 0
    r = clocked.driver("--dry-run", PATH=path)
    assert "session.sh run --model logged/model --effort high" in r.stdout, r.stdout


def test_alert_sinks(lab: Lab):
    import loopstate as ls
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.path, self.headers.get("Title"),
                        self.rfile.read(int(self.headers["Content-Length"])).decode()))
            self.send_response(200); self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/inference-judge-topic"
    cfg = ls.Config.from_env(lab.env(IJ_ALERT=url))
    ls.send_alert(cfg, "driver paused", now=1.0)
    assert got == [("/inference-judge-topic", "inference-judge alert", "driver paused")]
    cfg = ls.Config.from_env(lab.env(IJ_ALERT=f"file:{lab.tmp / 'a.txt'}"))
    ls.send_alert(cfg, "second", level="warn", now=2.0)
    assert "[warn] second" in (lab.tmp / "a.txt").read_text()
    cfg = ls.Config.from_env(lab.env(IJ_ALERT="http://127.0.0.1:9/unreachable"))
    ls.send_alert(cfg, "sink down", now=3.0)                # must not raise
    assert "sink down" in (lab.state / "ALERTS.md").read_text()


def test_heartbeat_writes_status_and_alerts_once_on_wedge(clocked: Lab):
    clocked.script("ok")
    clocked.driver("--once")
    hb = clocked.run(["bash", str(LOOP / "heartbeat.sh")])
    assert hb.returncode == 0, hb.stderr
    doc = json.loads((clocked.state / "heartbeat.json").read_text())
    assert {"driver_alive", "runner", "accel_used_h", "end", "disk_pct"} <= set(doc)
    later = time.time() + 3600 + 1000
    for _ in range(3):
        clocked.run(["bash", str(LOOP / "heartbeat.sh")], IJ_NOW=later)
    assert clocked.alerts.read_text().count("logged nothing") == 1


def test_bg_sh_rejects_bad_names(lab: Lab):
    r = lab.run(["bash", str(LOOP / "bg.sh"), "../x", "true"])
    assert r.returncode == 2 and list(lab.spool.iterdir()) == []

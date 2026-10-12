"""The agent adapters (research/adapters): what each one sends to its agent and
what it hands back to the driver.

The agents are the stubs in conftest.py.  The codex and pi stubs print the
event streams those tools document; no test here starts a real agent, so a
passing run says the adapters follow the documentation, not that the tools do.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from conftest import ADAPTERS, GUARD, Lab, pid_alive, wait_until

AGENTS = ["claude", "codex", "pi"]
STREAMING = ["codex", "pi"]          # these two print events; the adapter reduces them
PROMPT = "UTC now: 2026-01-01. Read BRIEF.md first.\nSecond line with \"quotes\" and $dollars."


def session(lab: Lab, agent: str, *args: str, stdin: str | None = PROMPT, **env) -> subprocess.CompletedProcess:
    return subprocess.run([str(ADAPTERS / agent / "session.sh"), *args], env=lab.env(**env), cwd=lab.research,
                          input=stdin, capture_output=True, text=True, timeout=60)


def run(lab: Lab, agent: str, model: str = "primary", **env) -> tuple[subprocess.CompletedProcess, dict]:
    r = session(lab, agent, "run", "--model", model, "--effort", "high", **env)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(lines) == 1, f"one JSON object on stdout, got: {r.stdout!r} / {r.stderr!r}"
    return r, json.loads(lines[0])


# ------------------------------------------------------ what the agent is sent
def test_claude_gets_the_prompt_the_model_and_the_effort(lab: Lab):
    r, out = run(lab, "claude")
    assert r.returncode == 0 and out["is_error"] is False
    [call] = lab.agent_calls()
    assert call["argv"] == ["-p", PROMPT, "--model", "opus", "--effort", "high",
                            "--dangerously-skip-permissions", "--output-format", "json"]
    assert call["cwd"] in (str(lab.research), str(lab.research.resolve()))
    assert call["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == "high"


@pytest.mark.parametrize("asked, sent", [("primary", "opus"), ("fallback", "sonnet"), ("some-model", "some-model")])
def test_claude_maps_the_two_role_words_and_passes_any_other_name(lab: Lab, asked, sent):
    run(lab, "claude", model=asked)
    argv = lab.agent_calls()[-1]["argv"]
    assert argv[argv.index("--model") + 1] == sent


def test_codex_reads_the_prompt_on_stdin_and_prints_events(lab: Lab):
    run(lab, "codex")
    [call] = lab.agent_calls()
    assert call["argv"] == ["exec", "--json", "--skip-git-repo-check",
                            "--dangerously-bypass-approvals-and-sandbox", "--dangerously-bypass-hook-trust",
                            "-c", 'model_reasoning_effort="high"', "-"]
    assert call["prompt"].rstrip("\n") == PROMPT


def test_pi_gets_the_prompt_the_guard_and_json_mode(lab: Lab):
    run(lab, "pi")
    [call] = lab.agent_calls()
    assert call["argv"] == ["--mode", "json", "--no-approve", "-e", str(ADAPTERS / "pi" / "guard.ts"),
                            "--thinking", "high", PROMPT]
    assert call["stdin"] == ""                       # pi adds piped stdin to the prompt
    assert Path(call["env"]["IJ_GUARD_DIR"]).resolve() == GUARD.resolve()


@pytest.mark.parametrize("agent, flag", [("codex", "-m"), ("pi", "--model")])
def test_codex_and_pi_use_their_own_model_unless_the_operator_names_one(lab: Lab, agent, flag):
    for role in ("primary", "fallback"):
        run(lab, agent, model=role)
        assert flag not in lab.agent_calls()[-1]["argv"]
    run(lab, agent, model="vendor/some-model")
    argv = lab.agent_calls()[-1]["argv"]
    assert argv[argv.index(flag) + 1] == "vendor/some-model"


# --------------------------------------------------- what the driver gets back
@pytest.mark.parametrize("agent", STREAMING)
@pytest.mark.parametrize("mode", ["ok", "retry"])
def test_a_finished_session_gives_the_last_message(lab: Lab, agent, mode):
    lab.script(mode)                                  # retry: an error event, then success
    r, out = run(lab, agent)
    assert r.returncode == 0
    assert out["type"] == "result" and out["is_error"] is False
    assert out["result"] == "queued one experiment"
    assert out["session_id"]


@pytest.mark.parametrize("agent", STREAMING)
@pytest.mark.parametrize("mode, text", [("limit", "limit"), ("auth", "401"), ("fail", "")])
def test_a_failed_session_passes_the_agents_own_error_text(lab: Lab, agent, mode, text):
    lab.script(mode)
    r, out = run(lab, agent)
    assert r.returncode != 0
    assert out["is_error"] is True and out["result"] and text in out["result"]


@pytest.mark.parametrize("agent", AGENTS)
def test_a_crash_before_any_output_is_an_error_and_keeps_stderr(lab: Lab, agent):
    lab.script("crash")
    r, out = run(lab, agent)
    assert r.returncode != 0 and out["is_error"] is True
    assert "no result" in out["result"]
    assert r.stderr.strip()                           # the driver looks for limit and login words here


@pytest.mark.parametrize("agent", AGENTS)
def test_output_the_adapter_cannot_read_is_passed_on_as_the_error(lab: Lab, agent):
    lab.script("garbage")                             # e.g. a version of the tool that lost a flag
    r, out = run(lab, agent)
    assert r.returncode == 2 and out["is_error"] is True
    assert "no result" in out["result"]
    assert any(w in out["result"] for w in ("unknown option", "unexpected argument", "Unknown option"))


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("mode", ["softfail", "silent"])
def test_a_failure_never_exits_0(lab: Lab, agent, mode):
    """The driver counts an error result with exit status 0 as a finished session."""
    lab.script(mode)                                  # the agent exits 0 after a failed turn, or after printing nothing
    r, out = run(lab, agent)
    assert out["is_error"] is True and r.returncode != 0


def test_claude_code_s_own_result_is_passed_on_unchanged(lab: Lab):
    r, out = run(lab, "claude")
    assert out == {"type": "result", "is_error": False, "result": "done", "num_turns": 3,
                   "total_cost_usd": 0.5, "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_usage_and_turns_are_reported_in_the_names_the_driver_records(lab: Lab):
    _, codex = run(lab, "codex")
    assert codex["num_turns"] == 1
    assert codex["usage"] == {"input_tokens": 24763, "cached_input_tokens": 24448,
                              "output_tokens": 122, "reasoning_output_tokens": 7}
    _, pi = run(lab, "pi")
    assert pi["num_turns"] == 2 and pi["total_cost_usd"] == pytest.approx(0.5)
    assert pi["usage"] == {"input_tokens": 200, "output_tokens": 40,
                           "cache_read_input_tokens": 10, "cache_creation_input_tokens": 4}


@pytest.mark.parametrize("agent, model", [("codex", "primary"), ("pi", "vendor/some-model")])
def test_a_session_that_failed_while_logged_out_says_so(lab: Lab, agent, model):
    lab.script("fail")
    _, out = run(lab, agent, model=model, FAKE_AGENT_LOGIN=0)
    assert out["result"].startswith("not logged in: ")
    lab.script("fail")
    _, out = run(lab, agent, model=model, FAKE_AGENT_LOGIN=1)
    assert not out["result"].startswith("not logged in")


# ------------------------------------------------------------ the time limit
@pytest.mark.parametrize("agent", AGENTS)
def test_sigterm_reaches_the_agent(lab: Lab, agent):
    """The driver ends a long session with SIGTERM to session.sh (through sudo,
    which signals only its own child).  The agent must not be left running."""
    lab.script("sleep60")
    p = subprocess.Popen([str(ADAPTERS / agent / "session.sh"), "run", "--model", "primary", "--effort", "high"],
                         env=lab.env(), cwd=lab.research, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    p.stdin.write(PROMPT); p.stdin.close()
    assert wait_until(lambda: len(lab.agent_calls()) == 1)
    pid = lab.agent_calls()[0]["pid"]
    assert pid_alive(pid)
    os.kill(p.pid, signal.SIGTERM)
    t0 = time.time()
    assert wait_until(lambda: not pid_alive(pid), timeout=10)
    p.wait(timeout=10)
    assert time.time() - t0 < 10 and p.returncode != 0


# ------------------------------------------------------------- the other verbs
@pytest.mark.parametrize("agent", AGENTS)
def test_version(lab: Lab, agent):
    r = session(lab, agent, "version", stdin=None)
    assert r.returncode == 0 and "stub" in r.stdout


def test_auth_says_logged_in_logged_out_or_cannot_tell(lab: Lab, tmp_path):
    assert session(lab, "codex", "auth", stdin=None, FAKE_AGENT_LOGIN=1).returncode == 0
    assert session(lab, "codex", "auth", stdin=None, FAKE_AGENT_LOGIN=0).returncode == 1
    # pi can only be asked about a named model
    assert session(lab, "pi", "auth", stdin=None).returncode == 2
    assert session(lab, "pi", "auth", stdin=None, IJ_PRIMARY_MODEL="vendor/m", FAKE_AGENT_LOGIN=1).returncode == 0
    assert session(lab, "pi", "auth", stdin=None, IJ_PRIMARY_MODEL="vendor/m", FAKE_AGENT_LOGIN=0).returncode == 1
    # claude says whether it is logged in
    assert session(lab, "claude", "auth", stdin=None, FAKE_AGENT_LOGIN=1).returncode == 0
    assert session(lab, "claude", "auth", stdin=None, FAKE_AGENT_LOGIN=0).returncode == 1
    # an older claude cannot say: no settings in the home directory means never logged in; otherwise
    # only a session can tell
    home = tmp_path / "home"; home.mkdir()
    assert session(lab, "claude", "auth", stdin=None, HOME=home, FAKE_CLAUDE_OLD=1).returncode == 1
    (home / ".claude.json").write_text("{}")
    assert session(lab, "claude", "auth", stdin=None, HOME=home, FAKE_CLAUDE_OLD=1).returncode == 2


@pytest.mark.parametrize("agent", AGENTS + ["TEMPLATE"])
@pytest.mark.parametrize("args, stdin", [(["run", "--bogus"], PROMPT), (["run", "--model"], PROMPT),
                                         (["run", "--model", "primary", "--effort", "high"], ""), ([], None),
                                         (["frobnicate"], None)])
def test_a_wrong_call_exits_64_and_starts_no_agent(lab: Lab, agent, args, stdin):
    r = session(lab, agent, *args, stdin=stdin)
    assert r.returncode == 64 and r.stdout == "" and r.stderr.strip()
    assert lab.agent_calls() == []


# ---------------------------------------------------------------- the template
def test_the_template_wraps_an_agent_that_prints_plain_text(lab: Lab):
    agent = lab.bin / "my-agent"
    agent.write_text("#!/usr/bin/env bash\n[ \"${1:-}\" = --version ] && { echo 'my-agent 1.0'; exit 0; }\n"
                     "p=$(cat); echo \"read ${#p} characters\"; exit \"${MY_RC:-0}\"\n")
    agent.chmod(0o755)
    assert session(lab, "TEMPLATE", "version", stdin=None).stdout.strip() == "my-agent 1.0"
    assert session(lab, "TEMPLATE", "auth", stdin=None).returncode == 2
    r, out = run(lab, "TEMPLATE")
    assert r.returncode == 0 and out["is_error"] is False and "characters" in out["result"]
    r, out = run(lab, "TEMPLATE", MY_RC=3)
    assert r.returncode == 3 and out["is_error"] is True


# --------------------------------------------------------- files and the guard
@pytest.mark.parametrize("agent", AGENTS + ["TEMPLATE"])
def test_every_adapter_has_an_executable_session_script_that_parses(agent):
    script = ADAPTERS / agent / "session.sh"
    assert os.access(script, os.X_OK)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_adapter_scripts_pass_shellcheck():
    scripts = sorted(str(p) for p in ADAPTERS.glob("*/*.sh")) + [str(ADAPTERS / "common.sh")]
    r = subprocess.run(["shellcheck", "-x", "-S", "warning", *scripts], capture_output=True, text=True, cwd=ADAPTERS)
    assert r.returncode == 0, r.stdout


def lay_out(agent: str, root: Path) -> Path:
    """What setup does: the research repo's guard, plus the adapter's project files at the top."""
    shutil.copytree(GUARD, root / "guard")
    project = ADAPTERS / agent / "project"
    if project.is_dir():
        shutil.copytree(project, root, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def hook_commands(doc: dict) -> list[str]:
    return [h["command"] for groups in doc["hooks"].values() for g in groups for h in g["hooks"]]


def test_claude_project_files_run_the_guard_and_ship_three_helper_agents(tmp_path):
    root = lay_out("claude", tmp_path / "research")
    settings = json.loads((root / ".claude" / "settings.json").read_text())
    cmds = hook_commands(settings)
    assert len(cmds) == 3
    for c in cmds:
        r = subprocess.run(["bash", "-c", f"test -x {c}"], env={**os.environ, "CLAUDE_PROJECT_DIR": str(root)})
        assert r.returncode == 0, c
    assert sorted(p.stem for p in (ADAPTERS / "claude" / "project" / ".claude" / "agents").glob("*.md")
                  if "-" not in p.stem) == ["hypothesizer", "sceptic", "scout"]
    assert "@AGENTS.md" in (root / "CLAUDE.md").read_text()


def test_codex_project_files_run_the_guard_for_commands_and_patches(tmp_path):
    root = lay_out("codex", tmp_path / "research")
    doc = json.loads((root / ".codex" / "hooks.json").read_text())
    pre = {g["matcher"]: g["hooks"][0]["command"] for g in doc["hooks"]["PreToolUse"]}
    assert set(pre) == {"^Bash$", "^apply_patch$"}
    assert "pretooluse-bash.sh" in pre["^Bash$"] and "pretooluse-edit.sh" in pre["^apply_patch$"]
    for c in hook_commands(doc):
        assert subprocess.run(["bash", "-c", f"test -x {c}"], cwd=root).returncode == 0, c


NODE = shutil.which("node")
DRIVE_PI_GUARD = r"""
const mod = await import(process.argv[2]);
const handlers = {};
mod.default({ on: (name, fn) => { handlers[name] = fn; } });
const ctx = { cwd: process.argv[3] };
const out = [];
for (const [toolName, input] of JSON.parse(process.argv[4])) {
  const r = await handlers.tool_call({ type: "tool_call", toolName, input }, ctx);
  out.push(r && r.block ? { block: true, reason: r.reason } : { block: false });
  await handlers.tool_result({ type: "tool_result", toolName, input }, ctx);
}
console.log(JSON.stringify(out));
"""


def node_runs_typescript() -> bool:
    if not NODE:
        return False
    v = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")
    return (int(v[0]), int(v[1])) >= (23, 6)


@pytest.mark.skipif(not node_runs_typescript(), reason="needs node 23.6 or newer to load a .ts file")
def test_the_pi_guard_extension_sends_tool_calls_through_the_guard(lab: Lab, tmp_path):
    """guard.ts with a stand-in for pi's extension API and the real guard scripts."""
    driver = tmp_path / "drive.mjs"; driver.write_text(DRIVE_PI_GUARD)
    research = lab.research
    (research / "loop").mkdir()
    calls = [["bash", {"command": "ls"}],
             ["bash", {"command": "docker ps"}],
             ["write", {"path": str(research / "notes.md"), "content": "x"}],
             ["write", {"path": str(research / "loop" / "run.sh"), "content": "x"}],
             ["edit", {"path": "loop/run.sh"}],
             ["read", {"path": str(research / "loop" / "run.sh")}],
             ["some_extension_tool", {"path": str(research / "loop" / "x")}]]
    env = lab.env(IJ_GUARD_DIR=GUARD, IJ_ALLOWED_WRITE_ROOTS=research, IJ_PROTECTED_ROOTS=research / "loop",
                  IJ_AUDIT_LOG=tmp_path / "audit.log", IJ_GUARD_ENV=tmp_path / "none")

    def drive(e):
        r = subprocess.run([NODE, str(driver), str(ADAPTERS / "pi" / "guard.ts"), str(research), json.dumps(calls)],
                           env=e, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout.strip().splitlines()[-1])

    got = drive(env)
    assert [g["block"] for g in got] == [False, True, False, True, True, False, True]
    assert "docker" in got[1]["reason"] and "BLOCKED" in got[3]["reason"]
    assert len((tmp_path / "audit.log").read_text().splitlines()) == len(calls)
    # no guard, no tool calls: a missing script is a refusal, except for the readers
    got = drive({**env, "IJ_GUARD_DIR": str(tmp_path / "nowhere")})
    assert [g["block"] for g in got] == [True, True, True, True, True, False, True]
    assert "missing" in got[0]["reason"]

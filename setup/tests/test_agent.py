"""The setup side installs and checks whichever agent the operator picked
(research/adapters/<name>/; research/adapters/README.md is the contract).

bootstrap's steps are run in library mode (IJ_BOOTSTRAP_LIB=1) against a
throwaway copy of the repo, with fakes for chown, sudo and npm.  No real agent
is started: the agent's command is a stub, or absent.  check.sh and rehearse.sh
need root and the installed layout, so they are read as text, and what
rehearse.sh looks for is held against the line the real driver prints.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO

ADAPTERS = REPO / "research" / "adapters"

FAKES = {
    "chown": 'echo "chown $*" >> "$FAKE/chown.log"\n',
    "npm": 'echo "npm $*" >> "$FAKE/npm.log"\n',
    # sudo -u USER -H bash -lc CMD  ->  bash -c CMD
    "sudo": ('while [ $# -gt 0 ]; do case "$1" in -u) shift 2 ;; -H|-n) shift ;; *) break ;; esac; done\n'
             '[ "$1" = bash ] && [ "$2" = -lc ] && { shift 2; exec bash -c "$@"; }\n'
             'exec "$@"\n'),
}


class Box:
    """A copy of research/ in a throwaway repo, and a way to run one bootstrap step on it."""

    def __init__(self, root: Path):
        self.root = root
        self.fake, self.bin, self.home = root / "fake", root / "bin", root / "home-researcher"
        self.repo = root / "repo"
        for d in (self.fake, self.bin, self.home):
            d.mkdir()
        for name, body in FAKES.items():
            self.command(name, body)
        shutil.copytree(REPO / "research", self.repo / "research",
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "state.json"))
        (self.repo / "bench").mkdir()
        self.research = self.home / "research"
        self.loop_env = root / "loop.env"

    def command(self, name: str, body: str) -> Path:
        p = self.bin / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return p

    def bare_adapter(self, name: str = "bare") -> None:
        """An adapter that is a session.sh and nothing else, as TEMPLATE is."""
        shutil.copytree(ADAPTERS / "TEMPLATE", self.repo / "research" / "adapters" / name)

    def run(self, agent: str, script: str) -> subprocess.CompletedProcess:
        # No /opt/homebrew/bin or ~/.local/bin: a real agent installed on this machine stays out of reach.
        e = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home), "LC_ALL": "C",
             "FAKE": str(self.fake), "IJ_BOOTSTRAP_LIB": "1", "IJ_TARGET_PYTHON": sys.executable,
             "IJ_REPO": str(self.repo), "IJ_AGENT": agent, "IJ_LOOP_ENV_FILE": str(self.loop_env),
             "GIT_CONFIG_NOSYSTEM": "1"}
        body = (f'. "{REPO}/setup/bootstrap.sh"; RESEARCHER_HOME="{self.home}"; RESEARCH_DIR="{self.research}"; '
                f'ensure_bench_access() {{ :; }}; ( set -e; {script} )')
        return subprocess.run(["/bin/bash", "-c", body], env=e, cwd=self.root, capture_output=True,
                              text=True, timeout=120)

    def log(self, name: str) -> list[str]:
        p = self.fake / f"{name}.log"
        return p.read_text().splitlines() if p.exists() else []


@pytest.fixture
def box(tmp_path: Path) -> Box:
    return Box(tmp_path)


# ------------------------------------------------------------- installing it
@pytest.mark.parametrize("agent,package", [("claude", "@anthropic-ai/claude-code"), ("codex", "@openai/codex"),
                                           ("pi", "@earendil-works/pi-coding-agent")])
def test_an_agent_that_is_not_on_the_path_is_installed_by_its_adapter(box, agent, package):
    r = box.run(agent, "s_agent")
    assert r.returncode == 0, r.stderr
    assert box.log("npm") == [f"npm install -g {package}"]
    # the log-in line is the adapter's own, run as the researcher
    assert "su - researcher, then:" in r.stdout
    login = subprocess.run(["bash", str(ADAPTERS / agent / "setup.sh"), "login"], capture_output=True, text=True)
    assert login.stdout.strip() and login.stdout.strip() in r.stdout


def test_an_agent_already_on_the_path_is_left_alone(box):
    box.command("codex", 'echo "codex-stub 9.9"\n')
    r = box.run("codex", "s_agent")
    assert r.returncode == 0, r.stderr
    assert box.log("npm") == []
    assert "agent codex: codex-stub 9.9" in r.stdout


def test_an_adapter_without_setup_sh_does_not_fail_the_install(box):
    box.bare_adapter()
    r = box.run("bare", "s_agent")
    assert r.returncode == 0, r.stderr
    assert box.log("npm") == []
    assert "install the `bare` command yourself" in r.stderr
    # and once the operator has installed it, the step reports it like any other
    box.command("bare", 'echo "bare 0.1"\n')
    r = box.run("bare", "s_agent")
    assert r.returncode == 0 and "agent bare: bare 0.1" in r.stdout
    assert "its own documentation" in r.stdout          # no setup.sh, so no log-in line to quote


# ------------------------------------------------------- the research repo
def project_files(agent: str) -> list[str]:
    top = ADAPTERS / agent / "project"
    return sorted(str(p.relative_to(top)) for p in top.rglob("*") if p.is_file()) if top.is_dir() else []


@pytest.mark.parametrize("agent", ["claude", "codex", "pi"])
def test_the_research_repo_gets_the_agents_own_files_at_its_top(box, agent):
    r = box.run(agent, "s_research")
    assert r.returncode == 0, r.stderr
    for rel in project_files(agent):
        assert (box.research / rel).is_file(), rel
    assert (box.research / ".claude" / "settings.json").exists() == (agent == "claude")
    assert (box.research / ".codex" / "hooks.json").exists() == (agent == "codex")
    for rel in ("AGENTS.md", "guard/test-hooks.sh", "loop/run.sh", f"adapters/{agent}/session.sh"):
        assert (box.research / rel).is_file(), rel
    assert any(l.startswith("chown -R researcher:researcher ") for l in box.log("chown"))
    assert (box.research / ".git").is_dir()


def test_the_scripts_the_loop_runs_are_executable_after_the_copy(box):
    src = box.repo / "research"
    scripts = [*(src / "guard").glob("*.sh"), *(src / "loop").glob("*.sh"),
               *(src / "adapters").glob("*/session.sh"), *(src / "adapters").glob("*/setup.sh")]
    assert len(scripts) > 10
    for p in scripts:
        p.chmod(0o644)
    r = box.run("claude", "s_research")
    assert r.returncode == 0, r.stderr
    for p in scripts:
        assert os.access(box.research / p.relative_to(src), os.X_OK), p


# ---------------------------------------------------------------- loop.env
def test_an_agent_outside_sudos_secure_path_is_named_in_loop_env(box):
    stub = box.command("codex", "echo codex-stub\n")
    box.loop_env.write_text("IJ_AGENT=codex\n")
    r = box.run("codex", "s_loop_env")
    assert r.returncode == 0, r.stderr
    assert box.loop_env.read_text().splitlines() == ["IJ_AGENT=codex", f"IJ_AGENT_BIN={stub}"]
    assert box.run("codex", "s_loop_env").returncode == 0           # once is enough
    assert box.loop_env.read_text().count("IJ_AGENT_BIN=") == 1
    assert "IJ_CLAUDE_BIN" not in box.loop_env.read_text()


def test_an_agent_that_is_not_installed_adds_nothing_to_loop_env(box):
    box.loop_env.write_text("IJ_AGENT=pi\n")
    assert box.run("pi", "s_loop_env").returncode == 0
    assert box.loop_env.read_text() == "IJ_AGENT=pi\n"


def test_bootstrap_hands_the_agent_to_harden():
    text = (REPO / "setup" / "bootstrap.sh").read_text()
    assert 'AGENT="$IJ_AGENT" bash "$SCRIPT_DIR/harden_research.sh"' in text


# ------------------------------------------------- the example loop.env
def example_env() -> tuple[dict, str]:
    text = (REPO / "setup" / "systemd" / "loop.env.example").read_text()
    live = dict(l.split("=", 1) for l in text.splitlines() if l and not l.startswith("#"))
    return live, text


def test_the_example_loop_env_names_the_agent_and_freezes_the_new_folders():
    live, text = example_env()
    assert live["IJ_AGENT"] == "claude"
    r = "/home/researcher/research"
    assert live["IJ_FROZEN_PATHS"].split(":") == ["/opt/inference-judge", f"{r}/loop", f"{r}/guard",
                                                  f"{r}/adapters", f"{r}/.claude"]
    assert "#IJ_AGENT_BIN=" in text and "IJ_CLAUDE_BIN" not in text
    assert "#IJ_PRIMARY_MODEL=primary" in text and "#IJ_FALLBACK_MODEL=fallback" in text
    assert "hooks" not in text


# ------------------------------------------------ check.sh and rehearse.sh
CHECK = REPO / "setup" / "check.sh"
REHEARSE = REPO / "setup" / "rehearse.sh"


@pytest.mark.parametrize("path", [CHECK, REHEARSE, REPO / "setup" / "bootstrap.sh",
                                  REPO / "setup" / "harden_research.sh"])
def test_no_setup_script_names_the_old_layout(path):
    text = path.read_text()
    if path.name == "harden_research.sh":
        # harden names the old key once, to drop it from a loop.env written before the adapters
        assert text.count("IJ_CLAUDE_BIN") == 1 and "^IJ_CLAUDE_BIN=" in text
        text = text.replace("IJ_CLAUDE_BIN", "")
    for old in (".claude/hooks", "IJ_CLAUDE_BIN", "inference-judge/hooks", "claude -p", "--model opus",
                "--dangerously-skip-permissions", "npm install"):
        assert old not in text, f"{path.name} still names {old}"


@pytest.mark.parametrize("path", [CHECK, REHEARSE])
def test_the_checks_take_the_agent_from_loop_env_and_ask_its_adapter(path):
    text = path.read_text()
    assert "IJ_AGENT" in text and "adapters/$AGENT/session.sh" in text
    assert "version" in text
    # the driver's route: the researcher's sudo prefix, then env IJ_AGENT_BIN=..., then the adapter
    assert "$IJ_RESEARCHER_SUDO env" in text and "IJ_AGENT_BIN=$IJ_AGENT_BIN" in text
    assert "claude --version" not in text


def test_check_hands_the_adapter_the_model_name_so_pi_can_check_its_login():
    """Pi checks a login per model: its `auth` cannot tell without IJ_PRIMARY_MODEL, and sudo drops it."""
    assert "IJ_PRIMARY_MODEL=$IJ_PRIMARY_MODEL" in CHECK.read_text()


def test_check_asks_the_adapter_whether_the_agent_is_logged_in():
    text = CHECK.read_text()
    assert "session.sh auth" in text or '"$ADAPTER" auth' in text or "agent_session auth" in text
    assert "log-in state not inspected" in text                    # status 2: cannot tell
    assert "setup.sh\" login" in text or "setup.sh login" in text   # status 1: the hint
    assert ".claude.json" not in text                              # the adapter knows where a login lives


def test_check_looks_for_the_files_of_the_new_layout():
    text = CHECK.read_text()
    for needle in ("AGENTS.md", "guard/test-hooks.sh", '"adapters/$AGENT/session.sh"', "adapters/$AGENT/project",
                   "/opt/inference-judge/guard", "/opt/inference-judge/adapters", "/etc/inference-judge/guard.env"):
        assert needle in text, needle
    assert '"$RESEARCH/guard/pretooluse-edit.sh"' in text and '"$RESEARCH/guard/pretooluse-bash.sh"' in text
    assert ".claude/settings.json" not in text                     # it comes from the adapter's project/


def test_check_and_rehearse_are_clean():
    if not shutil.which("shellcheck"):
        pytest.skip("shellcheck not installed")
    for path in (CHECK, REHEARSE):
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0, path
        r = subprocess.run(["shellcheck", "-S", "warning", str(path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout


def driver_dry_run_line(tmp_path: Path, agent: str, **more: str) -> str:
    """The command line the real driver prints for its next session (no agent is started)."""
    env = dict(os.environ, IJ_STATE_DIR=str(tmp_path / "state"), IJ_RESEARCH_DIR=str(REPO / "research"),
               IJ_ALERT="", IJ_FROZEN_PATHS="", IJ_AGENT=agent, IJ_LEGACY_STATE="/nonexistent",
               IJ_RESEARCHER_SUDO="sudo -n -u someone -H", IJ_AGENT_BIN=f"/usr/local/bin/{agent}", **more)
    run = ["bash", str(REPO / "research" / "loop" / "run.sh")]
    subprocess.run([*run, "--start-clock"], env=env, capture_output=True, text=True, timeout=60, check=True)
    r = subprocess.run([*run, "--dry-run"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if "session.sh run" in l]
    assert len(lines) == 1, r.stdout
    return lines[0]


@pytest.mark.parametrize("agent", ["claude", "codex", "pi"])
def test_rehearse_looks_for_what_the_driver_really_prints(tmp_path, agent):
    if not shutil.which("jq"):
        pytest.skip("jq not installed")
    line = driver_dry_run_line(tmp_path, agent)
    text = REHEARSE.read_text()
    # how rehearse.sh finds the line, and each thing it then wants in it
    assert "grep -F -- \"adapters/$AGENT/session.sh run\"" in text
    assert '"--model $PRIMARY_MODEL" "--effort $EFFORT"' in text and '"-u $RESEARCHER_USER"' in text
    for want in (f"adapters/{agent}/session.sh run", " -u someone", " --model primary", " --effort high",
                 f"env IJ_AGENT_BIN=/usr/local/bin/{agent}"):
        assert want in line, (want, line)


def test_rehearse_takes_the_model_and_the_effort_from_loop_env(tmp_path):
    """An operator who names a model in loop.env must not fail the rehearsal's driver case."""
    if not shutil.which("jq"):
        pytest.skip("jq not installed")
    line = driver_dry_run_line(tmp_path, "pi", IJ_PRIMARY_MODEL="vendor/some-model", IJ_EFFORT="medium")
    assert " --model vendor/some-model" in line and " --effort medium" in line
    text = REHEARSE.read_text()
    for name, default in (("IJ_PRIMARY_MODEL", "primary"), ("IJ_EFFORT", "high")):
        assert f"s/^{name}=//p" in text, name                  # read from loop.env ...
        assert ":-" + default + "}" in text, (name, default)   # ... with the driver's default


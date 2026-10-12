"""setup/harden_research.sh: the permissions the guard only warns about.

On the Mac: shellcheck and a --dry-run.  With Docker: the real script, as
root, in a throwaway ubuntu:24.04 container with the three users, fake
accelerator device nodes and a stub /opt/bench/run.sh; then the researcher
tries to get round it, and the driver runs one queue item through the real
runner and reaps a detached researcher job."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import LOOP

REPO = LOOP.parent.parent
HARDEN = REPO / "setup" / "harden_research.sh"
IMAGES = ("inference-judge-harden-test:24.04", "ubuntu:24.04")   # the first has sudo, python3 and jq baked in


def test_shellcheck_clean():
    if not shutil.which("shellcheck"):
        pytest.skip("shellcheck not installed")
    r = subprocess.run(["shellcheck", str(HARDEN)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_dry_run_prints_every_step_and_touches_nothing(tmp_path):
    env = dict(os.environ, RESEARCH_DIR=str(REPO / "research"), STATE_DIR=str(tmp_path / "state"),
               INSTALL_DIR=str(tmp_path / "opt-inference-judge"), MANAGED_SETTINGS=str(tmp_path / "managed.json"),
               GUARD_ENV=str(tmp_path / "guard.env"),
               SUDOERS_DIR=str(tmp_path / "sudoers.d"), SYSTEMD_DIR=str(tmp_path / "systemd"),
               ENV_FILE=str(tmp_path / "loop.env"), DEV_DIR=str(tmp_path / "dev"))
    env.pop("AGENT", None), env.pop("IJ_AGENT", None)
    r = subprocess.run(["bash", str(HARDEN), "--dry-run", "--no-venv"], capture_output=True,
                       text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    for step in range(1, 9):
        assert f"== {step}. " in r.stdout
    assert '"allowManagedHooksOnly": true' in r.stdout
    assert "NOPASSWD: /opt/bench/run.sh" in r.stdout
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- one agent or another
# --dry-run prints every file it would write ("   + write <path> (<mode> <owner>):"
# and the content behind "       | "), so what each agent gets is read from there.
STATE, INSTALL, RESULTS = "/home/driver/research-state", "/opt/inference-judge", "/opt/bench/results"


def dry(tmp_path, agent=None, research=None, **extra):
    research = str(research or REPO / "research")
    env = {k: v for k, v in os.environ.items() if k not in ("AGENT", "IJ_AGENT", "MANAGED_SETTINGS", "GUARD_ENV")}
    env.update(RESEARCH_DIR=research, STATE_DIR=STATE, INSTALL_DIR=INSTALL, ENV_FILE=str(tmp_path / "loop.env"),
               SUDOERS_DIR=str(tmp_path / "sudoers.d"), SYSTEMD_DIR=str(tmp_path / "systemd"),
               DEV_DIR=str(tmp_path / "dev"))
    env.update(extra)
    if agent is not None:
        env["AGENT"] = agent
    r = subprocess.run(["bash", str(HARDEN), "--dry-run", "--no-venv"], capture_output=True,
                       text=True, env=env, timeout=60)
    assert [p for p in tmp_path.iterdir() if p.name != "research"] == []
    return r


def written(stdout: str) -> dict:
    """{path: (mode and owner, content)} for every file the dry run would write."""
    out, path = {}, None
    for ln in stdout.splitlines():
        if ln.startswith("   + write "):
            path, _, meta = ln[len("   + write "):].rstrip(":").partition(" (")
            out[path] = [meta.rstrip(")"), []]
        elif ln.startswith("       | ") and path:
            out[path][1].append(ln[len("       | "):])
        elif ln.strip() == "|" and path:
            out[path][1].append("")
        else:
            path = None
    return {k: (m, "\n".join(body) + "\n") for k, (m, body) in out.items()}


def frozen(research: str, *project_dirs: str) -> list:
    return [f"{research}/{d}" for d in ("loop", "guard", "adapters", *project_dirs)]


def assert_guard_env(files: dict, research: str, *project_dirs: str) -> None:
    meta, body = files["/etc/inference-judge/guard.env"]
    assert meta == "0644 root:root"
    roots = ":".join([*frozen(research, *project_dirs), STATE, INSTALL])
    assert body.splitlines() == [f"IJ_PROTECTED_ROOTS={roots}", f"IJ_STATE_DIR={STATE}",
                                 f"IJ_RESEARCH_DIR={research}", f"IJ_RESULTS_DIR={RESULTS}"]


def assert_loop_env(files: dict, tmp_path, agent: str, research: str, *project_dirs: str) -> None:
    body = files[str(tmp_path / "loop.env")][1]
    live = dict(l.split("=", 1) for l in body.splitlines() if l and not l.startswith("#"))
    assert live["IJ_AGENT"] == agent
    assert live["IJ_FROZEN_PATHS"].split(":") == [INSTALL, *frozen(research, *project_dirs)]
    assert live["IJ_RESEARCH_DIR"] == research and "IJ_CLAUDE_BIN" not in body


def test_claude_gets_managed_settings_that_run_the_guard(tmp_path):
    research = str(REPO / "research")
    r = dry(tmp_path)                                   # no AGENT: the default is claude
    assert r.returncode == 0, r.stderr
    assert dry(tmp_path, "claude").stdout == r.stdout
    files = written(r.stdout)
    meta, body = files["/etc/claude-code/managed-settings.json"]
    assert meta == "0644 root:root"
    d = json.loads(body)
    assert d["allowManagedHooksOnly"] is True
    cmds = [h["command"] for ev in d["hooks"].values() for m in ev for h in m["hooks"]]
    assert sorted(cmds) == [f"{INSTALL}/guard/{n}.sh" for n in ("posttooluse-audit", "pretooluse-bash", "pretooluse-edit")]
    roots = [*frozen(research, ".claude"), STATE, INSTALL]
    assert d["permissions"]["deny"] == [f"Edit(/{p}/**)" for p in roots]
    assert d["env"] == {"IJ_PROTECTED_ROOTS": ":".join(roots), "IJ_STATE_DIR": STATE,
                        "IJ_RESEARCH_DIR": research, "IJ_RESULTS_DIR": RESULTS}
    assert "/hooks/" not in r.stdout
    assert_guard_env(files, research, ".claude")
    assert_loop_env(files, tmp_path, "claude", research, ".claude")
    assert f"{INSTALL}/guard.new" in r.stdout and f"{INSTALL}/adapters.new" in r.stdout


def test_codex_gets_requirements_that_run_the_guard(tmp_path):
    research = str(REPO / "research")
    r = dry(tmp_path, "codex")
    assert r.returncode == 0, r.stderr
    files = written(r.stdout)
    assert "/etc/claude-code/managed-settings.json" not in files
    meta, body = files["/etc/codex/requirements.toml"]
    assert meta == "0644 root:root"
    assert "allow_managed_hooks_only = true" in body.splitlines()
    cmds = [l.split("=", 1)[1].strip().strip('"') for l in body.splitlines() if l.startswith("command =")]
    assert sorted(cmds) == [f"{INSTALL}/guard/{n}.sh" for n in ("posttooluse-audit", "pretooluse-bash", "pretooluse-edit")]
    if sys.version_info >= (3, 11):
        import tomllib
        t = tomllib.loads(body)
        assert t["allow_managed_hooks_only"] is True and t["features"]["hooks"] is True
        assert len(t["hooks"]["PreToolUse"]) == 2 and len(t["hooks"]["PostToolUse"]) == 1
    assert_guard_env(files, research, ".codex")
    assert_loop_env(files, tmp_path, "codex", research, ".codex")


def test_pi_gets_no_managed_file_and_the_same_guard_env(tmp_path):
    research = str(REPO / "research")
    r = dry(tmp_path, "pi")
    assert r.returncode == 0, r.stderr
    files = written(r.stdout)
    assert not [p for p in files if "claude-code" in p or "/codex/" in p]
    assert "the pi adapter names no managed settings file" in r.stdout
    assert_guard_env(files, research)
    assert_loop_env(files, tmp_path, "pi", research)
    for step in range(1, 9):
        assert f"== {step}. " in r.stdout


def test_the_agent_can_come_from_ij_agent_and_agent_wins(tmp_path):
    assert "/etc/codex/requirements.toml" in written(dry(tmp_path, IJ_AGENT="codex").stdout)
    assert "/etc/codex/requirements.toml" not in written(dry(tmp_path, "pi", IJ_AGENT="codex").stdout)


def test_a_managed_file_is_not_written_for_an_agent_that_has_none(tmp_path):
    """MANAGED_SETTINGS moves the file; it does not make one for an agent without any."""
    r = dry(tmp_path, "pi", MANAGED_SETTINGS="/etc/somewhere/managed.json")
    assert r.returncode == 0 and "/etc/somewhere/managed.json" not in r.stdout
    r = dry(tmp_path, "codex", MANAGED_SETTINGS="/etc/somewhere/requirements.toml")
    assert "/etc/somewhere/requirements.toml" in written(r.stdout) and "/etc/codex/" not in r.stdout


def test_an_adapter_with_no_setup_sh_is_hardened_like_any_other(tmp_path):
    research = tmp_path / "research"                    # never inside the real tree
    shutil.copytree(REPO / "research", research, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copytree(research / "adapters" / "TEMPLATE", research / "adapters" / "bare")
    assert not (research / "adapters" / "bare" / "setup.sh").exists()
    r = dry(tmp_path, "bare", research=research)
    assert r.returncode == 0, r.stderr
    files = written(r.stdout)
    assert "the bare adapter has no setup.sh: no managed settings file" in r.stdout
    assert not [p for p in files if "claude-code" in p or "/codex/" in p]
    assert_guard_env(files, str(research))
    assert_loop_env(files, tmp_path, "bare", str(research))
    for step in range(1, 9):
        assert f"== {step}. " in r.stdout


def test_an_existing_loop_env_follows_a_change_of_agent_and_keeps_the_rest(tmp_path):
    """A box hardened before the adapters, or for another agent: loop.env is the operator's,
    but the driver must run the agent the machine was hardened for."""
    research = str(REPO / "research")
    box = tmp_path / "research"                         # the one name dry() lets a test leave behind
    box.mkdir()
    old = box / "loop.env"
    old.write_text("# mine\nIJ_TIMER_S=60\nIJ_FROZEN_PATHS=/opt/inference-judge:/x/.claude:/x/loop\n"
                   "IJ_CLAUDE_BIN=/home/researcher/.local/bin/claude\nIJ_AGENT_BIN=/home/researcher/.local/bin/claude\n")
    before = old.read_text()
    r = dry(tmp_path, "codex", ENV_FILE=str(old))
    assert r.returncode == 0, r.stderr
    assert old.read_text() == before                    # a dry run
    meta, body = written(r.stdout)[str(old)]
    assert meta == "0640 root:driver"
    assert body.splitlines() == ["# mine", "IJ_TIMER_S=60", "IJ_AGENT=codex",
                                 "IJ_FROZEN_PATHS=" + ":".join([INSTALL, *frozen(research, ".codex")])]
    # the same agent and the same folders: the file is not touched, and its IJ_AGENT_BIN stays
    old.write_text(body + "IJ_AGENT_BIN=/home/researcher/.local/bin/codex\n")
    assert str(old) not in written(dry(tmp_path, "codex", ENV_FILE=str(old)).stdout)


@pytest.mark.parametrize("name", ["gemini", "../claude", "claude/../codex", "-x", "a b"])
def test_an_agent_without_an_adapter_is_refused_before_anything_is_done(tmp_path, name):
    r = dry(tmp_path, name)
    assert r.returncode != 0
    assert "harden_research:" in r.stderr and "adapter" in r.stderr
    assert "== 1. " not in r.stdout and "+ " not in r.stdout


def test_help_says_the_agent_not_claude_p():
    r = subprocess.run(["bash", str(HARDEN), "--help"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "claude -p" not in r.stdout and "AGENT" in r.stdout and "guard" in r.stdout
    assert "set -euo pipefail" not in r.stdout          # the header comment and nothing after it
    assert "test_harden.py" in r.stdout                 # ... all of it


def _docker_image():
    if not shutil.which("docker"):
        return None
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return None
    for img in IMAGES:
        if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode == 0:
            return img
    return None


IN_CONTAINER = r"""
set -u
fails=0
check() {  # check <want-rc> <description> <command...>
  local want="$1" desc="$2"; shift 2
  "$@" >/tmp/check.out 2>&1; local got=$?
  if { [ "$want" = 0 ] && [ "$got" = 0 ]; } || { [ "$want" != 0 ] && [ "$got" != 0 ]; }; then
    echo "PASS $desc"
  else
    echo "FAIL $desc (rc=$got)"; sed 's/^/    /' /tmp/check.out | head -5; fails=$((fails+1))
  fi
}
if ! command -v sudo >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq sudo python3 procps jq >/dev/null
fi
useradd --system --create-home --home-dir /home/bench --shell /bin/bash bench
useradd --create-home --home-dir /home/researcher --shell /bin/bash researcher
cp -R /src/research /home/researcher/research
cp -R /src/research/adapters/claude/project/. /home/researcher/research/     # bootstrap's s_research does this
chown -R researcher:researcher /home/researcher/research
mkdir -p /opt/bench/results /home/researcher/.venv/lib
echo torch > /home/researcher/.venv/lib/torch
cat > /opt/bench/run.sh <<'EOF'
#!/bin/bash
printf '%s\n' "$@" > /tmp/bench.argv
id="${!#}"; mkdir -p "/opt/bench/results/$id"
printf '{"id":"%s","verdict":"REJECTED","duration_s":3}\n' "$id" > "/opt/bench/results/$id/results.json"
EOF
chmod 0755 /opt/bench/run.sh; chown -R bench:bench /opt/bench; chmod 0755 /opt/bench /opt/bench/results
echo 'researcher ALL=(bench) NOPASSWD: /opt/bench/run.sh' > /etc/sudoers.d/inference-judge-researcher; chmod 0440 /etc/sudoers.d/inference-judge-researcher
mkdir -p /fakedev/vfio; for d in nvidia0 nvidiactl nvidia-uvm accel0 vfio/12; do echo x > /fakedev/$d; chmod 0666 /fakedev/$d; done
mkdir -p /etc/systemd/system

export DEV_DIR=/fakedev
check 0 "harden runs"            bash /src/setup/harden_research.sh --no-venv
cat /tmp/check.out | tail -3
check 0 "harden is idempotent"   bash /src/setup/harden_research.sh --no-venv

R=/home/researcher/research
as_researcher() { sudo -u researcher -H "$@"; }
as_driver()  { sudo -u driver -H "$@"; }
check 1 "researcher cannot write loop/"              as_researcher touch $R/loop/x
check 1 "researcher cannot overwrite a guard script" as_researcher sh -c "echo 'exit 0' > $R/guard/pretooluse-bash.sh"
check 1 "researcher cannot overwrite an adapter"     as_researcher sh -c "echo 'exit 0' > $R/adapters/claude/session.sh"
check 1 "researcher cannot add an adapter"           as_researcher mkdir $R/adapters/mine
check 1 "researcher cannot edit settings.json"       as_researcher sh -c "echo '{}' > $R/.claude/settings.json"
check 0 "researcher still writes CLAUDE.md and AGENTS.md" as_researcher sh -c "echo >> $R/CLAUDE.md && echo >> $R/AGENTS.md"
check 1 "researcher cannot write the installed loop" as_researcher touch /opt/inference-judge/loop/x
check 1 "researcher cannot write the installed guard" as_researcher touch /opt/inference-judge/guard/x
check 1 "researcher cannot write the installed adapters" as_researcher touch /opt/inference-judge/adapters/claude/x
check 0 "the installed adapter and guard run"        sh -c 'test -x /opt/inference-judge/adapters/claude/session.sh -a -x /opt/inference-judge/guard/pretooluse-bash.sh'
check 1 "no tests or caches in the install"          sh -c 'find /opt/inference-judge -name tests -o -name __pycache__ | grep -q .'
check 1 "no old hooks folder in the install"         test -e /opt/inference-judge/hooks
check 1 "researcher cannot write the state dir"      as_researcher touch /home/driver/research-state/x
check 0 "driver writes the state dir"                as_driver sh -c 'echo brief > /home/driver/research-state/BRIEF.md'
check 0 "researcher reads BRIEF.md"                  as_researcher cat /home/driver/research-state/BRIEF.md
check 1 "researcher cannot read a GPU node"          as_researcher cat /fakedev/nvidia0
check 1 "researcher cannot read a TPU node"          as_researcher cat /fakedev/accel0
check 1 "researcher cannot open vfio"                as_researcher cat /fakedev/vfio/12
check 0 "devices are group bench 0660"               sh -c '[ "$(stat -c %G:%a /fakedev/nvidia0)" = bench:660 ] && [ "$(stat -c %G:%a /fakedev/vfio/12)" = bench:660 ]'
check 0 "bench can read a GPU node"                  sudo -u bench cat /fakedev/nvidia0
check 1 "researcher cannot run the bench directly"   as_researcher sudo -n -u bench /opt/bench/run.sh exp-x
check 0 "driver runs the bench"                      as_driver sudo -n -u bench /opt/bench/run.sh exp-y
check 0 "driver acts as the researcher"              sh -c '[ "$(sudo -u driver sudo -n -u researcher id -un)" = researcher ]'
check 1 "researcher cannot become the driver"        as_researcher sudo -n -u driver true
check 0 "sudoers parse"                              visudo -c
check 0 "managed settings are JSON with managed hooks only" python3 -c 'import json; d=json.load(open("/etc/claude-code/managed-settings.json")); assert d["allowManagedHooksOnly"] is True; assert d["hooks"]["PreToolUse"][1]["hooks"][0]["command"].startswith("/opt/inference-judge/guard/"); assert d["env"]["IJ_STATE_DIR"] == "/home/driver/research-state"'
check 0 "guard.env is root-owned, world-readable, and names the frozen folders" sh -c '[ "$(stat -c %U:%G:%a /etc/inference-judge/guard.env)" = root:root:644 ] && sudo -u researcher grep -qx "IJ_PROTECTED_ROOTS=/home/researcher/research/loop:/home/researcher/research/guard:/home/researcher/research/adapters:/home/researcher/research/.claude:/home/driver/research-state:/opt/inference-judge" /etc/inference-judge/guard.env && grep -qx "IJ_RESULTS_DIR=/opt/bench/results" /etc/inference-judge/guard.env'
check 1 "researcher cannot write guard.env"          as_researcher sh -c 'echo x >> /etc/inference-judge/guard.env'
check 0 "loop.env names the agent and the frozen folders" sh -c 'grep -qx "IJ_AGENT=claude" /etc/inference-judge/loop.env && grep -qx "IJ_FROZEN_PATHS=/opt/inference-judge:/home/researcher/research/loop:/home/researcher/research/guard:/home/researcher/research/adapters:/home/researcher/research/.claude" /etc/inference-judge/loop.env'
check 1 "researcher cannot read the GPU venv"        as_researcher cat /home/researcher/.venv/lib/torch
check 0 "driver reads the research repo"             as_driver ls $R/experiments
check 0 "units installed"                            test -f /etc/systemd/system/inference-judge-driver.service -a -f /etc/systemd/system/inference-judge-heartbeat.timer
check 0 "env file readable by the driver only"       sh -c 'sudo -u driver cat /etc/inference-judge/loop.env >/dev/null && ! sudo -u researcher cat /etc/inference-judge/loop.env 2>/dev/null'

# ---- the loop itself, in the three-user layout, configured only by loop.env ----
printf '%s\n' '#!/bin/bash' \
  '# run "$@" as the driver user with /etc/inference-judge/loop.env, as the systemd units do' \
  'exec sudo -u driver -H bash -c '\''set -a; . /etc/inference-judge/loop.env; set +a; IJ_LEGACY_STATE=/nonexistent; exec "$@"'\'' as-driver "$@"' \
  > /usr/local/bin/as-driver
chmod 0755 /usr/local/bin/as-driver
as_researcher mkdir -p $R/queue $R/experiments/exp-ok
as_researcher sh -c "printf '#!/usr/bin/env bash\necho hi\n' > $R/experiments/exp-ok/launch.sh"
as_researcher sh -c "echo '{\"id\":\"H1\",\"experiment\":\"exp-ok\",\"prediction\":-5}' >> $R/HYPOTHESES.jsonl"
as_researcher sh -c "echo '{\"id\":\"exp-ok\",\"kind\":\"candidate\",\"cmd\":\"; touch /tmp/pwned\"}' > $R/queue/exp-ok.json"
LS=/opt/inference-judge/loop/loopstate.py
check 0 "driver starts the clock"      as-driver python3 -I $LS start-clock --accel-hours 24 --wall-hours 24
check 0 "runner runs one item"         as-driver /opt/inference-judge/loop/runner.sh --once
check 0 "bench argv is exactly the id" sh -c '[ "$(cat /tmp/bench.argv)" = exp-ok ]'
check 1 "no injected command ran"      test -e /tmp/pwned
check 0 "verdict landed in runs.jsonl" grep -q REJECTED /home/driver/research-state/runs.jsonl
jobpid=$(as_researcher bash $R/loop/bg.sh sweep sleep 300 | cut -d' ' -f1)
check 0 "researcher job started"       kill -0 "$jobpid"
check 0 "researcher preflight sees the driver's runs" sh -c "sudo -u researcher -H env IJ_STATE_DIR=/home/driver/research-state $R/loop/preflight.sh exp-ok | grep -q 'runs.jsonl has a REJECTED verdict'"
check 0 "driver ticks"                 as-driver python3 -I $LS next
check 0 "the job is in the ledger"     grep -q "\"pgid\": $jobpid" /home/driver/research-state/events.jsonl
check 0 "driver sets a past deadline"  as-driver python3 -I $LS start-clock --accel-hours 24 --deadline 1
check 0 "driver finishes"              sh -c "as-driver python3 -I $LS next | grep -q '^finish'"
sleep 1
check 1 "the job was reaped"           sh -c "ps -o stat= -p $jobpid | grep -qv Z"
check 0 "driver dry run"               as-driver bash /opt/inference-judge/loop/run.sh --dry-run
echo "FAILS=$fails"
"""


def test_harden_in_a_container():
    img = _docker_image()
    if img is None:
        pytest.skip("docker or an ubuntu:24.04 image is not available")
    r = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO}:/src:ro", img, "bash", "-c", IN_CONTAINER],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert "FAILS=0" in out, out
    assert out.count("PASS ") >= 30, out

"""setup/systemd/: the units that keep the loop alive for days.

A structural check: the files parse, point at scripts that exist in loop/,
and carry the restart policy the driver's exit codes rely on (exit 10 =
finished, anything else = restart).  systemd-analyze is not run here."""
from __future__ import annotations

import configparser
import re
import shlex

import pytest

from conftest import LOOP

UNITS = LOOP.parent.parent / "setup" / "systemd"
INSTALL = "/opt/inference-judge/loop/"


def unit(name: str) -> configparser.ConfigParser:
    p = configparser.ConfigParser(strict=False, interpolation=None)
    p.optionxform = str
    p.read_string((UNITS / name).read_text())
    return p


@pytest.mark.parametrize("name", ["inference-judge-driver.service", "inference-judge-runner.service", "inference-judge-heartbeat.service"])
def test_services_run_the_installed_loop_as_the_driver_user(name):
    u = unit(name)
    s = u["Service"]
    assert s["User"] == "driver" and s["Group"] == "driver"
    assert s["EnvironmentFile"] == "/etc/inference-judge/loop.env"
    exe = shlex.split(s["ExecStart"])[0]
    assert exe.startswith(INSTALL), "units must run the root-owned copy, not the researcher's repo"
    assert (LOOP / exe[len(INSTALL):]).is_file()
    assert u["Unit"]["Description"]


def test_driver_restarts_on_anything_but_finished():
    u = unit("inference-judge-driver.service")
    s = u["Service"]
    assert s["Restart"] == "always"
    assert "10" in s["RestartPreventExitStatus"].split() and "10" in s["SuccessExitStatus"].split()
    assert u["Unit"]["StartLimitIntervalSec"] == "0"          # never give up restarting
    assert s["KillMode"] == "mixed"
    wants = u["Unit"]["Wants"].split()
    assert "inference-judge-runner.service" in wants and "inference-judge-heartbeat.timer" in wants
    assert u["Install"]["WantedBy"] == "multi-user.target"
    # run.sh's own exit code for "finished" is the one systemd is told about
    assert re.search(r"^EXIT_FINISHED=10$", (LOOP / "run.sh").read_text(), re.M)


def test_runner_is_restarted_and_given_time_to_finish_a_measurement():
    s = unit("inference-judge-runner.service")["Service"]
    assert s["Restart"] in ("on-failure", "always")
    assert int(s["TimeoutStopSec"]) >= 120


def test_heartbeat_timer_fires_every_minute():
    t = unit("inference-judge-heartbeat.timer")
    assert t["Timer"]["Unit"] == "inference-judge-heartbeat.service"
    assert int(t["Timer"]["OnUnitActiveSec"]) <= 120 and t["Timer"]["OnBootSec"]
    assert t["Install"]["WantedBy"] == "timers.target"
    assert unit("inference-judge-heartbeat.service")["Service"]["Type"] == "oneshot"


def env_file() -> dict:
    out = {}
    for ln in (UNITS / "loop.env.example").read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        k, _, v = ln.partition("=")
        assert re.fullmatch(r"[A-Z_][A-Z0-9_]*", k), ln
        if " " in v:   # systemd and `. loop.env` agree only on quoted values
            assert v.startswith('"') and v.endswith('"'), f"quote {k}: {ln}"
        out[k] = v.strip('"')
    return out


def test_env_file_keys_are_ones_the_loop_reads():
    env = env_file()
    code = "".join((LOOP / f).read_text() for f in ("loopstate.py", "run.sh", "runner.py", "preflight.sh"))
    for k in env:
        if k.startswith("IJ_"):
            assert k in code, f"{k} is set in loop.env but nothing reads it"
    assert env["IJ_PREFLIGHT_CMD"].startswith(INSTALL)
    assert env["IJ_RESEARCHER_SUDO"].split()[:2] == ["sudo", "-n"]
    assert env["IJ_BENCH_CMD"] == "sudo -n -u bench /opt/bench/run.sh"
    assert "/opt/inference-judge" in env["IJ_FROZEN_PATHS"].split(":")

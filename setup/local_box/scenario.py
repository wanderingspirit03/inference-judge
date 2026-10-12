#!/usr/bin/python3 -I
"""The local box's end-to-end scenario.  Runs as root INSIDE the box container
(setup/local_box/run_local_box.sh starts it with `docker exec`).

    scenario.py                  every stage, then exit 0 iff every check passed

Stages (each check prints PASS/FAIL and lands in /var/log/localbox/summary.json):
  install   stage the read-only /src checkout, plant a world-readable sealed-file
            canary and a fake accelerator device, run setup/bootstrap.sh for
            targets/cpu-mock.yaml, run it again (install-or-resume), check the
            three users, sudoers, frozen loop, modes and the clock.
  operator  what the runbook has the operator do next, before the units start:
            setup/check.sh must pass, and setup/rehearse.sh --quick must pass
            through the queue and the installed runner, leave nothing in the
            real run's state, results or queue, and restart the clock.
  A         the driver's first claude session (the stub) queues four candidates
            and probes the researcher boundary; the runner pre-flights and
            benches them: verdicts, no slot for the broken patch, BRIEF.md.
  B         a session and a bench run are both in flight; the driver is
            SIGKILLed; it must restart and resume from events.jsonl with no
            double counting.
  C         the stub crashes on every call: back-off, three crashes on the
            primary model, the fallback model, three more, then a pause with an
            alert -- never an exit.  The runner keeps draining the queue.
  D         heartbeat, git commits of the research repo, ledgers.

Tier 1 (default): /opt/bench/run.sh is replaced by setup/local_box/stub_bench.sh,
a fake ruler, after bootstrap has installed and locked the real one.
Tier 2 (LOCALBOX_TIER=2): the real run.sh stays.  The operator stage then also
runs the runbook's reference run and calibration (two launches) of the mock, and
the whole rehearsal (hang and "same" included) goes through the real ruler in
the box's own docker.  The box stops there: phases A-D are timed against a ruler
that answers in seconds, so they are tier 1.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True          # never write into /opt/inference-judge (it is hashed)

SRC = Path("/src")
REPO = Path("/root/inference-judge")
HERE = SRC / "setup" / "local_box"
LOG = Path("/var/log/localbox")
STATE = Path("/home/driver/research-state")
RESEARCH = Path("/home/researcher/research")
RESULTS = Path("/opt/bench/results")
MODE = Path("/etc/inference-judge/localbox/mode")
UNITS = Path("/etc/systemd/system")
SUP_PID = Path("/run/inference-judge-supervise")
SUP_LEDGER = Path("/var/log/inference-judge/supervise.jsonl")
TARGET = "targets/cpu-mock.yaml"

CHECKS: list[dict] = []
T0 = time.time()
TIMES: dict[str, float] = {}


# ------------------------------------------------------------------ plumbing
def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + str(detail)[:300]) if detail and not ok else ''}",
          flush=True)
    return bool(ok)


def sh(cmd: str | list, user: str | None = None, check_rc: bool = False, timeout: int = 1800,
       env: dict | None = None, cwd: str | Path | None = None, log: Path | None = None):
    argv = ["bash", "-c", cmd] if isinstance(cmd, str) else list(cmd)
    if user:
        argv = ["sudo", "-n", "-u", user, "-H", *argv]
    e = dict(os.environ)
    e.update(env or {})
    if log:
        with open(log, "ab") as f:
            p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=e, cwd=cwd, timeout=timeout)
        p.stdout = log.read_text(errors="replace")[-3000:]
    else:
        p = subprocess.run(argv, capture_output=True, text=True, env=e, cwd=cwd, timeout=timeout)
        p.stdout = (p.stdout or "") + (p.stderr or "")
    if check_rc and p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {cmd}\n{p.stdout[-2000:]}")
    return p


def wait_for(what: str, pred, timeout: float, every: float = 1.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(every)
    say(f"timed out after {timeout:.0f}s waiting for: {what}")
    return None


def events() -> list[dict]:
    out = []
    try:
        for line in (STATE / "events.jsonl").read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def loopstate():
    sys.path.insert(0, "/opt/inference-judge/loop")
    import loopstate as ls  # noqa: PLC0415
    return ls


def state():
    ls = loopstate()
    return ls.replay(ls.load_events(STATE / "events.jsonl"))


def verdicts_for(eid: str) -> list[dict]:
    return [r for r in state().runs.values() if r.get("id") == eid]


def claude_calls() -> list[dict]:
    p = LOG / "claude.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def set_mode(m: str) -> None:
    MODE.write_text(m + "\n")
    say(f"stub claude mode -> {m}")


def sup_ledger() -> list[dict]:
    return [json.loads(l) for l in SUP_LEDGER.read_text().splitlines()] if SUP_LEDGER.exists() else []


# ------------------------------------------------------------------- install
def stage() -> int:
    say("staging /src -> /root/inference-judge")
    sh(["rsync", "-a", "--delete", "--exclude", ".git/", "--exclude", ".venv*/", "--exclude", ".cache/",
        "--exclude", "__pycache__/", "--exclude", ".pytest_cache/", "--exclude", "*.egg-info/",
        "--exclude", "/bench/results/",
        "--exclude", "/bench/bench/", "--exclude", "/docs/", f"{SRC}/", f"{REPO}/"], check_rc=True)
    tier = int(os.environ.get("LOCALBOX_TIER") or 1)
    say(f"tier {tier}: {'the real bench/run.sh' if tier == 2 else 'stub ruler (setup/local_box/stub_bench.sh)'}")
    LOG.mkdir(parents=True, exist_ok=True)
    LOG.chmod(0o1777)
    shutil.copy(HERE / "stub_claude.py", "/usr/local/bin/claude")
    os.chmod("/usr/local/bin/claude", 0o755)
    MODE.parent.mkdir(parents=True, exist_ok=True)
    set_mode("research")
    os.chmod(MODE, 0o644)
    # A fake accelerator, world-readable before harden; harden must hand it to bench.
    Path("/fakedev").mkdir(exist_ok=True)
    Path("/fakedev/accel0").write_bytes(b"\0")
    os.chmod("/fakedev/accel0", 0o666)
    # A sealed ruler file planted world-readable: bootstrap must lock it down.
    Path("/opt/bench/canary").mkdir(parents=True, exist_ok=True)
    Path("/opt/bench/canary/localbox.json").write_text('{"canary": "LOCALBOX-SEALED-CANARY"}\n')
    os.chmod("/opt/bench/canary/localbox.json", 0o644)
    return tier


BOOT_ENV = {"DEV_DIR": "/fakedev", "IJ_HARDEN_FLAGS": "--no-venv", "IJ_ACCEL_HOURS": "2",
            "IJ_WALL_HOURS": "2"}


def install(tier: int) -> None:
    t = time.time()
    p = sh(f"bash setup/bootstrap.sh --target {TARGET}", cwd=REPO, env=BOOT_ENV, log=LOG / "bootstrap.log")
    TIMES["bootstrap_s"] = round(time.time() - t, 1)
    if not check("bootstrap for cpu-mock exits 0", p.returncode == 0, p.stdout[-1500:]):
        raise SystemExit(1)
    # The stand-in agent has no login.  adapters/claude/session.sh auth reads a
    # ~/.claude.json as "has logged in once", and setup/check.sh asks it.
    sh("echo '{}' > /home/researcher/.claude.json", user="researcher", check_rc=True)
    out = (LOG / "bootstrap.log").read_text()
    ran = [l[7:] for l in out.splitlines() if l.startswith("PASS ")]
    gpuish = [l for l in ran if any(w in l.lower() for w in ("gpu", "nsight", "clone vllm", "inference-judge-base", "model"))]
    check("bootstrap ran no GPU, engine-build or model step", ran and not gpuish, gpuish or out[-500:])
    check("bootstrap skipped the accelerator, nsys and model steps",
          all(s in out for s in ("SKIP   accelerator driver", "SKIP   nsight-systems", "SKIP   model download")))
    runcfg = [e for e in events() if e["type"] == "run_config"]
    check("clock started through loopstate start-clock (as driver)",
          len(runcfg) == 1 and runcfg[0].get("source") == "start-clock", runcfg)
    check("the state dir and events.jsonl belong to driver",
          sh("stat -c %U /home/driver/research-state /home/driver/research-state/events.jsonl").stdout.split()
          == ["driver", "driver"])

    t = time.time()
    p = sh(f"bash setup/bootstrap.sh --target {TARGET}", cwd=REPO, env=BOOT_ENV, log=LOG / "bootstrap-2.log")
    TIMES["bootstrap_rerun_s"] = round(time.time() - t, 1)
    check("bootstrap re-run (install-or-resume) exits 0", p.returncode == 0, p.stdout[-800:])
    check("the re-run did not restart the clock",
          [e for e in events() if e["type"] == "run_config"] == runcfg)

    # Users, sudoers, frozen code, modes.
    check("users bench, researcher and driver exist", sh("id bench && id researcher && id driver").returncode == 0)
    sud = sh("cat /etc/sudoers.d/inference-judge-driver").stdout
    check("sudoers: driver may run /opt/bench/run.sh as bench", "driver ALL=(bench) NOPASSWD: /opt/bench/run.sh" in sud, sud)
    check("sudoers: driver may act as researcher", "driver ALL=(researcher) NOPASSWD: ALL" in sud, sud)
    lc = sh("sudo -l -U researcher").stdout
    check("sudoers: researcher has no route to run.sh", "/opt/bench/run.sh" not in lc, lc)
    check("visudo accepts the sudoers files", sh("visudo -c").returncode == 0)
    check("/opt/inference-judge/loop is root-owned and not writable by others",
          sh("stat -c '%U %a' /opt/inference-judge/loop").stdout.split()[0] == "root"
          and sh("find /opt/inference-judge -perm -o+w | head -1").stdout.strip() == "")
    frozen = ["loop", "guard", "adapters", ".claude"]
    own = sh("stat -c '%U %a' " + " ".join(f"{RESEARCH}/{d}" for d in frozen)).stdout.split()
    check("research/loop, guard, adapters and .claude are root-owned and only root may write them",
          own[0::2] == ["root"] * len(frozen) and all(int(m, 8) & 0o022 == 0 for m in own[1::2]), own)
    check("/opt/inference-judge has loop, guard and adapters, and the claude adapter can be run",
          sh("test -d /opt/inference-judge/loop && test -x /opt/inference-judge/guard/pretooluse-bash.sh"
             " && test -x /opt/inference-judge/adapters/claude/session.sh").returncode == 0)
    genv = sh("cat /etc/inference-judge/guard.env").stdout
    check("guard.env is root's (0644) and names the research repo and the frozen paths",
          sh("stat -c '%U:%G %a' /etc/inference-judge/guard.env").stdout.strip() == "root:root 644"
          and f"IJ_RESEARCH_DIR={RESEARCH}" in genv.splitlines()
          and all(f"{RESEARCH}/{d}" in genv for d in frozen), genv)
    check("sealed-file canary is bench-only (0640)",
          sh("stat -c '%U:%G %a' /opt/bench/canary/localbox.json").stdout.strip() == "bench:bench 640")
    check("fake accelerator handed to group bench (0660)",
          sh("stat -c '%G %a' /fakedev/accel0").stdout.strip() == "bench 660")
    check("target installed: TARGET + targets/cpu-mock.yaml bench 0640, public copy 0644",
          sh("stat -c '%U %a' /opt/bench/TARGET /opt/bench/targets/cpu-mock.yaml /etc/inference-judge/target.yaml"
             ).stdout.split() == ["bench", "640", "bench", "640", "root", "644"]
          and Path("/opt/bench/TARGET").read_text().strip() == "cpu-mock")
    if "--lock-ruler" in (REPO / "bench" / "run.sh").read_text():
        check("RULER_LOCK written at clock start, through the driver's sudo route",
              sh("test -s /opt/bench/RULER_LOCK && stat -c %U /opt/bench/RULER_LOCK").stdout.strip() == "bench")
    if "IJ_RULER_IMAGE" in (REPO / "bench" / "run.sh").read_text():
        check("ruler image inference-judge-ruler:latest built", sh("docker image inspect inference-judge-ruler:latest").returncode == 0)
    check("engine image inference-judge-mock:latest built in the box's own docker",
          sh("docker image inspect inference-judge-mock:latest").returncode == 0)
    check("units installed for the supervisor",
          all((UNITS / u).exists() for u in ("inference-judge-driver.service", "inference-judge-runner.service",
                                             "inference-judge-heartbeat.timer", "inference-judge-heartbeat.service")))

    # Fast policy for the box, then the ruler for this tier.
    with open("/etc/inference-judge/loop.env", "a") as f:
        f.write("\n# local box: fast policy (scenario.py)\n" + "\n".join([
            "IJ_LEGACY_STATE=/nonexistent", "IJ_BACKOFF_BASE_S=1", "IJ_BACKOFF_MAX_S=4",
            "IJ_MIN_GAP_S=2", "IJ_POLL_S=1", "IJ_TIMER_S=6", "IJ_RUN_MIN_S=10",
            "IJ_RUNNER_POLL_S=1", "IJ_ITER_TIMEOUT_S=90", "IJ_KILL_GRACE=2",
            "IJ_TICK_CAP_S=30"]) + "\n")
    if tier == 1:
        shutil.copy(HERE / "stub_bench.sh", "/opt/bench/run.sh")
        sh("chown bench:bench /opt/bench/run.sh && chmod 0750 /opt/bench/run.sh", check_rc=True)
        # The fake ruler's incumbent: rehearse.sh refuses a tps target without one.
        sh("install -d -o bench -g bench -m 0755 /opt/bench/leaderboard && "
           "printf '%s\n' '{\"stub\": true, \"best\": {\"id\": \"baseline-mock\"}}' "
           "> /opt/bench/leaderboard/cpu-mock.json && chown bench:bench /opt/bench/leaderboard/cpu-mock.json",
           check_rc=True)


def operator_baseline() -> None:
    """The runbook's tps baseline, as the operator types it: the reference
    launch under a baseline-* id, accepted only while .allow-baseline exists."""
    exp = RESEARCH / "experiments" / "baseline-mock"
    sh(f"mkdir -p {exp} && printf '%s\\n' '#!/usr/bin/env bash' 'set -euo pipefail' "
       f"'exec python3 /opt/inference-judge/mock_server.py --host 0.0.0.0 --port 8000 --model mock --behaviour honest' "
       f"> {exp}/launch.sh && chmod 0755 {exp}/launch.sh", user="researcher", check_rc=True)
    sh("install -o bench -g bench -m 0640 /dev/null /opt/bench/.allow-baseline", check_rc=True)
    t = time.time()
    p = sh("sudo -u bench /opt/bench/run.sh baseline-mock", log=LOG / "baseline.log", timeout=1800)
    TIMES["baseline_s"] = round(time.time() - t, 1)
    sh("rm -f /opt/bench/.allow-baseline")
    try:
        v = json.loads((RESULTS / "baseline-mock" / "results.json").read_text()).get("verdict")
    except (OSError, ValueError):
        v = None
    check("operator baseline of the mock -> BASELINE (real ruler, sealed launch)", v == "BASELINE",
          v or p.stdout[-600:])
    check("the baseline made the incumbent (leaderboard/cpu-mock.json)",
          Path("/opt/bench/leaderboard/cpu-mock.json").exists())


# The operator modes are refused through sudo (run.sh trusts no flag when
# SUDO_USER is set), so the runbook drops SUDO_USER and enters as bench.
OPERATOR = "env -u SUDO_USER runuser -u bench -- /opt/bench/run.sh"


def operator_calibrate() -> None:
    """The runbook's Step 0, typed the way the runbook has it, from a sudo shell."""
    p = sh("sudo -u bench /opt/bench/run.sh --calibrate calibrate-sudo")
    check("an operator mode (--calibrate) is refused through sudo",
          p.returncode != 0 and not (RESULTS / "calibrate-sudo").exists(), p.stdout[-300:])
    t = time.time()
    # two launches, not ten: stats.calibrate's minimum, to keep the box short
    p = sh(f"sudo bash -c 'CALIBRATE_LAUNCHES=2 {OPERATOR} --calibrate calibrate-1'",
           log=LOG / "calibrate.log", timeout=3600)
    TIMES["calibrate_s"] = round(time.time() - t, 1)
    try:
        v = json.loads((RESULTS / "calibrate-1" / "results.json").read_text()).get("verdict")
        cal = json.loads(Path("/opt/bench/calibration/cpu-mock.json").read_text())
    except (OSError, ValueError):
        v, cal = None, {}
    check("operator calibration (env -u SUDO_USER runuser -u bench) -> CALIBRATED", v == "CALIBRATED",
          v or p.stdout[-600:])
    check("the calibration is installed (calibration/cpu-mock.json: sigma_L, mde)",
          cal.get("n_launches") == 2 and "sigma_L" in cal and "mde" in cal, cal)
    # Step 3 needs a promoted run (seven launches); here only its dry run.
    p = sh(f"sudo bash -c '{OPERATOR} --dry-run --confirm baseline-mock'", log=LOG / "confirm-dry.log")
    say(f"finding: --confirm dry run through the operator route exits {p.returncode} (reported, not asserted)")
    TIMES["confirm_dry_rc"] = p.returncode


def operator(tier: int) -> None:
    t = time.time()
    p = sh("bash setup/check.sh", cwd=REPO, log=LOG / "check.log")
    fails = [l for l in p.stdout.splitlines() if "FAIL" in l and "failed" not in l]
    check("setup/check.sh passes on the fresh box", p.returncode == 0, fails or p.stdout[-600:])
    TIMES["check_s"] = round(time.time() - t, 1)

    if tier == 2:
        operator_baseline()
        operator_calibrate()
    before = [e for e in events() if e["type"] == "run_config"]
    t = time.time()
    cases = "--quick" if tier == 1 else ""      # tier 2: every case, "same" included
    p = sh(f"bash setup/rehearse.sh {cases}".strip(), cwd=REPO, log=LOG / "rehearse.log", timeout=7200)
    TIMES["rehearse_s"] = round(time.time() - t, 1)
    out = (LOG / "rehearse.log").read_text(errors="replace")
    fails = [l for l in out.splitlines() if l.startswith("\x1b[31mFAIL")]
    check(f"setup/rehearse.sh {cases or '(every case)'} passes (queue -> installed runner -> ruler)", p.returncode == 0,
          fails or out[-600:])
    want = ["preflight failure", "build failure", "hang is ended by the ruler", "hang: container cleaned up",
            "driver runs adapters/claude/session.sh", "driver passes -u researcher",
            "driver passes --model primary", "driver passes --effort high"]
    if tier == 2:       # the real gate must not promote a copy of the incumbent
        want.append("same as reference")
    for case in want:
        check(f"rehearse: {case}", any(case in l and "PASS" in l for l in out.splitlines()))
    check("rehearse: its verdicts stayed out of the real run (events.jsonl)",
          not any("rehearsal-" in str(e.get("id", "")) for e in events()))
    left = sorted(x.name for x in RESULTS.glob("rehearsal-*")) + \
        sorted(x.name for x in (RESEARCH / "queue").glob("rehearsal-*"))
    check("rehearse: no rehearsal results dir or queue file left behind", not left, left)
    ev = sorted(Path("/var/log/inference-judge").glob("rehearse-*/results/rehearsal-*"))
    check("rehearse: its results moved to /var/log/inference-judge/rehearse-*/ as evidence", len(ev) >= 2, ev)
    after = [e for e in events() if e["type"] == "run_config"]
    check("rehearse: restarted the run clock with the same budget",
          len(after) == len(before) + 1 and after[-1]["clock_start"] > before[-1]["clock_start"]
          and after[-1]["accel_budget_s"] == before[-1]["accel_budget_s"], after[-1:])


# ------------------------------------------------------------------- phases
def start_supervisor() -> subprocess.Popen:
    say("starting the supervisor (inference-judge-runner, inference-judge-driver, inference-judge-heartbeat.timer)")
    return subprocess.Popen(["python3", "-I", str(HERE / "supervise.py"), str(UNITS), "inference-judge-runner.service",
                             "inference-judge-driver.service", "inference-judge-heartbeat.timer"],
                            env={**os.environ, "LOCALBOX_TIME_SCALE": "0.05"},
                            stdout=open(LOG / "supervise.out", "ab"), stderr=subprocess.STDOUT)


FOUR = ("same-as-base", "broken-patch", "fast-mock", "junk-mock")


def phase_a(tier: int) -> None:
    t = time.time()
    ok = wait_for("verdicts for the four candidates", lambda: all(verdicts_for(i) for i in FOUR), 300)
    TIMES["phase_a_s"] = round(time.time() - t, 1)
    check("A: all four candidates reached a verdict", ok, {i: [r.get("verdict") for r in verdicts_for(i)] for i in FOUR})
    v = {i: [r.get("verdict") for r in verdicts_for(i)] for i in FOUR}
    evs = events()
    check("A: broken-patch -> PREFLIGHT_FAILED", v["broken-patch"] == ["PREFLIGHT_FAILED"], v["broken-patch"])
    check("A: broken-patch spent no ruler slot (no run_start, gpu_s 0, no results dir)",
          not any(e["type"] == "run_start" and e.get("id") == "broken-patch" for e in evs)
          and all(float(r.get("gpu_s") or 0) == 0 for r in verdicts_for("broken-patch"))
          and not (RESULTS / "broken-patch").exists())
    check("A: same-as-base is not PROMOTED", v["same-as-base"] and "PROMOTED" not in v["same-as-base"], v["same-as-base"])
    check("A: fast-mock is PROMOTED" + (" (stub ruler)" if tier == 1 else ""), v["fast-mock"] == ["PROMOTED"], v["fast-mock"])
    check("A: junk-mock is INVALID or REJECTED", len(v["junk-mock"]) == 1 and v["junk-mock"][0] in ("INVALID", "REJECTED"),
          v["junk-mock"])
    order = [e["id"] for e in evs if e["type"] == "run_start"]
    check("A: the runner ran the queue FIFO, one at a time",
          order[:3] == ["same-as-base", "fast-mock", "junk-mock"], order)

    calls = claude_calls()
    c0 = calls[0] if calls else {}
    check("A: the driver ran claude (through its adapter) as the researcher, in the research repo, on opus",
          bool(calls) and c0.get("cwd") == str(RESEARCH) and c0.get("model") == "opus"
          and c0.get("uid") == int(sh("id -u researcher").stdout.strip()), c0)
    check("A: the driver passes --dangerously-skip-permissions and --output-format json",
          "--dangerously-skip-permissions" in c0.get("argv", []) and "json" in c0.get("argv", []), c0.get("argv"))
    check("A: the prompt points at BRIEF.md and the stub could read it", c0.get("brief_copied") is True, c0)

    att = json.loads((LOG / "attacks.json").read_text()) if (LOG / "attacks.json").exists() else {}
    check("A: the boundary probes ran", bool(att), "no attacks.json")
    for name, r in att.get("must_fail", {}).items():
        check(f"A: researcher cannot {name}", r["rc"] != 0, r)
    for name, r in att.get("must_pass", {}).items():
        check(f"A: researcher can {name}", r["rc"] == 0, r)
    check("A: the sealed-file canary never appears in a probe's output",
          "LOCALBOX-SEALED-CANARY" not in json.dumps(att))

    brief_ok = wait_for("BRIEF.md to show the verdicts",
                        lambda: "fast-mock" in (STATE / "BRIEF.md").read_text()
                        and "PROMOTED" in (STATE / "BRIEF.md").read_text(), 60)
    check("A: BRIEF.md was rewritten with the new verdicts", bool(brief_ok))
    briefs = sorted(LOG.glob("brief-*.md"), key=lambda p: int(p.stem.split("-")[1]))
    check("A: successive sessions were sent different BRIEFs",
          len(briefs) >= 2 and len({b.read_text() for b in briefs}) >= 2, [b.name for b in briefs])
    s = state()
    distinct = {e["run_id"] for e in events() if e["type"] == "verdict"}
    t_ = s.tallies()
    check("A: tallies match the distinct run ids in events.jsonl",
          t_["runs"] + t_["preflight_failed"] + t_["baseline"] == len(distinct), (t_, len(distinct)))
    check("A: tallies count 1 promoted, 1 pre-flight failure",
          t_["promoted"] == 1 and t_["preflight_failed"] == 1, t_)


def driver_pid() -> int:
    return int((SUP_PID / "inference-judge-driver.pid").read_text())


def phase_b() -> None:
    t = time.time()
    n_starts = sum(1 for e in events() if e["type"] == "driver_start")
    set_mode("slow")
    inflight = wait_for(
        "a slow session and the slow-mock bench run, both in flight",
        lambda: any(c["mode"] == "slow" for c in claude_calls())
        and any(e["type"] == "run_start" and e.get("id") == "slow-mock" for e in events())
        and not verdicts_for("slow-mock")
        and state().session_open is not None, 120, 0.5)
    check("B: a session and a bench run were in flight together", bool(inflight))
    sessions_before = state().sessions
    pid = driver_pid()
    say(f"SIGKILL driver pid {pid}")
    os.kill(pid, signal.SIGKILL)
    set_mode("quiet")
    restarted = wait_for("the supervisor to restart the driver",
                         lambda: sum(1 for e in events() if e["type"] == "driver_start") == n_starts + 1, 60)
    check("B: the driver was restarted (Restart=always) and logged driver_start", bool(restarted))
    check("B: the dead session was closed by the restart (no session left open)",
          wait_for("session_open cleared", lambda: state().session_open is None
                   or state().sessions > sessions_before, 30) is not None)
    done = wait_for("slow-mock's verdict", lambda: verdicts_for("slow-mock"), 120)
    check("B: the bench run survived the driver kill and reached a verdict", bool(done))
    check("B: slow-mock counted exactly once", len(verdicts_for("slow-mock")) == 1, verdicts_for("slow-mock"))
    later = wait_for("a session after the restart",
                     lambda: any(e["type"] == "session_end" for e in events()
                                 if e["seq"] > max(x["seq"] for x in events() if x["type"] == "driver_start")), 60)
    check("B: the restarted driver ran a new session", bool(later))
    s = state()
    t_ = s.tallies()
    distinct = {e["run_id"] for e in events() if e["type"] == "verdict"}
    check("B: tallies still equal the distinct run ids (no double counting)",
          t_["runs"] + t_["preflight_failed"] + t_["baseline"] == len(distinct), (t_, len(distinct)))
    sess_starts = sum(1 for e in events() if e["type"] == "session_start")
    check("B: the session count equals the session_start events", s.sessions == sess_starts, (s.sessions, sess_starts))
    rl = [json.loads(l) for l in (STATE / "runs.jsonl").read_text().splitlines()] if (STATE / "runs.jsonl").exists() else []
    ids = [r.get("run_id") for r in rl]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    TIMES["runs_jsonl_duplicates"] = len(dups)
    say(f"finding: runs.jsonl has {len(ids)} lines, {len(dups)} duplicated run_id(s) {dups[:5]}"
        " (replay dedupes; reported, not asserted)")
    TIMES["phase_b_s"] = round(time.time() - t, 1)


def queue_as_researcher(eid: str) -> None:
    script = f"""
set -e
cd {RESEARCH}
mkdir -p experiments/{eid} queue
printf '#!/usr/bin/env bash\\nexec python3 /opt/inference-judge/mock_server.py --port 8000 --behaviour honest\\n' > experiments/{eid}/launch.sh
chmod +x experiments/{eid}/launch.sh
printf '%s\\n' '{{"experiment": "{eid}", "prediction": 1.0}}' >> HYPOTHESES.jsonl
printf '%s' '{{"id": "{eid}", "kind": "candidate"}}' > queue/.{eid}.tmp && mv queue/.{eid}.tmp queue/{eid}.json
"""
    sh(script, user="researcher", check_rc=True)


def phase_c() -> None:
    t = time.time()
    n_calls = len(claude_calls())
    n_starts = sum(1 for e in events() if e["type"] == "driver_start")
    set_mode("crash")
    paused = wait_for("the driver to pause on crashes",
                      lambda: (state().paused or {}).get("reason") == "crashes", 240)
    check("C: six crashed sessions end in a pause (reason crashes)", bool(paused), state().paused)
    time.sleep(4)
    models = [c["model"] for c in claude_calls()[n_calls:] if c["mode"] == "crash"]
    check("C: three crashes on opus, then three on sonnet, then nothing",
          models == ["opus"] * 3 + ["sonnet"] * 3, models)
    ends = [e for e in events() if e["type"] == "session_end" and e.get("class") == "crash"][-6:]
    gaps = [round(b["ts"] - a["ts"], 1) for a, b in zip(ends, ends[1:])]
    check("C: the driver backed off between crashes (gaps grow within each model)",
          len(gaps) == 5 and gaps[1] >= gaps[0] - 0.5 and gaps[4] >= gaps[3] - 0.5, gaps)
    alerts = [e for e in events() if e["type"] == "alert"]
    check("C: an alert was raised for the switch to the fallback model",
          any("consecutive crashed sessions on the primary model" in e.get("msg", "") for e in alerts),
          alerts[-3:])
    check("C: an alert was raised for the pause",
          any("crashing on the fallback model" in e.get("msg", "") for e in alerts), alerts[-3:])
    push = Path("/home/driver/research-state/ALERTS.push")
    check("C: the alert reached IJ_ALERT (the push file)", push.exists() and push.stat().st_size > 0)
    check("C: the driver never exited (no new driver_start, supervisor saw no exit)",
          sum(1 for e in events() if e["type"] == "driver_start") == n_starts
          and sum(1 for e in sup_ledger() if e["unit"] == "inference-judge-driver" and e["event"] == "exit") == 1)
    queue_as_researcher("after-pause")
    drained = wait_for("the runner to bench a candidate queued while paused",
                       lambda: verdicts_for("after-pause"), 60)
    check("C: the runner keeps draining the queue while the driver is paused (by design)", bool(drained))
    TIMES["phase_c_s"] = round(time.time() - t, 1)


def phase_d() -> None:
    hb = STATE / "heartbeat.json"
    ok = wait_for("heartbeat.json", lambda: hb.exists() and time.time() - hb.stat().st_mtime < 30, 30)
    check("D: the heartbeat timer writes heartbeat.json", bool(ok))
    if hb.exists():
        h = json.loads(hb.read_text())
        check("D: the heartbeat sees the live driver and the pause",
              h.get("driver_alive") is True and (h.get("paused") or {}).get("reason") == "crashes", h)
    log = sh(f"git -C {RESEARCH} log --format=%s", user="researcher").stdout
    check("D: the driver committed the research repo after sessions", "driver: session" in log, log[:300])
    tracked = sh(f"git -C {RESEARCH} ls-files queue", user="researcher").stdout.strip()
    check("D: queue/ is not committed (research/.gitignore)", tracked == "", tracked)
    check("D: the frozen paths never changed (no frozen pause)",
          not any(e["type"] == "pause" and "frozen" in e.get("reason", "") for e in events()))


def finish(tier: int) -> int:
    TIMES["total_s"] = round(time.time() - T0, 1)
    fails = [c for c in CHECKS if not c["ok"]]
    (LOG / "summary.json").write_text(json.dumps({"tier": tier, "times": TIMES, "checks": CHECKS}, indent=1))
    say(f"{len(CHECKS) - len(fails)}/{len(CHECKS)} checks passed; times {TIMES}")
    return 1 if fails else 0


def main() -> int:
    tier = stage()
    install(tier)
    operator(tier)
    if tier == 2:
        return finish(tier)
    sup = start_supervisor()
    try:
        phase_a(tier)
        phase_b()
        phase_c()
        phase_d()
    finally:
        sup.send_signal(signal.SIGTERM)
        try:
            sup.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sup.kill()
    return finish(tier)

if __name__ == "__main__":
    sys.exit(main())

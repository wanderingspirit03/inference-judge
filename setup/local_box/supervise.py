#!/usr/bin/env python3
"""A very small systemd stand-in for the local box (no systemd in a container).

It reads the INSTALLED unit files (the ones setup/harden_research.sh wrote to
/etc/systemd/system), so the box exercises the real units -- their User=,
EnvironmentFile=, ExecStart=, WorkingDirectory= and restart policy -- not a
hand-written copy of them.  Supported, because the inference-judge units use them:

    [Service] Type=simple|oneshot  User=  Group=  EnvironmentFile=  WorkingDirectory=
              ExecStart=  Restart=always|on-failure|no  RestartSec=
              RestartPreventExitStatus=  SuccessExitStatus=
    [Timer]   OnBootSec=  OnUnitActiveSec=  Unit=

When a service's main process exits, every process it started that is still
alive is killed (systemd does this with the unit's cgroup; here the supervisor
remembers the process tree it last saw), then the restart policy is applied.

    supervise.py UNIT_DIR UNIT...            run in the foreground until SIGTERM
    LOCALBOX_TIME_SCALE=0.05                 multiply RestartSec and timer periods

The ledger /var/log/inference-judge/supervise.jsonl gets one line per start and exit:
{"t", "unit", "event": "start"|"exit", "pid", "rc"}; /run/inference-judge-supervise/<unit>.pid
holds the current main pid.  Both are for run_local_box.sh's assertions.
"""
from __future__ import annotations

import grp
import json
import os
import pwd
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOCALBOX_LOG_DIR", "/var/log/inference-judge"))
PID_DIR = Path(os.environ.get("LOCALBOX_PID_DIR", "/run/inference-judge-supervise"))
SCALE = float(os.environ.get("LOCALBOX_TIME_SCALE", "1"))


def parse_unit(path: Path) -> dict:
    """{section: {key: value}}; later keys win, as in systemd for these keys."""
    out: dict = {}
    sec = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = out.setdefault(line[1:-1], {})
            continue
        if sec is not None and "=" in line:
            k, v = line.split("=", 1)
            sec[k.strip()] = v.strip()
    return out


def parse_env_file(path: str) -> dict:
    """systemd EnvironmentFile: KEY=VALUE lines, # comments, outer quotes stripped."""
    env = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")) or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        env[k.strip()] = v
    return env


def seconds(v: str | None, default: float) -> float:
    if not v:
        return default
    v = v.strip()
    for suf, mul in (("min", 60), ("ms", 0.001), ("s", 1), ("h", 3600)):
        if v.endswith(suf):
            return float(v[: -len(suf)]) * mul
    return float(v)


def ledger(**kv) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "supervise.jsonl", "a") as f:
        f.write(json.dumps({"t": round(time.time(), 3), **kv}) + "\n")


def children_of(pids: set[int]) -> set[int]:
    """Every live descendant of `pids`, from /proc."""
    parent: dict[int, int] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
            parent[int(d)] = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    out, frontier = set(), set(pids)
    while frontier:
        nxt = {p for p, pp in parent.items() if pp in frontier and p not in out}
        out |= nxt
        frontier = nxt
    return out


class Service:
    def __init__(self, name: str, unit: dict):
        s = unit.get("Service", {})
        self.name = name
        self.type = s.get("Type", "simple")
        self.user = s.get("User", "root")
        self.group = s.get("Group")
        self.argv = shlex.split(s["ExecStart"])
        self.cwd = s.get("WorkingDirectory") or "/"
        self.env_file = s.get("EnvironmentFile")
        self.restart = s.get("Restart", "no")
        self.restart_s = seconds(s.get("RestartSec"), 0.1) * SCALE
        self.prevent = {int(x) for x in s.get("RestartPreventExitStatus", "").split() if x.isdigit()}
        self.success = {0} | {int(x) for x in s.get("SuccessExitStatus", "").split() if x.isdigit()}
        self.proc: subprocess.Popen | None = None
        self.tree: set[int] = set()
        self.next_start = 0.0
        self.done = False

    def env(self) -> dict:
        pw = pwd.getpwnam(self.user)
        e = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
             "HOME": pw.pw_dir, "USER": self.user, "LOGNAME": self.user, "LANG": "C.UTF-8"}
        if self.env_file:
            e.update(parse_env_file(self.env_file.lstrip("-")))
        return e

    def start(self) -> None:
        pw = pwd.getpwnam(self.user)
        gid = grp.getgrnam(self.group).gr_gid if self.group else pw.pw_gid
        groups = os.getgrouplist(self.user, gid)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = open(LOG_DIR / f"{self.name}.log", "ab")
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, env=self.env(), stdin=subprocess.DEVNULL,
                                     stdout=out, stderr=subprocess.STDOUT, user=pw.pw_uid, group=gid,
                                     extra_groups=groups, start_new_session=True)
        out.close()
        self.tree = set()
        PID_DIR.mkdir(parents=True, exist_ok=True)
        (PID_DIR / f"{self.name}.pid").write_text(f"{self.proc.pid}\n")
        ledger(unit=self.name, event="start", pid=self.proc.pid)

    def kill_leftovers(self) -> None:
        live = {p for p in self.tree if Path(f"/proc/{p}").exists()}
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in live:
                try:
                    os.kill(p, sig)
                except ProcessLookupError:
                    pass
            time.sleep(0.5)
            live = {p for p in live if Path(f"/proc/{p}").exists()}
            if not live:
                break

    def poll(self, now: float) -> None:
        if self.done:
            return
        if self.proc is None:
            if now >= self.next_start:
                self.start()
            return
        rc = self.proc.poll()
        if rc is None:
            self.tree |= children_of({self.proc.pid})
            return
        ledger(unit=self.name, event="exit", pid=self.proc.pid, rc=rc)
        self.kill_leftovers()
        self.proc = None
        clean = rc in self.success
        again = (self.restart == "always" and rc not in self.prevent) or \
                (self.restart == "on-failure" and not clean and rc not in self.prevent)
        if again:
            self.next_start = now + self.restart_s
        else:
            self.done = True

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.tree |= children_of({self.proc.pid})
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            ledger(unit=self.name, event="exit", pid=self.proc.pid, rc=self.proc.returncode, stopped=True)
        self.kill_leftovers()
        self.done = True


class Timer:
    def __init__(self, svc: Service, unit: dict):
        t = unit.get("Timer", {})
        self.svc = svc
        self.svc.done = True                    # only the timer starts it
        self.period = seconds(t.get("OnUnitActiveSec"), 60) * SCALE
        self.next = time.time() + seconds(t.get("OnBootSec"), 0) * SCALE

    def poll(self, now: float) -> None:
        if self.svc.proc is not None:
            self.svc.done = False
            self.svc.poll(now)                  # reaps a finished oneshot
            self.svc.done = True
            if self.svc.proc is not None and self.svc.proc.poll() is None:
                return
            self.svc.proc = None
        if now >= self.next:
            self.svc.start()
            self.next = now + self.period


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    unit_dir = Path(argv[0])
    services: dict[str, Service] = {}
    timers: list[Timer] = []
    for name in argv[1:]:
        u = parse_unit(unit_dir / name)
        if name.endswith(".timer"):
            target = u.get("Timer", {}).get("Unit") or name.replace(".timer", ".service")
            svc = Service(target.removesuffix(".service"), parse_unit(unit_dir / target))
            timers.append(Timer(svc, u))
        else:
            services[name.removesuffix(".service")] = Service(name.removesuffix(".service"), u)
    stopping = False

    def on_term(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    while not stopping:
        now = time.time()
        for s in services.values():
            s.poll(now)
        for t in timers:
            t.poll(now)
        time.sleep(0.25)
    for s in services.values():
        s.stop()
    for t in timers:
        t.svc.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

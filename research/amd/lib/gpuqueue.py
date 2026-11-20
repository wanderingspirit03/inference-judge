"""One GPU, one queue, one holder.  Stdlib only, Linux.

State lives in <state_dir>/gpu/:

    lock            the card.  Its holder keeps an exclusive flock(2) on it.
                    The kernel drops the flock when the holder's process
                    ends in any way, kill -9 included, so a dead holder
                    always frees the card.
    holder.json     who took the lock last: pid, command, cwd, start time,
                    token.  It is only believed while the lock is held.
    tickets/        one file per waiter, named by a sequence number.  The
                    waiter keeps a flock on its own ticket; a ticket whose
                    flock can be taken belongs to a dead waiter and is removed.
                    The waiter with the lowest number goes next (first come,
                    first served).
    seq             the last sequence number given out.

A job runs with IJ_GPU_JOB=<token> in its environment.  A command started
inside a job (./check under ./try, ./ref from a script under ./try) sees that
its token is the holder's and runs inside the same hold instead of waiting for
itself.  When a job ends, every process that still carries its token is
killed.  When a new holder takes the card, every process that carries any
other token (the leftovers of a holder that died) is killed first.
Processes that use /dev/kfd and carry no token are only reported.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

ENV = "IJ_GPU_JOB"
POLL_S = 0.25


def qdir(state: Path) -> Path:
    """$IJ_AMD_QUEUE names the queue directory when several Unix users share
    one GPU but not one state directory (the two-machine design: `builder`
    and `judge` on the GPU machine)."""
    d = Path(os.environ["IJ_AMD_QUEUE"]) if os.environ.get("IJ_AMD_QUEUE") else state / "gpu"
    (d / "tickets").mkdir(parents=True, exist_ok=True)
    return d


def _now_iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(time.time() if t is None else t))


def _pid_start(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat: the start time, so a reused pid is not
    taken for the old process."""
    try:
        s = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return s.rsplit(")", 1)[1].split()[19]


def _alive(pid: int, start: str | None) -> bool:
    st = _pid_start(pid)
    return st is not None and (start is None or st == start)


def _next_seq(d: Path) -> int:
    with open(d / "seq.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        p = d / "seq"
        try:
            n = int(p.read_text().strip() or 0) + 1
        except (FileNotFoundError, ValueError):
            n = 1
        p.write_text(f"{n}\n")
        return n


def _ticket_live(path: Path) -> bool:
    """True while the waiter that wrote the ticket still holds its flock."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return True
        raise
    finally:
        os.close(fd)
    return False


def waiters(d: Path, prune: bool = True) -> list[dict]:
    out = []
    for p in sorted((d / "tickets").iterdir()):
        if p.name.endswith(".tmp"):
            continue
        if not _ticket_live(p):
            if prune:
                p.unlink(missing_ok=True)
            continue
        try:
            info = json.loads(p.read_text())
        except (OSError, ValueError):
            info = {}
        info["ticket"] = p.name
        out.append(info)
    return out


def lock_held(d: Path) -> bool:
    fd = os.open(d / "lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return True
        raise
    finally:
        os.close(fd)
    return False


def holder(d: Path) -> dict | None:
    """The holder's record, or None when nobody holds the card."""
    if not lock_held(d):
        return None
    try:
        return json.loads((d / "holder.json").read_text())
    except (OSError, ValueError):
        return {"pid": None, "cmd": "?", "since": "?"}


def _environ_token(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    pre = (ENV + "=").encode()
    for kv in raw.split(b"\0"):
        if kv.startswith(pre):
            return kv[len(pre):].decode("ascii", "replace")
    return None


def _pids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return "?"


def _container_token(pid: int, containers: list[dict]) -> str | None:
    """The ij.gpu.job label of the Docker container the process runs in."""
    try:
        cg = open(f"/proc/{pid}/cgroup").read()
    except OSError:
        return None
    for c in containers:
        if c["id"] and c["id"] in cg:
            return c["token"]
    return None


def kfd_users() -> list[dict]:
    """Processes that have /dev/kfd open (any process that uses the GPU
    through ROCm does), with the job token they carry, if any: in their
    environment, or as the label of the queue's container they run in.

    /proc/<pid>/fd of another account is readable by root only, so a plain
    user sees only its own processes. On the GPU machine (IJ_AMD_REAP_SUDO)
    the judge asks the root-owned reaper (`ij-reap --kfd-users`), which sees
    those of builder, judge and cand; if that is refused (builder may not),
    the local view is used."""
    if os.geteuid() != 0 and os.environ.get("IJ_AMD_REAP_SUDO") and os.path.exists(REAPER):
        try:
            r = subprocess.run(["sudo", "-n", REAPER, "--kfd-users"], capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return json.loads(r.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    return kfd_users_local()


def kfd_users_local() -> list[dict]:
    """kfd_users() as far as this process can see /proc."""
    out = []
    me = os.getpid()
    containers = None
    for pid in _pids():
        if pid == me:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(f"/proc/{pid}/fd/{fd}") == "/dev/kfd":
                    tok = _environ_token(pid)
                    if tok is None:
                        if containers is None:
                            containers = labelled_containers()
                        tok = _container_token(pid, containers)
                    out.append({"pid": pid, "cmd": _cmdline(pid)[:160], "token": tok})
                    break
            except OSError:
                continue
    return out


REAPER = "/opt/ij-amd/amd/controller/box/ij-reap"


def _reap_other_users(keep: str, log=None) -> None:
    """Another Unix user's processes cannot be read or killed without root.
    On the GPU machine a root-owned reaper, allowed in sudoers for `builder`
    and `judge`, kills every process and queue container that carries a job
    token other than `keep`."""
    if os.geteuid() == 0 or not os.environ.get("IJ_AMD_REAP_SUDO") or not os.path.exists(REAPER):
        return
    r = subprocess.run(["sudo", "-n", REAPER, "--keep", keep], capture_output=True, text=True, timeout=180)
    if log and r.stdout.strip():
        log(r.stdout.strip())
    if r.returncode != 0 and log:
        log(f"the reaper failed ({r.returncode}): {r.stderr.strip()[:200]}")


def _kill_tokens(pred, why: str, log=None) -> list[int]:
    """SIGKILL every process whose job token satisfies pred(token)."""
    me = os.getpid()
    hit = []
    for pid in _pids():
        if pid == me:
            continue
        tok = _environ_token(pid)
        if tok is not None and pred(tok):
            try:
                os.kill(pid, signal.SIGKILL)
                hit.append(pid)
            except OSError:
                pass
    if hit and log:
        log(f"killed {len(hit)} process(es) {why}: {' '.join(map(str, hit))}")
    return hit


LAUNCHER = "/opt/ij-amd/amd/controller/box/ij-docker"
DOCKER_SOCK = "/var/run/docker.sock"


def docker_direct() -> bool:
    """Whether this account may talk to Docker itself (root, or the docker
    group).  On the GPU machine builder and cand may not: they go through
    the launcher (controller/box/ij-docker), and so does this module."""
    return os.geteuid() == 0 or os.access(DOCKER_SOCK, os.W_OK) or bool(os.environ.get("DOCKER_HOST"))


def via_launcher() -> bool:
    return not docker_direct() and bool(os.environ.get("IJ_AMD_REAP_SUDO")) and os.path.exists(LAUNCHER)


def _parse_jobs(out: str) -> list[dict]:
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append({"id": parts[0], "name": parts[1], "token": parts[2],
                         "owner": parts[3] if len(parts) > 3 and parts[3] != "-" else ""})
    return rows


def labelled_containers() -> list[dict]:
    """Running Docker containers with the label ij.gpu.job (the queue's own)."""
    argv = (["sudo", "-n", LAUNCHER, "jobs"] if via_launcher() else
            ["docker", "ps", "--filter", "label=ij.gpu.job",
             "--format", '{{.ID}} {{.Names}} {{.Label "ij.gpu.job"}} {{.Label "ij.owner"}}'])
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return _parse_jobs(out)


def _rm_labelled_containers(pred, why: str, log=None, launcher_args: list[str] | None = None) -> None:
    """Remove the queue's containers whose token satisfies pred.  Through the
    launcher, which checks that the caller holds the queue, launcher_args
    ("--keep TOKEN" or "--end TOKEN") say the same thing."""
    if via_launcher():
        if not launcher_args:
            return
        r = subprocess.run(["sudo", "-n", LAUNCHER, "reap", *launcher_args], capture_output=True, text=True, timeout=300)
        if log and r.stdout.strip():
            log(f"{r.stdout.strip()} {why}")
        if r.returncode != 0 and log:
            log(f"the launcher could not remove the job containers ({r.returncode}): {r.stderr.strip()[:200]}")
        return
    hit = [c for c in labelled_containers() if pred(c["token"])]
    for c in hit:
        subprocess.run(["docker", "rm", "-f", c["id"]], capture_output=True, timeout=120)
    if hit and log:
        log(f"removed {len(hit)} container(s) {why}: {' '.join(c['name'] for c in hit)}")


def owner_containers(user: str) -> list[dict]:
    """The queue's containers started for `user` (label ij.owner), running."""
    return [c for c in labelled_containers() if c.get("owner") == user]


def rm_owner_containers(user: str, log=None) -> list[dict]:
    """Remove the containers started for `user` (the judge, which may use
    Docker itself, after the engine stopped); returns those still running."""
    hit = owner_containers(user)
    if hit and docker_direct():
        for c in hit:
            subprocess.run(["docker", "rm", "-f", c["id"]], capture_output=True, timeout=120)
        if log:
            log(f"removed {len(hit)} container(s) of {user}: {' '.join(c['name'] for c in hit)}")
        return owner_containers(user)
    return hit


class Hold:
    """The card, held.  Use as a context manager."""

    def __init__(self, state: Path, kind: str, argv: list[str], log=None, quiet=False):
        self.d = qdir(state)
        self.kind = kind
        self.argv = argv
        self.log = log or (lambda m: print(f"[gpu] {m}", file=sys.stderr, flush=True))
        self.quiet = quiet
        self.token = None
        self.nested = False
        self.lock_fd = None
        self.waited_s = 0.0

    def _record(self) -> dict:
        return {"pid": os.getpid(), "pid_start": _pid_start(os.getpid()), "kind": self.kind,
                "cmd": " ".join(self.argv)[:300], "cwd": os.getcwd(), "since": _now_iso(),
                "since_epoch": time.time()}

    def __enter__(self) -> "Hold":
        tok = os.environ.get(ENV)
        if tok:
            h = holder(self.d)
            if h and h.get("token") == tok and _alive(h.get("pid") or -1, h.get("pid_start")):
                self.nested, self.token = True, tok
                if not self.quiet:
                    self.log(f"inside the hold of pid {h['pid']} ({h.get('cmd', '?')[:60]}); not queueing again")
                return self
        t0 = time.monotonic()
        seq = _next_seq(self.d)
        tpath = self.d / "tickets" / f"{seq:012d}-{os.getpid()}"
        tfd = os.open(tpath, os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(tfd, fcntl.LOCK_EX)
        os.write(tfd, json.dumps({**self._record(), "seq": seq}).encode())
        said = False
        try:
            while True:
                ws = waiters(self.d)
                first = ws[0]["ticket"] if ws else None
                if first == tpath.name:
                    lfd = os.open(self.d / "lock", os.O_RDWR | os.O_CREAT, 0o666)
                    try:
                        fcntl.flock(lfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as e:
                        os.close(lfd)
                        if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                            raise
                    else:
                        self.lock_fd = lfd
                        break
                if not said and not self.quiet:
                    h = holder(self.d)
                    ahead = [w for w in ws if w["ticket"] < tpath.name]
                    who = f"pid {h.get('pid')} ({h.get('cmd', '?')[:60]})" if h else "nobody"
                    self.log(f"waiting: held by {who}; {len(ahead)} ahead of you in the queue")
                    said = True
                time.sleep(POLL_S)
        finally:
            tpath.unlink(missing_ok=True)
            os.close(tfd)
        self.waited_s = time.monotonic() - t0
        self.token = secrets.token_hex(8)
        rec = {**self._record(), "token": self.token}
        tmp = self.d / f"holder.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(rec) + "\n")
        os.replace(tmp, self.d / "holder.json")
        _kill_tokens(lambda t: t != self.token, "left over from an earlier holder", self.log)
        _rm_labelled_containers(lambda t: True, "left over from an earlier holder", self.log,
                                ["--keep", self.token])
        _reap_other_users(self.token, self.log)
        if not self.quiet:
            self.log(f"GPU taken after {self.waited_s:.1f} s in the queue")
        return self

    def env(self) -> dict:
        e = dict(os.environ)
        e[ENV] = self.token
        return e

    def __exit__(self, *exc):
        if self.nested:
            return False
        _kill_tokens(lambda t: t == self.token, "of this job still running at its end", self.log)
        _rm_labelled_containers(lambda t: t == self.token, "of this job still running at its end", self.log,
                                ["--end", self.token])
        try:
            (self.d / "holder.json").unlink(missing_ok=True)
        finally:
            if self.lock_fd is not None:
                os.close(self.lock_fd)      # releases the flock
                self.lock_fd = None
        return False


def _pdeathsig():
    """In the child: die with SIGKILL when the parent (the holder) dies."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def run(hold: Hold, argv: list[str], max_s: float | None = None, env: dict | None = None,
        cwd: str | None = None, stdout=None, stderr=None) -> int:
    """Run argv inside the hold in its own process group, with stdio passed
    through.  Returns its exit code (128+signal when killed; 124 on the time
    limit).  On Ctrl-C or SIGTERM the whole group gets SIGTERM, then SIGKILL."""
    e = env if env is not None else hold.env()
    try:
        p = subprocess.Popen(argv, env=e, cwd=cwd, start_new_session=True, preexec_fn=_pdeathsig,
                             stdout=stdout, stderr=stderr)
    except FileNotFoundError:
        print(f"{argv[0]}: command not found", file=sys.stderr)
        return 127
    except PermissionError:
        print(f"{argv[0]}: permission denied", file=sys.stderr)
        return 126
    stop = {"sig": None}

    def on_sig(signum, _frame):
        stop["sig"] = signum

    old = {s: signal.signal(s, on_sig) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    t0 = time.monotonic()
    timed_out = False
    try:
        while True:
            try:
                rc = p.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if stop["sig"] is not None or (max_s and time.monotonic() - t0 > max_s):
                timed_out = stop["sig"] is None
                if timed_out:
                    hold.log(f"time limit of {max_s:.0f} s reached: stopping the job")
                _kill_group(p.pid, signal.SIGTERM)
                try:
                    rc = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _kill_group(p.pid, signal.SIGKILL)
                    rc = p.wait()
                break
    finally:
        _kill_group(p.pid, signal.SIGKILL)       # anything the job left in its group
        for s, h in old.items():
            signal.signal(s, h)
    if timed_out:
        return 124
    return rc if rc >= 0 else 128 - rc


def _kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass

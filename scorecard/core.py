"""Shared vocabulary of the suite: one Outcome per table row, the two evidence
tiers, and the process helpers the judge adapter and the scenarios use.

Nothing in here imports bench code. Code of the judge only ever runs in a
child process, started the way bench/run.sh starts it (see judge.py).
"""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCORECARD = Path(__file__).resolve().parent
# The judge's real code ran from start to finish against a mock server or a
# fixture on this machine.
END_TO_END = "end-to-end"
# The judge's real decision function was applied to inputs the scenario made up.
SIMULATED = "simulated"


@dataclass
class Outcome:
    """One row of the table.

    `tried` is the cheat or the failure in plain words, `expected` what the
    judge must do about it, `observed` what it did. `passed` is judged against
    `expected`; None means the row could not be judged here (no Docker).
    """
    scenario: str
    tried: str
    expected: str
    observed: str
    passed: bool | None
    tier: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - any failure means not up yet
            last = exc
        time.sleep(0.1)
    raise RuntimeError(f"server at {url} never came up: {last}")


@contextlib.contextmanager
def server(argv: list[str], health_url: str, log_path: Path,
           cwd: Path | None = None) -> Iterator[subprocess.Popen]:
    """Start a server in its own process group; always kill the whole group."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(cwd) if cwd else None,
                            start_new_session=True)
    try:
        try:
            wait_http(health_url)
        except RuntimeError:
            if proc.poll() is not None:
                log.flush()
                raise RuntimeError(f"server exited rc={proc.returncode}: "
                                   + log_path.read_text()[-600:])
            raise
        yield proc
    finally:
        kill_group(proc)
        log.close()


def kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def run(argv: list[str], *, timeout: float, env: dict | None = None,
        cwd: Path | None = None, input: str | None = None
        ) -> subprocess.CompletedProcess:
    """subprocess.run in its own process group, killed as a group on timeout.

    Returns a CompletedProcess with returncode 124 on timeout (like timeout(1)),
    because macOS has no timeout binary and several scenarios depend on
    "still running at the deadline" being an observable result.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                            env=env, cwd=str(cwd) if cwd else None, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(input=input, timeout=timeout)
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        kill_group(proc)
        out, err = proc.communicate()
        return subprocess.CompletedProcess(argv, 124, out or "", err or "")


def python() -> str:
    """The bench venv interpreter (Python 3.12) the suite itself runs on."""
    return sys.executable

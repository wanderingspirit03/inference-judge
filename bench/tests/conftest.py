"""Shared fixtures: a mock server on a free port."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_up(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:                              # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError(f"mock server on {port} never came up")


class MockServer:
    def __init__(self, **flags: object) -> None:
        self.port = free_port()
        argv = [sys.executable, str(BENCH / "mock_server.py"),
                "--port", str(self.port)]
        for key, value in flags.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        wait_up(self.port)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="session")
def mock_server():
    server = MockServer(ttft_ms=40, tpot_ms=2)
    yield server
    server.stop()

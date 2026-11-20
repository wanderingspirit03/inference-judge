"""What every command shares: the config, the state directory, the log line
and a way to take the GPU."""
from __future__ import annotations

import sys
import time

import config as C
import gpuqueue


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", file=sys.stderr, flush=True)


def setup():
    cfg = C.load()
    state = C.state_dir(cfg)
    return cfg, state


def hold(state, kind: str, argv: list[str], quiet: bool = False) -> gpuqueue.Hold:
    return gpuqueue.Hold(state, kind, argv, log=lambda m: log(f"gpu: {m}"), quiet=quiet)


def fmt_s(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"

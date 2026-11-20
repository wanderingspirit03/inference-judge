"""The judge's record of submissions: <state>/submissions.jsonl, one JSON
object per line, appended under a lock."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path


def _path(state: Path) -> Path:
    return state / "submissions.jsonl"


def read(state: Path) -> list[dict]:
    try:
        with open(_path(state)) as f:
            return [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        return []


def next_id(state: Path) -> int:
    with open(state / "submissions.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        p = state / "submissions.seq"
        try:
            n = int(p.read_text().strip()) + 1
        except (FileNotFoundError, ValueError):
            n = 1
        p.write_text(f"{n}\n")
        return n


def append(state: Path, rec: dict) -> None:
    with open(state / "submissions.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        with open(_path(state), "a") as out:
            out.write(json.dumps(rec) + "\n")

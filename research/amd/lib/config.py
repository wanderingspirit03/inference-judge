"""The one config of the AMD commands: research/amd/config.json.

$IJ_AMD_CONFIG names another file (the tests use it).  Relative paths in the
file are relative to the directory that holds the file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # research/amd


def path() -> Path:
    return Path(os.environ.get("IJ_AMD_CONFIG") or HERE / "config.json")


def load() -> dict:
    p = path()
    cfg = json.loads(p.read_text())
    cfg["_dir"] = str(p.resolve().parent)
    cfg["_file"] = str(p.resolve())
    return cfg


def resolve(cfg: dict, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(cfg["_dir"]) / q


def state_dir(cfg: dict) -> Path:
    d = Path(os.environ.get("IJ_AMD_STATE") or cfg["state_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d

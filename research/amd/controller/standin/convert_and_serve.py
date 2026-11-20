#!/usr/bin/env python3
"""A stand-in engine that makes its own weights at start-up (the e2e's test of
"an agent's engine brings its own smaller weights").  Stdlib.

  convert_and_serve.py <port>

It reads every file of the model folder (/models/standin inside a judged
container, /srv/models/standin when serve runs as cand on the host, as on the
stand-in machine, which has no Docker), writes a derived "converted weights"
file under its own temporary directory ($TMPDIR, else /tmp), and serves the
toy model from that file only (fake_engine.py --weights).  This is path 1 of
measurements/2026-11-17/NOTES.md, "Smaller weights: what a judged engine can
reach": conversion at start-up from the read-only models.

Before serving it writes $HOME/convert-report.json with what it saw: the model
folder, whether it could write there (it must not; then it exits 3), the
converted file, and which of the probe paths it could read (the e2e puts
files there as builder, to record what a judged engine on the host sees)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
FOLDERS = ("/models/standin", "/srv/models/standin")
PROBES = ("/var/lib/ij-amd/builder/e2e-probe", "/tmp/ij-e2e-probe", "/home/builder/work/e2e-probe")


def readable(p: str) -> bool:
    try:
        with open(p, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False


def main(argv: list[str]) -> int:
    port = argv[0]
    src = next((d for d in FOLDERS if os.path.isdir(d)), None)
    if src is None:
        print("convert_and_serve: no model folder", file=sys.stderr)
        return 2
    h, n = hashlib.sha256(), 0
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                data = f.read()
            h.update(name.encode() + b"\0" + data)
            n += len(data)
    try:
        fd, probe = tempfile.mkstemp(dir=src)
        os.close(fd)
        os.unlink(probe)
        models_writable = True
    except OSError:
        models_writable = False
    out_dir = os.path.join(os.environ.get("TMPDIR") or "/tmp", "converted")
    os.makedirs(out_dir, exist_ok=True)
    weights = os.path.join(out_dir, "weights.json")
    with open(weights, "w") as f:
        json.dump({"source": src, "source_sha256": h.hexdigest(), "source_bytes": n, "damage": 0.0}, f)
    report = {"models": src, "models_bytes_read": n, "models_writable": models_writable, "weights": weights,
              "weights_bytes": os.path.getsize(weights), "probes_readable": {p: readable(p) for p in PROBES}}
    home = os.environ.get("HOME") or "/tmp"
    with open(os.path.join(home, "convert-report.json"), "w") as f:
        json.dump(report, f)
    if models_writable:
        print("convert_and_serve: the model folder is writable, which a judged engine must not find", file=sys.stderr)
        return 3
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, "fake_engine.py"), port, "--weights", weights])
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

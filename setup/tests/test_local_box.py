"""The local box, end to end (setup/local_box/README.md).  Slow (minutes) and
needs a running Docker that allows --privileged, so it is opt-in:

    cd bench && IJ_LOCALBOX=1 .venv/bin/python -m pytest -m slow ../setup/tests/test_local_box.py

One command, one assertion: run_local_box.sh exits 0 only if every check of
setup/local_box/scenario.py passed; its per-check PASS/FAIL lines are in the
captured output and in <out>/summary.json.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow


def docker_ok() -> bool:
    return shutil.which("docker") is not None and \
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.skipif(os.environ.get("IJ_LOCALBOX") != "1", reason="opt-in: IJ_LOCALBOX=1")
def test_local_box_end_to_end(tmp_path):
    if not docker_ok():
        pytest.skip("needs a running docker daemon")
    out = tmp_path / "box"
    r = subprocess.run(["bash", str(REPO / "setup" / "local_box" / "run_local_box.sh"), "--out", str(out)],
                       capture_output=True, text=True, timeout=3600)
    summary = out / "summary.json"
    fails = []
    if summary.exists():
        fails = [c for c in json.loads(summary.read_text())["checks"] if not c["ok"]]
    assert r.returncode == 0 and summary.exists(), (r.stdout[-6000:] + r.stderr[-2000:], fails)

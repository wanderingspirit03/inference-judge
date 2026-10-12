"""5. Isolation: can code inside a scored container reach the network or leave
a file behind on the host?

The judge says which `docker run` flags a scored launch uses (from
`run.sh --dry-run`). They are replayed here with python:3.12-slim and a probe
as the command: it tries to open a connection to the internet and to write a
file under /out. `--gpus`, `-p`, `--publish`, `--name`, `--label`, `--device`,
`--runtime` and `--entrypoint` are dropped (no accelerator here; the port and
name would clash; the probe is the command);
every bind-mount source is replaced by a fresh temp dir, so nothing of the
host is exposed. write=ok only if the file shows up on the host side of /out.

Needs Docker and the python:3.12-slim image; without them the row is SKIP.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from core import END_TO_END, Outcome
from core import run as sh

IMAGE = "python:3.12-slim"
DROP = {"--gpus", "-p", "--publish", "--name", "--label", "--device", "--runtime",
        "--entrypoint"}
PROBE = r"""
import json, socket
res = {}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close(); res["net"] = "ok"
except Exception as e:
    res["net"] = "blocked"; res["net_err"] = type(e).__name__
try:
    open("/out/stolen.jsonl", "w").write(json.dumps({"secret": "x"})); res["write"] = "ok"
except Exception as e:
    res["write"] = "blocked"; res["write_err"] = type(e).__name__
print("EXFIL net=%s write=%s" % (res["net"], res["write"]))
print(json.dumps(res))
"""


def translate(flags: list[str], tmp: Path) -> tuple[list[str], Path | None, list[str]]:
    """Scored-launch flags -> runnable local flags, the host dir behind /out,
    and what was dropped."""
    out, dropped, out_host, i = [], [], None, 0
    while i < len(flags):
        f = flags[i]
        key = f.split("=", 1)[0]
        has_inline = "=" in f
        val = None if has_inline else (flags[i + 1] if i + 1 < len(flags) and
                                        not flags[i + 1].startswith("-") else None)
        step = 1 if has_inline or val is None else 2
        if key == "-d":
            dropped.append(f)
        elif key in DROP:
            dropped.append(f if has_inline else f"{f} {val}")
        elif key in ("-v", "--volume"):
            spec = f.split("=", 1)[1] if has_inline else val
            parts = spec.split(":")
            dst, mode = parts[1], (parts[2] if len(parts) > 2 else "rw")
            host = Path(tempfile.mkdtemp(prefix="mnt-", dir=tmp))
            if dst == "/out":
                out_host = host
            out += ["-v", f"{host}:{dst}:{mode}"]
        elif key == "--mount":
            spec = f.split("=", 1)[1] if has_inline else val
            kv = dict(p.split("=", 1) if "=" in p else (p, "true") for p in spec.split(","))
            if kv.get("type", "volume") == "bind":
                host = Path(tempfile.mkdtemp(prefix="mnt-", dir=tmp))
                kv["source" if "source" in kv else "src"] = str(host)
                if kv.get("target", kv.get("dst", kv.get("destination"))) == "/out":
                    out_host = host
            out += ["--mount", ",".join(k if v == "true" and k in ("readonly", "ro")
                                        else f"{k}={v}" for k, v in kv.items())]
        else:
            out += [f] if has_inline or val is None else [f, val]
        i += step
    return out, out_host, dropped


def run(judge, ctx) -> list:
    row = ("5", "code in a scored container connecting out and writing a file", "block both (net=blocked write=blocked)")
    flags = judge.scored_launch_flags()
    if not shutil.which("docker"):
        return [Outcome(*row, "docker not available: not run", None, END_TO_END,
                        {"scored_flags": flags})]
    if sh(["docker", "image", "inspect", IMAGE], timeout=60).returncode != 0:
        return [Outcome(*row, f"docker image {IMAGE} not present (or Docker not "
                        "running): not run", None, END_TO_END, {"scored_flags": flags})]
    tmp = Path(tempfile.mkdtemp(prefix="exfil-", dir=ctx.work))
    local, out_host, dropped = translate(flags, tmp)
    cname = f"scorecard-exfil-{uuid.uuid4().hex[:8]}"
    argv = ["docker", "run", "--rm", "--name", cname, *local, "--entrypoint", "python3",
            IMAGE, "-c", PROBE]
    try:
        p = sh(argv, timeout=240)
    finally:
        sh(["docker", "rm", "-f", cname], timeout=30)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("EXFIL")), "")
    if not line:
        return [Outcome(*row, f"the probe did not run (rc={p.returncode})", None,
                        END_TO_END, {"argv": argv, "stderr": p.stderr[-600:]})]
    net = line.split("net=")[1].split()[0]
    on_host = bool(out_host and (out_host / "stolen.jsonl").exists())
    write = "ok" if on_host else "blocked"
    return [Outcome(*row, f"net={net} write={write}",
                    net == "blocked" and write == "blocked", END_TO_END,
                    {"scored_flags": flags, "replayed_flags": local, "dropped": dropped,
                     "probe_line": line, "out_mounted": out_host is not None})]

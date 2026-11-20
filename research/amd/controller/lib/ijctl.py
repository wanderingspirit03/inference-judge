"""What the controller's programs share: the config, the GPU machine's state
and how to reach it over ssh.  Stdlib only.

The config is /etc/ij-amd/controller.json ($IJ_CTL_CONFIG overrides it; the
tests use that).  The GPU machine's state is <ctl_dir>/box.json, written by
the driver and readable by everyone:
  {"state": "none" | "creating" | "setting-up" | "ready" | "destroying" | "stopped",
   "since": epoch, "why": text, "id": droplet id, "name": ..., "ip": ..., "port": 22}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

CONFIG = Path(os.environ.get("IJ_CTL_CONFIG") or "/etc/ij-amd/controller.json")
ALIAS = "ij-box"          # the GPU machine's name in known_hosts, whatever its address


def load() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except FileNotFoundError:
        raise SystemExit(f"the controller is not installed: {CONFIG} is missing") from None


def ctl_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["ctl_dir"])


def box_state(cfg: dict) -> dict:
    try:
        return json.loads((ctl_dir(cfg) / "box.json").read_text())
    except (FileNotFoundError, ValueError):
        return {"state": "none", "since": None, "why": "the driver has not run yet"}


def write_box_state(cfg: dict, **kw) -> dict:
    """Driver only.  Keeps the old fields unless they are given; a new state
    gets a new `since`."""
    old = box_state(cfg)
    new = dict(old)
    if kw.get("state") != old.get("state"):
        new["since"] = time.time()
    new.update(kw)
    p = ctl_dir(cfg) / "box.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(new, indent=1))
    os.chmod(tmp, 0o644)
    os.replace(tmp, p)
    return new


def ago(t: float | None) -> str:
    if not t:
        return "unknown"
    s = int(time.time() - t)
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t))
    if s < 3600:
        return f"{when} ({s // 60} min ago)"
    return f"{when} ({s // 3600} h {(s % 3600) // 60:02d} min ago)"


def describe(st: dict) -> str:
    s = st.get("state")
    what = {"none": "there is no GPU machine",
            "creating": "a GPU machine is being created",
            "setting-up": "the GPU machine is being set up (weights download, reference)",
            "ready": "the GPU machine is ready",
            "destroying": "the GPU machine is being destroyed",
            "stopped": "the run is stopped; no GPU machine will be created"}.get(s, f"GPU machine state {s!r}")
    line = f"{what}, since {ago(st.get('since'))}"
    if st.get("why"):
        line += f": {st['why']}"
    return line


def ssh_base(cfg: dict, key: str, user: str, st: dict) -> list[str]:
    """ssh options that accept only the host key the driver generated.
    -F /dev/null: no system or user ssh config (Ubuntu's sends LANG and
    LC_*, which the GPU machine may not have and which a `-o SendEnv=-...`
    cannot remove, as the system file is read after it)."""
    return ["ssh", "-F", "/dev/null", "-i", key, "-p", str(st.get("port") or 22),
            "-o", f"HostKeyAlias={ALIAS}", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={ctl_dir(cfg) / 'box_known_hosts'}",
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "UpdateHostKeys=no",
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            f"{user}@{st['ip']}"]


def rsync_ssh(cfg: dict, key: str, st: dict) -> str:
    """The -e argument of rsync, with the same options."""
    import shlex
    return " ".join(shlex.quote(x) for x in ssh_base(cfg, key, "x", st)[:-1])


def wait_ready(cfg: dict, out, poll_s: float = 15.0, say_every_s: float = 300.0) -> dict:
    """Wait until the GPU machine is ready; say why while waiting."""
    st = box_state(cfg)
    if st.get("state") == "ready":
        return st
    print(f"[ij] waiting: {describe(st)}. GPU commands wait until it is ready "
          f"(Ctrl-C stops waiting).", file=out, flush=True)
    last = time.monotonic()
    last_state = st.get("state")
    while True:
        time.sleep(poll_s)
        st = box_state(cfg)
        if st.get("state") == "ready":
            print("[ij] the GPU machine is ready", file=out, flush=True)
            return st
        if st.get("state") != last_state or time.monotonic() - last > say_every_s:
            print(f"[ij] still waiting: {describe(st)}", file=out, flush=True)
            last, last_state = time.monotonic(), st.get("state")

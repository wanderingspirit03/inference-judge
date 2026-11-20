"""The engine under test: build it, start it, wait for it, ask it, stop it.

The engine is a directory with an executable `serve`; `serve <port>` must
answer HTTP on 127.0.0.1:<port>:
  GET  /v1/models        200 when ready; the first "id" is the model name used
  POST /v1/completions   {"model", "prompt": [token ids], "max_tokens",
                          "temperature": 0, "logprobs": 5,
                          "return_tokens_as_token_ids": true, "stream": false}
       -> choices[0].logprobs = {"tokens": ["token_id:<n>", ...],
                                 "token_logprobs": [...], "top_logprobs": [{"token_id:<n>": lp}, ...]}
          choices[0].finish_reason = "length" or "stop"
       or, with no logprobs at all, choices[0].token_ids = [<n>, ...]: the
       judge uses the engine's logprobs for no verdict (only for a figure it
       reports), so token ids alone are enough.  The request still carries
       "logprobs": 5; an engine may ignore it.
Every request the judge sends has exactly this shape: the ones it times are
the ones it checks.  If the directory has an executable `build`, it runs first.

With `user` (./submit on the GPU machine: config submit.engine_user), the
source is copied into that account's home and build and serve run as that
account through sudo, so they cannot read the judge's files.  Its processes
are killed by uid when the engine stops (kill_user).
"""
from __future__ import annotations

import os
import pwd
import subprocess
import time
from pathlib import Path

import client
import gpuqueue as Q

SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class EngineFailed(Exception):
    pass


def _tail(p: Path, n: int = 25) -> str:
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def as_user(user: str | None, hold, cwd: Path, argv: list[str]) -> list[str]:
    """argv to run `argv` in `cwd` as `user` through sudo, with a clean
    environment that holds only the GPU job token and the basics."""
    if not user:
        return argv
    home = pwd.getpwnam(user).pw_dir
    env = [f"PATH={SAFE_PATH}", f"HOME={home}", f"USER={user}", f"LOGNAME={user}", "LANG=C.UTF-8",
           f"TMPDIR={home}/tmp", f"{Q.ENV}={hold.token}", f"IJ_ENGINE_IMAGE={image_name(user)}"]
    return ["sudo", "-n", "-u", user, "--", "/usr/bin/env", "-i", *env,
            "/bin/sh", "-c", 'cd "$1" && shift && exec "$@"', "sh", str(cwd), *argv]


def stage(user: str, src: Path, log) -> Path:
    """Copy the engine's source into a fresh directory of `user`'s home and
    return it.  The judge's copy stays where only the judge reads it."""
    home = Path(pwd.getpwnam(user).pw_dir)
    dest = home / "run" / "src"
    r = subprocess.run(["sudo", "-n", "-u", user, "--", "/bin/sh", "-c",
                        'umask 077; rm -rf "$1/run" "$1/tmp" && mkdir -p "$1/run/src" "$1/tmp"', "sh", str(home)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise EngineFailed(f"could not prepare the engine account {user}: {r.stderr.strip()[:300]}")
    tar = subprocess.Popen(["tar", "-C", str(src), "-cf", "-", "."], stdout=subprocess.PIPE)
    r = subprocess.run(["sudo", "-n", "-u", user, "--", "tar", "-xf", "-", "-C", str(dest)], stdin=tar.stdout,
                       capture_output=True, text=True, timeout=600)
    tar.stdout.close()
    if tar.wait() != 0 or r.returncode != 0:
        raise EngineFailed(f"could not copy the source to the engine account {user}: {r.stderr.strip()[:300]}")
    log(f"engine: source copied to {dest} (runs as {user})")
    return dest


def user_pids(user: str) -> list[int]:
    uid = pwd.getpwnam(user).pw_uid
    out = []
    for pid in Q._pids():
        try:
            if os.stat(f"/proc/{pid}").st_uid == uid:
                out.append(pid)
        except OSError:
            pass
    return out


def kill_user(user: str, log=None) -> list[int]:
    """SIGKILL every process of `user` and wait until none is left; returns
    the pids still alive after 20 s (an empty list when all are gone)."""
    t0 = time.monotonic()
    first = user_pids(user)
    while True:
        subprocess.run(["sudo", "-n", "-u", user, "--", "pkill", "-KILL", "-u", user],
                       capture_output=True, timeout=60)
        left = user_pids(user)
        if not left or time.monotonic() - t0 > 20:
            break
        time.sleep(0.5)
    if first and log:
        log(f"engine: killed {len(first)} process(es) of {user}" + (f"; still alive: {left}" if left else ""))
    return left


def _runnable(p: Path) -> bool:
    return p.is_file() and bool(p.stat().st_mode & 0o100)


def build(cfg: dict, hold, src: Path, logdir: Path, log, user: str | None = None,
          run_dir: Path | None = None) -> float | None:
    """src: the judge's copy (looked at); run_dir: the copy that runs (the
    engine account's, from stage(); src when there is no such account)."""
    run_dir = run_dir or src
    name = cfg["engine"]["build"]
    if not _runnable(src / name):
        return None
    log(f"engine: running {name} (limit {cfg['engine']['build_timeout_s']} s)")
    t0 = time.monotonic()
    with open(logdir / "build.log", "wb") as f:
        rc = Q.run(hold, as_user(user, hold, run_dir, [str(run_dir / name)]), max_s=cfg["engine"]["build_timeout_s"],
                   env=None if user else {**hold.env(), "IJ_ENGINE_IMAGE": image_name(None)},
                   cwd=None if user else str(run_dir), stdout=f, stderr=subprocess.STDOUT)
    if user:
        kill_user(user, log)                 # a build leaves nothing running
    if rc != 0:
        raise EngineFailed(f"./build exited with {rc}; its last lines:\n{_tail(logdir / 'build.log')}")
    return time.monotonic() - t0


class Server:
    def __init__(self, cfg: dict, hold, src: Path, logdir: Path, log, user: str | None = None,
                 run_dir: Path | None = None):
        self.cfg, self.hold, self.src, self.logdir, self.log, self.user = cfg, hold, src, logdir, log, user
        self.run_dir = run_dir or src
        self.port = cfg["engine"]["port"]
        self.url = f"http://127.0.0.1:{self.port}"
        self.top = cfg["check"]["engine_top"]
        self.p = None
        self.model = None
        self.ready_s = None

    def answers_on_port(self) -> bool:
        try:
            client.get_json(self.url, self.cfg["engine"]["health_path"], timeout=2)
            return True
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> float:
        name = self.cfg["engine"]["serve"]
        if not _runnable(self.src / name):
            raise EngineFailed(f"no executable {name} in the engine's directory")
        if self.answers_on_port():
            raise EngineFailed(f"port {self.port} already answers before the engine started: something else serves there")
        self.log(f"engine: starting {name} {self.port}")
        self.logf = open(self.logdir / "serve.log", "wb")
        argv = as_user(self.user, self.hold, self.run_dir, [str(self.run_dir / name), str(self.port)])
        self.p = subprocess.Popen(argv, cwd=None if self.user else str(self.run_dir),
                                  env=None if self.user else {**self.hold.env(), "IJ_ENGINE_IMAGE": image_name(None)},
                                  start_new_session=True, preexec_fn=Q._pdeathsig,
                                  stdout=self.logf, stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        limit = self.cfg["engine"]["health_timeout_s"]
        while time.monotonic() - t0 < limit:
            if self.p.poll() is not None:
                raise EngineFailed(f"./serve exited with {self.p.returncode} before it answered; "
                                   f"its last lines:\n{_tail(self.logdir / 'serve.log')}")
            try:
                doc = client.get_json(self.url, self.cfg["engine"]["health_path"], timeout=3)
                data = doc.get("data") or []
                self.model = data[0]["id"] if data else None
                self.ready_s = time.monotonic() - t0
                self.log(f"engine: ready after {self.ready_s:.0f} s (model id {self.model!r})")
                return self.ready_s
            except Exception:  # noqa: BLE001 - not ready yet
                time.sleep(2)
        raise EngineFailed(f"the engine did not answer GET {self.cfg['engine']['health_path']} within {limit} s")

    def ask(self, ids: list[int], max_tokens: int) -> dict:
        """One request in the judge's one shape.  Returns {tokens, lps, finish,
        seconds (send to full answer, on the judge's clock), prompt_tokens}."""
        body = client.build_body(self.model, "completions", ids, max_tokens, self.top, stream=False)
        r = client.request(self.url, "completions", body, timeout=self.cfg["engine"]["request_timeout_s"])
        return {"tokens": r["tokens"], "lps": r["lps"], "finish": r["finish"],
                "seconds": r["t_end"] - r["t0"], "prompt_tokens": len(ids)}

    def stop(self) -> list[int]:
        """Stop the engine.  As another user: kill every process of that user
        and remove the containers started for it (the reference profile of
        the launcher runs as root in its container, so killing by uid does
        not reach it); returns what survived, pids and container names (none,
        normally)."""
        left = []
        if self.p is not None:
            Q._kill_group(self.p.pid, 15)        # as another user this reaches sudo, which passes it on
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                Q._kill_group(self.p.pid, 9)
                self.p.wait()
            Q._kill_group(self.p.pid, 9)
            self.logf.close()
            self.p = None
        if self.user:
            left = kill_user(self.user, self.log)
            left += [f"container {c['name']}" for c in Q.rm_owner_containers(self.user, self.log)]
        return left


def image_build(cfg: dict, hold, src: Path, logdir: Path, log, user: str | None = None) -> float | None:
    """When the engine's tree has a Dockerfile, build its image before
    `build` runs: through the launcher (controller/box/ij-docker), with no
    network in RUN steps.  As builder (./check) the image is
    ij-builder/engine; for a judged run the judge builds ij-cand/engine from
    its own copy (cand cannot build: a build after the prompts arrive would
    be traffic the firewall does not see).  `build` and `serve` find the name
    in $IJ_ENGINE_IMAGE.  None when there is no Dockerfile."""
    if not (src / "Dockerfile").is_file():
        return None
    if not os.path.exists(Q.LAUNCHER):
        raise EngineFailed("the engine has a Dockerfile, but this machine has no Docker launcher")
    argv = ["sudo", "-n", Q.LAUNCHER, "build"] + (["--for", user] if user else []) + [str(src)]
    log(f"engine: building the image from the Dockerfile (limit {cfg['engine']['build_timeout_s']} s, no network)")
    t0 = time.monotonic()
    with open(logdir / "image.log", "wb") as f:
        rc = Q.run(hold, argv, max_s=cfg["engine"]["build_timeout_s"], stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise EngineFailed(f"the image build exited with {rc}; its last lines:\n{_tail(logdir / 'image.log')}")
    return time.monotonic() - t0


FIREWALL = "/opt/ij-amd/amd/controller/box/ij-fw"


def firewall_problem(argv: list[str] | None = None) -> str | None:
    """None when cand's firewall rule is in place (controller/box/ij-fw
    check, through sudo); otherwise why not.  Fails closed: a missing
    checker, a sudo refusal or any output but "ok" is a problem."""
    argv = argv or ["sudo", "-n", FIREWALL, "check"]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"the firewall check could not run: {e}"[:300]
    if r.returncode != 0 or r.stdout.strip() != "ok":
        return (f"the firewall rule for the engine account is not in place (exit {r.returncode}): "
                f"{(r.stdout + r.stderr).strip()[:300]}")
    return None


def image_name(user: str | None) -> str:
    return f"ij-{user or pwd.getpwuid(os.getuid()).pw_name}/engine:latest"

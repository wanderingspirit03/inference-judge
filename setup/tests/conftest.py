"""Shared harness for the ops/security script tests.

SAFETY. The scripts under test (setup/bootstrap.sh, setup/harden_research.sh, ...)
can ssh into boxes, call cloud tools and run sudo. They are NEVER run against the
real commands. Every run goes through `run_script`, which

  * builds the environment from scratch (nothing leaks in from the caller),
  * puts a directory of fake commands first on PATH ("<shims>:/usr/bin:/bin"),
    where each fake only appends its argv to a log and exits with a chosen code,
  * points HOME and CLOUDSDK_CONFIG at empty temp dirs,
  * asserts, before the script starts, that gcloud / ssh / sudo / ... resolve to
    the fakes.

The fakes are the test adapters at the seam to the true external commands;
tests drive the scripts only through their command line and read the fakes' logs.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# IJ_TEST_REPO lets the same tests run against another checkout (still shimmed).
REPO = Path(os.environ.get("IJ_TEST_REPO") or Path(__file__).resolve().parents[2])
SHIMMED = ("gcloud", "gsutil", "rsync", "ssh", "scp", "sudo", "docker", "curl", "pgrep", "python3")

# One fake serves every name. SHIM_RC_<name>[_<arg2>] picks the exit code,
# SHIM_OUT_<name>[_<arg2>] the stdout, SHIM_BYTES_<name> makes rsync "download"
# that many bytes into its last argument (the destination directory).
SHIM_BODY = r"""#!/bin/bash
name=$(basename "$0")
sub=${2:-}
printf '%s\n' "$name $*" >> "$SHIM_LOG"
if [ "$name" = rsync ] && [ "${SHIM_RSYNC_BYTES:-0}" -gt 0 ] && [ -z "${SHIM_RC_rsync:-}" ]; then
  dest=${!#}
  mkdir -p "$dest" && head -c "$SHIM_RSYNC_BYTES" /dev/zero > "$dest/payload.bin"
fi
out_sub="SHIM_OUT_${name}_${sub}"; out_name="SHIM_OUT_${name}"
[ -n "${!out_sub:-}" ] && printf '%s\n' "${!out_sub}"
[ -z "${!out_sub:-}" ] && [ -n "${!out_name:-}" ] && printf '%s\n' "${!out_name}"
rc_sub="SHIM_RC_${name}_${sub}"; rc_name="SHIM_RC_${name}"
exit "${!rc_sub:-${!rc_name:-0}}"
"""


@dataclass
class Sandbox:
    root: Path
    shims: Path
    log: Path
    home: Path
    extra: dict = field(default_factory=dict)

    # --- calls the fakes saw -------------------------------------------------
    def calls(self, name: str | None = None) -> list[str]:
        if not self.log.exists():
            return []
        lines = self.log.read_text().splitlines()
        return [l for l in lines if name is None or l.split(" ", 1)[0] == name]

    # --- environment ---------------------------------------------------------
    def env(self, **overrides: str) -> dict[str, str]:
        e = {
            "PATH": f"{self.shims}:/usr/bin:/bin",
            "HOME": str(self.home),
            "CLOUDSDK_CONFIG": str(self.root / "cloudsdk"),
            "SHIM_LOG": str(self.log),
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        }
        e.update(self.extra)
        e.update(overrides)
        return e

    def assert_shimmed(self, env: dict[str, str]) -> None:
        for name in SHIMMED:
            r = subprocess.run(["/bin/bash", "-c", f"command -v {name}"], env=env,
                               capture_output=True, text=True)
            assert r.stdout.strip() == str(self.shims / name), \
                f"{name} resolves to {r.stdout.strip()!r}, not the shim; refusing to run"

    def run(self, argv: list[str], timeout: int = 60, cwd: Path | None = None,
            **env_overrides: str) -> subprocess.CompletedProcess:
        env = self.env(**env_overrides)
        self.assert_shimmed(env)
        return subprocess.run(["/bin/bash", *argv], env=env, cwd=cwd or self.root,
                              capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def sb(tmp_path: Path) -> Sandbox:
    shims = tmp_path / "shims"
    shims.mkdir()
    for name in SHIMMED:
        p = shims / name
        p.write_text(SHIM_BODY)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "cloudsdk").mkdir()
    return Sandbox(root=tmp_path, shims=shims, log=tmp_path / "shim.log", home=home)


def make_repo(sb: Sandbox, with_git: bool = True) -> Path:
    """A throwaway copy of the files the scripts need, laid out like the real repo."""
    repo = sb.root / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "setup").mkdir()
    shutil.copytree(REPO / "tools", repo / "tools", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("bootstrap.sh", "check.sh", "target.sh", "tpu.sh", "gpu.sh"):
        shutil.copy(REPO / "setup" / f, repo / "setup" / f)
    (repo / "bench").mkdir()
    # bootstrap.sh refuses an agent that has no adapter, so the adapters travel with it.
    shutil.copytree(REPO / "research" / "adapters", repo / "research" / "adapters",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if with_git:
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=sb.env())
        (repo / "README").write_text("x\n")
        subprocess.run(["git", "add", "README"], cwd=repo, check=True, env=sb.env())
        subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True, env=sb.env())
    return repo


# A one-GPU target whose engine is built here (vllm on inference-judge-base): the case
# that needs the engine pins and runs the gpu steps.  No such target ships yet.
GPU_BUILT_TARGET = """\
id: gpu-built
runnable: true
accelerator: {kind: gpu, chips: 1, hosts: 1}
model:
  hf_id: example/some-model
  revision: 0123456789abcdef0123456789abcdef01234567
  local_dir: /opt/models/some-model
  served_name: some-model
  tokenizer: /opt/models/some-model/tokenizer.json
engine: {name: vllm, image: "inference-judge-base:latest", docker_flags: [], port: 8000, sources: [vllm, sglang]}
metric:
  kind: tps
  isl: 256
  osl: 64
  concurrency: [4]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 2
  window_s: 10
  design: abba
  blocks: 2
budgets: {build_s: 600, compile_s: 120, health_s: 120, run_s: 2400}
reference: {dir: "reference/gpu-built"}
quality: {allow_missing_logprobs: false}
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs (the local box)")

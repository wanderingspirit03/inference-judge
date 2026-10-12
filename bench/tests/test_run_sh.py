"""run.sh end to end, without a GPU.

--dry-run proves the Docker path is wired correctly by printing it.
--local proves the measurement path works by running launch.sh on this host,
where it starts mock_server.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent
# everything run.sh needs next to it (the ruler_hash files among them)
RULER_FILES = ("harness.py", "metrics.py", "quality2.py", "stats.py",
               "tokenizer.py", "target.py", "compare.py", "run.sh",
               "mock_server.py")
MOCK_LAUNCH = BENCH.parent / "research" / "experiments" / "baseline-mock" / "launch.sh"


MOCK_ENGINE = 'engine: {name: mock, image: "inference-judge-mock:latest", docker_flags: [], port: 8000}'
TWO_TREES = MOCK_ENGINE.replace("{", "{sources: [vllm, sglang], ")


def make_bench_root(tmp_path: Path, allow_baseline: bool = False) -> Path:
    root = tmp_path / "bench_root"
    root.mkdir()
    for name in RULER_FILES:
        shutil.copy(BENCH / name, root / name)
    shutil.copytree(BENCH / "canary", root / "canary")
    shutil.copytree(BENCH.parent / "targets", root / "targets")
    # The shipped mock target lists no engine tree. These tests give their copy
    # two, so that the copies, the snapshot and the mounts are exercised.
    mock = root / "targets" / "cpu-mock.yaml"
    assert MOCK_ENGINE in mock.read_text()
    mock.write_text(mock.read_text().replace(MOCK_ENGINE, TWO_TREES))
    if allow_baseline:
        # what bootstrap drops next to run.sh while it runs the two baselines
        (root / ".allow-baseline").write_text("bootstrap\n")
    return root


def bench_env(root: Path, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update({
        "BENCH_ROOT": str(root),
        "BENCH_PY": sys.executable,
        # the target in the repository; the end-to-end tests name their own
        "BENCH_TARGET": "cpu-mock",
        "HARD_TIMEOUT_S": "600",
    })
    env.pop("SUDO_USER", None)
    env.update(extra_env or {})
    return env


def run_sh(root: Path, args: list[str], extra_env: dict | None = None,
           cwd: Path | None = None, timeout: float = 600):
    return subprocess.run(["bash", str(root / "run.sh"), *args],
                          capture_output=True, text=True,
                          env=bench_env(root, extra_env),
                          cwd=str(cwd) if cwd else None, timeout=timeout)


def results_of(root: Path, exp_id: str) -> dict:
    return json.loads((root / "results" / exp_id / "results.json").read_text())


# The mock engine with windows of a second or two. Its goodput is
# deterministic, so the verdicts are too.
TPS_TARGET = """\
id: tps-test
runnable: true
accelerator: {kind: none, chips: 1, hosts: 1}
model: {served_name: mock, revision: "0", local_dir: "", tokenizer: "ws"}
engine: {sources: [vllm, sglang], name: mock, image: "inference-judge-mock:latest", docker_flags: [], port: 8000}
metric:
  kind: tps
  isl: 64
  osl: 32
  concurrency: [4, 8]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 0.3
  window_s: 1.5
  design: abba
  blocks: 2
budgets: {build_s: 600, compile_s: 60, health_s: 60, run_s: 1800}
reference: {dir: "reference/tps-test"}
quality: {allow_missing_logprobs: false}
"""


def local_bench(tmp_path: Path, *exp_ids: str, allow_baseline: bool = False):
    """A bench root for --local runs on TPS_TARGET and their environment.
    Every experiment named is the mock's honest launch.sh."""
    from conftest import free_port
    base = Path(os.path.realpath(tmp_path))
    root = make_bench_root(base, allow_baseline)
    (root / "targets" / "tps-test.yaml").write_text(TPS_TARGET)
    for exp_id in exp_ids:
        exp = base / "src" / "research" / "experiments" / exp_id
        exp.mkdir(parents=True)
        shutil.copy(MOCK_LAUNCH, exp / "launch.sh")
    return root, {"BENCH_TARGET": "tps-test",
                  "RESEARCH_ROOT": str(base / "src" / "research"),
                  "IJ_SRC_ROOT": str(base / "src"), "PORT": str(free_port())}


# ------------------------------------------------------------------ basics --
def test_help_mentions_the_phases():
    out = subprocess.run(["bash", str(BENCH / "run.sh"), "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for word in ("canary", "warm sweep", "blocks", "screen", "quality2",
                 "verdict", "--local"):
        assert word in out.stdout


def test_bad_id_is_refused(tmp_path):
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["--dry-run", "Bad_ID"])
    assert out.returncode != 0
    assert "bad id" in out.stderr


def test_existing_result_is_refused(tmp_path):
    root = make_bench_root(tmp_path)
    (root / "results" / "already-here").mkdir(parents=True)
    out = run_sh(root, ["already-here"])
    assert out.returncode != 0
    assert "already exists" in out.stderr


# ----------------------------------------------------------------- dry run --
# What a GPU target adds to the launch: the chips, the shared memory, the
# model directory, and its own build budget.
GPU_TARGET = (TPS_TARGET.replace("tps-test", "gpu-test")
              .replace("accelerator: {kind: none,", "accelerator: {kind: gpu,")
              .replace('local_dir: ""', 'local_dir: "/opt/models/mock"')
              .replace("port: 8000}", "port: 8000, shm_size: 16g}")
              .replace("build_s: 600", "build_s: 1200"))


def test_dry_run_prints_the_whole_docker_path(tmp_path):
    root = make_bench_root(tmp_path)
    (root / "targets" / "gpu-test.yaml").write_text(GPU_TARGET)
    out = run_sh(root, ["--dry-run", "exp-alpha"], {"BENCH_TARGET": "gpu-test"})
    assert out.returncode == 0, out.stderr
    text = out.stdout
    for expected in ("timeout 1200s docker build", "docker run", "--gpus all",
                     "/opt/models:ro", "--shm-size 16g",
                     "poll http://127.0.0.1:8000/v1/models",
                     "harness.py", "--mode canary", "--mode tps",
                     "quality2.py gate", "compare.py tps-verdict",
                     "compare.py tps-promote", "docker rm -f"):
        assert expected in text, f"dry run never mentions {expected!r}"
    # nothing was actually created
    assert not (root / "results" / "exp-alpha" / "results.json").exists()


# A target whose researcher edits no engine tree: engine.sources left out.
NO_TREES = TPS_TARGET.replace("tps-test", "tps-bare").replace("sources: [vllm, sglang], ", "")


def test_a_target_with_no_engine_tree_plans_no_copy_and_no_mount(tmp_path):
    """No engine is assumed. Without engine.sources the plan copies and mounts
    the experiment and nothing else, for any engine name."""
    root = make_bench_root(tmp_path)
    assert "sources" not in NO_TREES
    (root / "targets" / "tps-bare.yaml").write_text(NO_TREES.replace("name: mock", "name: my-engine"))
    out = run_sh(root, ["--dry-run", "exp-bare"], {"BENCH_TARGET": "tps-bare"})
    assert out.returncode == 0, out.stderr
    assert "docker run" in out.stdout and "/ro/exp" in out.stdout
    for absent in (":/ro/src/", "vllm-src", "sglang-src", ".head"):
        assert absent not in out.stdout, absent


@pytest.mark.slow
def test_a_target_with_no_engine_tree_runs_and_snapshots_none(tmp_path):
    root, env = local_bench(tmp_path, "bare")
    (root / "targets" / "tps-bare.yaml").write_text(NO_TREES)
    res = _tps_run((root, {**env, "BENCH_TARGET": "tps-bare"}), ["--instrument", "bare"])
    assert res["verdict"] == "INSTRUMENT", res["reasons"]
    snap = root / "results" / "bare" / "snapshot"
    assert snap.is_dir() and not list(snap.glob("*.head")) and not (snap / "src").exists()


def _sut_flags(plan: str) -> list[str]:
    """The scored launch's flags, the way the scorecard reads them."""
    line = next(l for l in plan.splitlines() if "docker run -d" in l)
    # up to the image (the container name starts with inference-judge- too)
    return line.split("docker run -d", 1)[1].split(" inference-judge-mock:")[0].split()


def test_dry_run_of_a_tps_target_is_sealed(tmp_path):
    """cpu-mock: no network, no published port, no GPU, read-only mounts
    only, and the TPS steps in ABBA order."""
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["--dry-run", "exp-alpha"])
    assert out.returncode == 0, out.stderr
    plan = out.stdout
    assert sum("docker run -d" in l for l in plan.splitlines()) == 4   # I C C I
    flags = _sut_flags(plan)
    assert "--network none" in " ".join(flags)
    assert "--gpus" not in flags and "-p" not in flags
    mounts = [flags[i + 1] for i, f in enumerate(flags) if f == "-v"]
    assert mounts and all(m.endswith(":ro") for m in mounts), mounts
    for expected in ("--mode pool", "--mode tps", "--phase warm", "--phase block1",
                     "quality2.py gate", "compare.py tps-launch",
                     "compare.py tps-screen", "compare.py tps-verdict",
                     "compare.py tps-promote", "network container:",
                     "exfil probe", "compare.py stamps"):
        assert expected in plan, f"dry run never mentions {expected!r}"


@pytest.mark.parametrize("args", [
    ["exp-alpha"], ["--confirm", "exp-alpha"], ["--calibrate", "cal-9"],
    ["--instrument", "exp-alpha"]])
def test_a_dry_run_prints_a_plan_and_no_verdict(tmp_path, args):
    """Nothing ran, so there is nothing to announce: the driver reads a
    VERDICT line as the outcome of a run."""
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["--dry-run", *args])
    assert out.returncode == 0, out.stderr
    assert "WOULD RUN" in out.stdout
    assert not re.search(r"^VERDICT\b", out.stdout + "\n" + out.stderr, re.M)
    assert "dry run" in out.stdout.splitlines()[-1]


@pytest.mark.parametrize("dry", [[], ["--dry-run"]])
def test_confirm_refuses_a_baseline_id(tmp_path, dry):
    """A baseline is the reference, never a promotion: there is nothing to
    confirm, with or without .allow-baseline, dry run or not."""
    root = make_bench_root(tmp_path, allow_baseline=True)
    (root / "results" / "baseline-mock").mkdir(parents=True)
    out = run_sh(root, [*dry, "--local", "--confirm", "baseline-mock"])
    assert out.returncode != 0
    assert "baseline-mock" in out.stderr and "--confirm" in out.stderr
    assert "VERDICT" not in out.stdout + out.stderr
    assert "WOULD RUN" not in out.stdout
    assert not (root / "results" / "baseline-mock" / "confirm").exists()


def test_a_target_sketch_is_refused(tmp_path):
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["--dry-run", "exp-alpha"], {"BENCH_TARGET": "sketch-tpu-8chip"})
    assert out.returncode == 70
    assert "runnable: false" in out.stderr


def test_dry_run_creates_no_files(tmp_path):
    root = make_bench_root(tmp_path)
    before = sorted(p.name for p in root.iterdir())
    run_sh(root, ["--dry-run", "exp-beta"])
    assert sorted(p.name for p in root.iterdir()) == before


# ---------------------------------------------------------- the judge's fence --
def test_dry_run_shows_the_guard_rails(tmp_path):
    """The lock, the watchdog, the safe copies, the snapshot, the artifact
    mount and the bounded build all have to be visible in the plan."""
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["--dry-run", "exp-guarded"])
    assert out.returncode == 0, out.stderr
    text = out.stdout
    for expected in (".lock", "watchdog", "--safe-links", "--no-specials",
                     "/home/researcher", "vllm.patch", "sglang.head",
                     "vllm.untracked", "diff HEAD",
                     # finding N1: snapshot git runs in a throwaway container,
                     # as the checkout's owner, read-only, with no network
                     "--network none", ":/r:ro", "alpine/git:latest",
                     "timeout 600s docker build", "deps.txt",
                     "alpine", "inference-judge-exp:*"):
        assert expected in text, f"dry run never mentions {expected!r}"
    # a scored launch has no writable mount at all (DESIGN 2.6)
    assert ":/out" not in text
    # only --instrument gets /out, and never the judge's own directory
    out = run_sh(root, ["--dry-run", "--instrument", "exp-guarded"])
    assert out.returncode == 0, out.stderr
    assert "results/exp-guarded/artifacts:/out:rw" in out.stdout
    assert f"{root}/results/exp-guarded:/out" not in out.stdout


def test_reserved_baseline_id_needs_the_operator_flag(tmp_path):
    """`baseline-*` is the operator's. Without /opt/bench/.allow-baseline the
    run is refused with a verdict, not with a crash."""
    root, env = local_bench(tmp_path, "baseline-sneaky")          # no flag file
    out = run_sh(root, ["--local", "baseline-sneaky"], env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "VERDICT REJECTED_RESERVED_ID" in out.stdout
    results = results_of(root, "baseline-sneaky")
    assert results["verdict"] == "REJECTED_RESERVED_ID"
    # nothing was measured and nothing of the operator's was written
    assert not (root / "results" / "baseline-sneaky" / "launches").exists()
    assert not (root / "leaderboard" / "tps-test.json").exists()
    assert not (root / "canary" / "tps-test.json").exists()
    assert not (root / "reference" / "tps-test").exists()


def test_reserved_id_is_accepted_with_the_flag(tmp_path):
    root = make_bench_root(tmp_path, allow_baseline=True)
    out = run_sh(root, ["--dry-run", "baseline-vllm"])
    assert out.returncode == 0, out.stderr
    assert "REJECTED_RESERVED_ID" not in out.stdout


# ------------------------------------------------------- the sudo entry point --
SUDO = {"SUDO_USER": "researcher"}


@pytest.mark.parametrize("args", [
    ["--dry-run", "exp-alpha"],
    ["--local", "exp-alpha"],
    ["--calibrate", "cal-1"],
    ["--help"],
    ["exp-alpha", "exp-beta"],
    ["-h"],
    [],
])
def test_under_sudo_only_a_bare_id_is_accepted(tmp_path, args):
    root = make_bench_root(tmp_path)
    out = run_sh(root, args, SUDO)
    assert out.returncode == 2, (out.returncode, out.stdout, out.stderr)
    assert "run.sh:" in out.stderr
    assert not (root / "results").exists()


def test_under_sudo_environment_overrides_are_ignored(tmp_path):
    """BENCH_ROOT and RESEARCH_ROOT in the caller's environment must not move
    the judge: the production run looks for /home/researcher/research only."""
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["exp-alpha"], {**SUDO, "RESEARCH_ROOT": str(tmp_path)})
    assert out.returncode == 70, out.stderr
    assert "/home/researcher/research/experiments/exp-alpha" in out.stderr


# ------------------------------------------------------------------- the lock --
def test_a_second_experiment_gets_busy(tmp_path):
    """Two runs can never share the GPU. macOS has no flock, so the mkdir lock
    is the one exercised here (run.sh picks flock when it exists)."""
    root, env = local_bench(tmp_path, "second-run")
    (root / ".lock.d").mkdir()                    # somebody else is running
    out = run_sh(root, ["--local", "second-run"],
                 {**env, "IJ_LOCK_MODE": "mkdir"}, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "VERDICT BUSY" in out.stdout
    assert results_of(root, "second-run")["verdict"] == "BUSY"
    assert (root / ".lock.d").exists()            # not stolen from its owner


def test_a_stale_lock_is_not_a_lock(tmp_path):
    root = make_bench_root(tmp_path)
    (root / ".lock.d").mkdir()
    (root / ".lock.d" / "pid").write_text("999999\n")   # long gone
    out = run_sh(root, ["--dry-run", "exp-alpha"], {"IJ_LOCK_MODE": "mkdir"})
    assert out.returncode == 0, out.stderr
    assert "BUSY" not in out.stdout


# ---------------------------------------------------------------- the watchdog --
@pytest.mark.slow
def test_a_tiny_budget_is_a_timeout(tmp_path):
    """The wall clock is enforced by bench, in whatever phase the run is."""
    root, env = local_bench(tmp_path, "baseline-mock", allow_baseline=True)
    out = run_sh(root, ["--local", "baseline-mock"],
                 {**env, "IJ_BUDGET_S": "3"}, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "VERDICT TIMEOUT" in out.stdout
    results = results_of(root, "baseline-mock")
    assert results["verdict"] == "TIMEOUT"
    assert results["duration_s"] < 120
    assert (root / "results" / "baseline-mock" / "summary.txt").exists()


# ------------------------------------------------------------------- deps.txt --
def test_deps_are_baked_into_an_image_before_the_sealed_launch(tmp_path):
    """deps.txt is installed by a bounded docker build (which has the
    network) into a derived image; the launch that runs launch.sh has none."""
    root = make_bench_root(tmp_path)
    out = run_sh(root, ["--dry-run", "exp-alpha"])
    plan = out.stdout
    assert "timeout 600s docker build -t inference-judge-exp:exp-alpha-deps" in plan
    assert plan.index("inference-judge-exp:exp-alpha-deps") < plan.index("docker run -d")
    assert "exec bash /exp/launch.sh" in plan
    assert "uv pip install" not in plan.split("docker run -d", 1)[1]


def _fake_docker(tmp_path: Path, rc: int) -> tuple[dict, Path]:
    """A `docker` first on PATH that notes how it was called and exits `rc`:
    no daemon is asked, and nothing is fetched or built."""
    bin_dir, calls = tmp_path / "bin", tmp_path / "docker.calls"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        f'#!/bin/sh\necho "$@" >> "{calls}"\nexit {rc}\n')
    (bin_dir / "docker").chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}, calls


@pytest.mark.slow
def test_deps_txt_is_recorded_when_the_experiment_brings_one(tmp_path):
    """results.json says whether deps.txt went in: true for an experiment
    that brought one and whose server came up, false for one without. One
    unscored launch each, with a docker that builds nothing."""
    root, env = local_bench(tmp_path, "with-deps", "no-deps")
    docker, _ = _fake_docker(tmp_path, 0)
    exps = Path(env["RESEARCH_ROOT"]) / "experiments"
    (exps / "with-deps" / "deps.txt").write_text("flashinfer-python==0.2.0\n")
    for exp_id, brought in (("with-deps", True), ("no-deps", False)):
        res = _tps_run((root, {**env, **docker}), ["--instrument", exp_id])
        assert res["verdict"] == "INSTRUMENT", res["reasons"]
        assert res["deps_installed"] is brought, exp_id


@pytest.mark.slow
def test_a_local_run_with_deps_txt_needs_no_docker(tmp_path):
    """--local runs launch.sh on this host: there is no image to bake
    deps.txt into, so docker is not asked and a host without it still gets
    its launch (README: a real run against the mock, with no Docker)."""
    root, env = local_bench(tmp_path, "with-deps")
    docker, calls = _fake_docker(tmp_path, 1)
    exp = Path(env["RESEARCH_ROOT"]) / "experiments" / "with-deps"
    (exp / "deps.txt").write_text("flashinfer-python==0.2.0\n")
    res = _tps_run((root, {**env, **docker}), ["--instrument", "with-deps"])
    assert not calls.exists(), calls.read_text()
    assert res["verdict"] == "INSTRUMENT", res["reasons"]
    assert res["deps_installed"] is True


# ----------------------------------------------------------------- isolation --
def test_a_module_in_the_callers_cwd_is_never_imported(tmp_path):
    """The researcher's cwd must not reach the judge's python.
    A json.py sitting where run.sh was called from would otherwise run as
    bench."""
    root, env = local_bench(tmp_path, "baseline-nope")
    evil = tmp_path / "evil"
    evil.mkdir()
    marker = tmp_path / "pwned"
    for name in ("json.py", "pathlib.py", "runpy.py", "os.py"):
        (evil / name).write_text(
            f"open({str(marker)!r}, 'w').write('{name}')\n"
            "raise SystemExit('hijacked')\n")
    out = run_sh(root, ["--local", "baseline-nope"], env, cwd=evil)
    # the run still produced its verdict file, written by the judge's python
    assert out.returncode == 0, out.stderr[-2000:]
    assert results_of(root, "baseline-nope")["verdict"] == "REJECTED_RESERVED_ID"
    assert not marker.exists(), "the caller's cwd was on sys.path"


# ------------------------------------------------------ the source snapshot --
def _git_repo(path: Path) -> None:
    """A checkout with a commit, a staged change, an unstaged change, an
    untracked file, and the two git settings that turn `git diff` into command
    execution (finding N1)."""
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(path / ".gitconfig-none"),
           "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=path, env=env, check=True,
                       capture_output=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (path / "tracked.py").write_text("first\n")
    (path / "also.py").write_text("first\n")
    git("add", "tracked.py", "also.py")
    git("commit", "-qm", "one")
    (path / "tracked.py").write_text("first\nstaged change\n")
    git("add", "tracked.py")
    (path / "also.py").write_text("first\nunstaged change\n")
    (path / "brand_new_kernel.cu").write_text("// untracked\n")
    # what the review says can execute as bench
    git("config", "diff.external", "/bin/sh -c 'touch %s; exit 0'"
        % (path / "PWNED"))
    git("config", "core.fsmonitor", "/bin/sh -c 'touch %s; exit 0'"
        % (path / "PWNED_FSMONITOR"))


def _snapshot_env(tmp_path: Path) -> dict:
    """Point run.sh at engine checkouts it is allowed to read, under tmp."""
    src = tmp_path / "src"
    _git_repo(src / "vllm-src")
    _git_repo(src / "sglang-src")
    return {"IJ_SRC_ROOT": str(src)}


def _have_git_image(image: str = "alpine/git:latest") -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.slow
def test_a_missing_git_image_is_not_a_failed_run(tmp_path):
    """Snapshotting is bookkeeping, never a gate. If the sandbox image is not
    on the box the run says so in the .head file and carries on."""
    root, env = local_bench(tmp_path, "no-git-image")
    env.update(_snapshot_env(Path(os.path.realpath(tmp_path))))
    env["IJ_GIT_IMAGE"] = "inference-judge-no-such-image:never-pulled"
    out = run_sh(root, ["--local", "no-git-image"], env)
    assert out.returncode == 0, out.stderr[-3000:]
    assert results_of(root, "no-git-image")["verdict"] == "INCONCLUSIVE"
    snap = root / "results" / "no-git-image" / "snapshot"
    for name in ("vllm", "sglang"):
        assert "alpine/git image missing" in (snap / f"{name}.head").read_text()
        assert (snap / f"{name}.patch").read_text() == ""
        assert (snap / f"{name}.untracked").read_text() == ""


@pytest.mark.slow
@pytest.mark.skipif(not _have_git_image(),
                    reason="alpine/git:latest is not pulled on this machine")
def test_the_snapshot_is_taken_by_a_sandboxed_git(tmp_path):
    """The git that reads the researcher's checkout runs in
    a throwaway container, and what it brings back is staged plus unstaged
    changes and the list of untracked files."""
    root, env = local_bench(tmp_path, "snapped")
    base = Path(os.path.realpath(tmp_path))
    env.update(_snapshot_env(base))
    out = run_sh(root, ["--local", "snapped"], env)
    assert out.returncode == 0, out.stderr[-3000:]
    snap = root / "results" / "snapped" / "snapshot"
    for name in ("vllm", "sglang"):
        head = (snap / f"{name}.head").read_text().strip()
        assert len(head) == 40, head                     # a commit sha
        patch = (snap / f"{name}.patch").read_text()
        assert "staged change" in patch                  # git diff HEAD
        assert "unstaged change" in patch
        untracked = (snap / f"{name}.untracked").read_text().split()
        assert untracked == ["brand_new_kernel.cu"]
    # and nothing in the checkout's git config ran on this host
    src = base / "src"
    assert not list(src.glob("*/PWNED"))
    assert not list(src.glob("*/PWNED_FSMONITOR"))


# ------------------------------------------- the watchdog's own cleanup path --
def _function_body(text: str, name: str) -> str:
    start = text.index(f"\n{name}() {{\n")
    end = text.index("\n}\n", start)
    return text[start:end]


def test_every_docker_call_the_watchdog_makes_is_bounded():
    """After a TIMEOUT the watchdog stops the container and wipes
    the workspace. A docker daemon that never answers must not turn the
    watchdog itself into the thing that hangs."""
    text = (BENCH / "run.sh").read_text()
    for name in ("stop_container", "collect_artifacts", "cleanup_workspace",
                 "final_cleanup", "prune_images"):
        for line in _function_body(text, name).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "printf" in stripped:
                continue
            if stripped.startswith("command -v docker"):
                continue                       # a probe, not an invocation
            if re.search(r"(^|\|\| |&& )docker ", stripped):
                assert stripped.startswith("with_timeout "), \
                    f"unbounded docker call in {name}(): {stripped}"
    body = _function_body(text, "stop_container")
    assert "docker stop -t 20" in body          # a grace period, then the axe
    assert "docker rm -f" in body


def test_with_timeout_escalates_to_sigkill_and_keeps_the_process_group():
    """Review 3 finding 5: `timeout` alone waits for ever on a child that
    ignores TERM, and GNU timeout puts that child in its own process group,
    where the watchdog's group-wide kill cannot reach it."""
    text = (BENCH / "run.sh").read_text()
    body = _function_body(text, "with_timeout")
    assert "-k \"$KILL_GRACE_S\"" in body        # TERM, then KILL
    assert "kill -KILL" in body                  # and in the bash fallback too
    assert "--foreground" in _function_body(text, "resolve_timeout")


def test_a_cancelled_timeout_leaves_nothing_holding_the_callers_pipe(tmp_path):
    """The bash fallback (no `timeout` binary): once the command is done the
    timer and its sleep are gone, so a reader of run.sh's output gets EOF."""
    text = (BENCH / "run.sh").read_text()
    script = tmp_path / "wt.sh"
    script.write_text(_function_body(text, "resolve_timeout") + "\n}\n"
                      + _function_body(text, "with_timeout") + "\n}\n"
                      "KILL_GRACE_S=1; TIMEOUT_BIN=none; TIMEOUT_FOREGROUND=\n"
                      "with_timeout 300 true; echo rc=$?\n"
                      "with_timeout 1 sleep 30; echo rc=$?\n")
    started = time.time()
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                         timeout=60)
    assert out.stdout.split() == ["rc=0", "rc=124"]
    assert time.time() - started < 20


def test_the_watchdogs_own_cleanup_is_bounded():
    """The watchdog calls cleanup outside the body it monitors. A TERM-proof
    docker command there hangs the watchdog itself and keeps the lock."""
    body = _function_body((BENCH / "run.sh").read_text(), "watchdog")
    for step in ("stop_container", "collect_artifacts", "cleanup_workspace"):
        assert re.search(rf"bounded \d+ {step}\b", body), \
            f"unbounded {step} in the watchdog's timeout path"


# ------------------------------------------------------------- ruler lock --
def test_a_ruler_edited_after_the_lock_is_refused(tmp_path):
    """DESIGN 2.6: --lock-ruler freezes the ruler; any later edit to a ruler
    file refuses the run (dry run included) before anything is measured."""
    root = make_bench_root(tmp_path)
    lock = tmp_path / "RULER_LOCK"
    env = {"BENCH_RULER_LOCK": str(lock)}
    r = run_sh(root, ["--lock-ruler"], extra_env=env)
    assert r.returncode == 0, r.stderr
    assert lock.exists() and "RULER_LOCK" in r.stdout
    ok = run_sh(root, ["--dry-run", "exp-locked"], extra_env=env)
    assert ok.returncode == 0, ok.stderr
    assert f"ruler lock {lock}: matched" in ok.stdout
    with (root / "compare.py").open("a") as fh:
        fh.write("\n# edited after the lock\n")
    bad = run_sh(root, ["--dry-run", "exp-locked"], extra_env=env)
    assert bad.returncode != 0
    assert "refusing to run" in bad.stdout + bad.stderr
    assert not (root / "results" / "exp-locked" / "results.json").exists()


# ------------------------------------------------------- TPS end to end ----
# The whole TPS path on the mock engine, launched with --local (launch.sh as a
# host process; the sealed Docker launch has its own test below).
TPS_CASES = {"baseline-mock": None, "same-1": None, "fast-1": "fast",
             "truncate-1": "truncate", "eos-1": "early_eos",
             "lie-1": "lie_usage", "junk-1": "junk", "mixed-1": None,
             "hung-1": None, "prof-1": None, "nocal-1": None,
             "orphan-1": None, "ctrlc-1": None, "after-ctrlc-1": None,
             "wedged-1": "wedged", "refused-1": "fast"}


@pytest.fixture(scope="module")
def tps_bench(tmp_path_factory):
    """A bench root with the operator's baseline and calibration done."""
    from conftest import free_port
    base = Path(os.path.realpath(tmp_path_factory.mktemp("tps")))
    root = make_bench_root(base, allow_baseline=True)
    (root / "targets" / "tps-test.yaml").write_text(TPS_TARGET)
    exps = base / "src" / "research" / "experiments"
    launch = BENCH.parent / "research" / "experiments" / "baseline-mock" / "launch.sh"
    for exp_id, behaviour in TPS_CASES.items():
        (exps / exp_id).mkdir(parents=True)
        shutil.copy(launch, exps / exp_id / "launch.sh")
        if behaviour:
            (exps / exp_id / "behaviour").write_text(behaviour + "\n")
    env = {"BENCH_TARGET": "tps-test", "RESEARCH_ROOT": str(base / "src" / "research"),
           "IJ_SRC_ROOT": str(base / "src"), "PORT": str(free_port()),
           "CALIBRATE_LAUNCHES": "3"}
    for args in (["baseline-mock"], ["--calibrate", "cal-1"]):
        r = run_sh(root, ["--local", *args], extra_env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return root, env


def _tps_run(bench, args, extra=None):
    root, env = bench
    r = run_sh(root, ["--local", *args], extra_env={**env, **(extra or {})},
               timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    exp_id = args[-1]
    sub = "confirm/" if "--confirm" in args else ""
    return json.loads((root / "results" / f"{exp_id}/{sub}results.json".replace(
        "//", "/")).read_text())


def _best(root: Path) -> str:
    board = json.loads((root / "leaderboard" / "tps-test.json").read_text())
    return board["best"]["id"]


@pytest.mark.slow
def test_tps_baseline_and_calibration(tps_bench):
    root, _ = tps_bench
    assert results_of(root, "baseline-mock")["verdict"] == "BASELINE"
    cal = results_of(root, "cal-1")
    assert cal["verdict"] == "CALIBRATED" and cal["kind"] == "calibration"
    assert (root / "calibration" / "tps-test.json").exists()
    assert (root / "reference" / "tps-test" / "manifest.json").exists()
    assert _best(root) == "baseline-mock"

    # an ABBA target: no calibration launch answers what the one before it did
    def sent(label):
        return {json.loads(l)["prompt_id"]
                for f in (root / "results" / "cal-1" / "launches" / label).glob(
                    "*/tps_records.jsonl") for l in f.read_text().splitlines()}
    assert sent("K1") and sent("K2") and sent("K3")
    assert not sent("K1") & sent("K2") and not sent("K2") & sent("K3")


@pytest.mark.slow
def test_tps_a_baseline_writes_r_from_cold_launches_when_the_target_says(
        tmp_path):
    """reference.launches: 3, what a real engine gets without asking: the
    baseline's measured launch and two more that only answer the quality
    prompts make R, with the continuation length the target names."""
    from conftest import free_port
    base = Path(os.path.realpath(tmp_path))
    root = make_bench_root(base, allow_baseline=True)
    line = 'reference: {dir: "reference/tps-test"}'
    assert line in TPS_TARGET
    (root / "targets" / "tps-test.yaml").write_text(TPS_TARGET.replace(
        line, 'reference: {dir: "reference/tps-test", launches: 3, max_tokens: 48}'))
    exp = base / "src" / "research" / "experiments" / "baseline-mock"
    exp.mkdir(parents=True)
    shutil.copy(BENCH.parent / "research" / "experiments" / "baseline-mock" / "launch.sh",
                exp / "launch.sh")
    env = {"BENCH_TARGET": "tps-test", "RESEARCH_ROOT": str(base / "src" / "research"),
           "IJ_SRC_ROOT": str(base / "src"), "PORT": str(free_port())}
    res = _tps_run((root, env), ["baseline-mock"])
    assert res["verdict"] == "BASELINE", res["reasons"]
    manifest = json.loads((root / "reference" / "tps-test" / "manifest.json").read_text())
    assert manifest["source"] == "launch" and manifest["n_launches"] == 3
    assert manifest["max_tokens"] == 48
    out = root / "results" / "baseline-mock"
    for label in ("B", "R1", "R2"):
        assert (out / "launches" / label / "server.log").exists(), label
    assert sorted(p.name for p in (out / "launches").iterdir()) == ["B", "R1", "R2"]
    assert not (out / "staged" / "reference-launches").exists()
    assert res["stamps"]["reference_hash"]


@pytest.mark.slow
def test_tps_a_byte_identical_candidate_does_not_promote(tps_bench):
    res = _tps_run(tps_bench, ["same-1"])
    assert res["schema"] == 2 and res["kind"] == "candidate"
    assert res["verdict"] in ("REJECTED", "INCONCLUSIVE"), res["reasons"]
    assert res["reasons"]
    assert _best(tps_bench[0]) == "baseline-mock"


@pytest.mark.slow
@pytest.mark.parametrize("exp_id", ["truncate-1", "eos-1", "lie-1", "junk-1"])
def test_tps_a_cheating_candidate_is_invalid_or_rejected(tps_bench, exp_id):
    res = _tps_run(tps_bench, [exp_id])
    assert res["verdict"] in ("INVALID", "REJECTED", "REJECTED_SCREEN",
                              "CANARY_FAILED"), res
    assert res["reasons"], res
    assert _best(tps_bench[0]) == "baseline-mock"


@pytest.mark.slow
def test_tps_mixed_stamps_do_not_promote(tps_bench):
    """An incumbent measured under another ruler cannot be compared with."""
    root, _ = tps_bench
    board_file = root / "leaderboard" / "tps-test.json"
    saved = board_file.read_text()
    board = json.loads(saved)
    board["best"].setdefault("stamps", {})["quality_version"] = "0.9"
    board_file.write_text(json.dumps(board))
    try:
        res = _tps_run(tps_bench, ["mixed-1"])
    finally:
        board_file.write_text(saved)
    assert res["verdict"] == "INVALID", res
    assert any("ruler_mismatch" in r and "[incumbent]" in r
               for r in res["reasons"]), res["reasons"]
    # known from the stamps alone, so it is said before anything is launched
    assert not res.get("launches"), res.get("launches")
    assert not (root / "results" / "mixed-1" / "launches").exists()


@pytest.mark.slow
def test_tps_a_run_without_a_calibration_ends_before_any_launch(tps_bench):
    """No calibration means no verdict. That is known before the first launch,
    so the run says so there, not after five launches."""
    root, _ = tps_bench
    cal = root / "calibration" / "tps-test.json"
    saved = cal.read_text()
    cal.unlink()
    try:
        res = _tps_run(tps_bench, ["nocal-1"])
    finally:
        cal.write_text(saved)
    assert res["verdict"] == "INCONCLUSIVE", res
    assert len(res["reasons"]) == 1 and "no calibration" in res["reasons"][0]
    assert "--calibrate" in res["reasons"][0]
    assert not res.get("launches"), res.get("launches")
    assert not (root / "results" / "nocal-1" / "launches").exists()
    pre = json.loads((root / "results" / "nocal-1" / "preflight.json").read_text())
    assert pre["ok"] is False and pre["verdict"] == "INCONCLUSIVE"


def test_tps_a_run_without_an_incumbent_ends_before_any_launch(tmp_path):
    """Before the operator's baseline there is nothing to compare with."""
    from conftest import free_port
    base = Path(os.path.realpath(tmp_path))
    root = make_bench_root(base)
    (root / "targets" / "tps-test.yaml").write_text(TPS_TARGET)
    exp = base / "src" / "research" / "experiments" / "early-1"
    exp.mkdir(parents=True)
    shutil.copy(BENCH.parent / "research" / "experiments" / "baseline-mock" / "launch.sh",
                exp / "launch.sh")
    env = {"BENCH_TARGET": "tps-test", "RESEARCH_ROOT": str(base / "src" / "research"),
           "IJ_SRC_ROOT": str(base / "src"), "PORT": str(free_port())}
    res = _tps_run((root, env), ["early-1"])
    assert res["verdict"] == "INCONCLUSIVE", res
    assert [r.split(":")[0].split(" on ")[0] for r in res["reasons"]] == [
        "no incumbent", "no calibration"]
    assert "run the operator baseline first" in res["reasons"][0]
    assert not (root / "results" / "early-1" / "launches").exists()


@pytest.mark.slow
def test_tps_a_sweep_past_its_bound_is_hung(tps_bench):
    res = _tps_run(tps_bench, ["hung-1"], extra={"SWEEP_TIMEOUT_S": "1"})
    assert res["verdict"] == "HUNG", res
    assert _best(tps_bench[0]) == "baseline-mock"


@pytest.mark.slow
def test_tps_instrument_mode_mounts_out_and_is_never_scored(tps_bench):
    res = _tps_run(tps_bench, ["--instrument", "prof-1"])
    assert res["verdict"] == "INSTRUMENT" and res["kind"] == "instrument"
    assert _best(tps_bench[0]) == "baseline-mock"


@pytest.mark.slow
def test_tps_a_server_that_wedges_after_ready_fails_the_canary_quickly(tps_bench):
    """Health answers and the warm sweep is served, the canary's requests
    never are. Two request timeouts in a row end the canary and the run: the
    other eight prompts are not sent, and the reason says what happened."""
    root, _ = tps_bench
    res = _tps_run(tps_bench, ["wedged-1"], extra={"CANARY_TIMEOUT_S": "1"})
    assert res["verdict"] == "CANARY_FAILED", res
    assert any("canary stopped early" in why
               and "2 consecutive request timeouts" in why
               for why in res["reasons"]), res["reasons"]
    assert any("0 of 10 fixed prompts unchanged" in why
               for why in res["reasons"]), res["reasons"]
    assert res["canary"] == {"matched": 0, "total": 10}
    out = root / "results" / "wedged-1"
    canary = json.loads((out / "launches" / "C1" / "canary.json").read_text())
    errors = [r["error"] or "" for r in canary["results"]]
    assert sum("Timeout" in e for e in errors) == 2, errors
    assert sum("not sent" in e for e in errors) == 8, errors
    # the run ended at the canary: no block of C1, and no launch after it
    assert sorted(p.name for p in (out / "launches").iterdir()) == ["C1", "I1"]
    assert sorted(p.name for p in (out / "launches" / "C1").iterdir()) == [
        "canary.json", "server.log", "warm"]
    assert _best(root) == "baseline-mock"


@pytest.fixture(scope="module")
def refused_promotion(tps_bench):
    """A faster candidate, run while the leaderboard cannot be written: the
    comparison says PROMOTED and compare.py tps-promote then exits non-zero."""
    root, _ = tps_bench
    board_dir = root / "leaderboard"
    before = (board_dir / "tps-test.json").read_bytes()
    mode = board_dir.stat().st_mode
    board_dir.chmod(0o555)                # read by every step, written by one
    try:
        res = _tps_run(tps_bench, ["refused-1"])
    finally:
        board_dir.chmod(mode)
    return res, before


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes where it may not")
def test_tps_a_refused_promotion_is_rejected_and_the_best_stays(
        tps_bench, refused_promotion):
    """tps_promote_step: the verdict was PROMOTED, the promotion itself was
    refused. The run is written up as REJECTED with the reason, everywhere a
    reader looks, and the leaderboard is what it was."""
    root, _ = tps_bench
    res, before = refused_promotion
    out = root / "results" / "refused-1"
    assert json.loads((out / "verdict.json").read_text())["verdict"] == "PROMOTED"
    assert res["verdict"] == "REJECTED", res
    assert any("compare.py refused the promotion" in why
               for why in res["reasons"]), res["reasons"]
    transcript = (out / "run.log").read_text()
    assert "recording REJECTED instead" in transcript
    assert "VERDICT REJECTED" in transcript and "VERDICT PROMOTED" not in transcript
    assert "verdict:    REJECTED" in (out / "summary.txt").read_text()
    assert (root / "leaderboard" / "tps-test.json").read_bytes() == before
    assert [p.name for p in (root / "leaderboard").iterdir()] == ["tps-test.json"]
    assert _best(root) == "baseline-mock"


@pytest.mark.slow
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes where it may not")
def test_tps_a_refused_promotion_gives_the_reason_it_was_refused_for(
        refused_promotion):
    """quality2 passed on every candidate launch here, and results.json says
    so; the reason must not say it did not."""
    res, _ = refused_promotion
    assert res["quality"]["passed"] is True
    assert not any("quality2 did not pass" in why for why in res["reasons"]), \
        res["reasons"]


def _start_detached(bench, args: list[str]) -> subprocess.Popen:
    root, env = bench
    return subprocess.Popen(["bash", str(root / "run.sh"), "--local", *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=bench_env(root, env),
                            start_new_session=True)


def _wait_for(path: Path, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while not path.exists():
        assert time.time() < deadline, f"{path} never appeared"
        time.sleep(0.2)


def _end_detached(proc: subprocess.Popen, wait_s: float = 30) -> None:
    """A run the test did not see end must not outlive the test: its body
    and its server are in a process group of their own, hold the lock and
    the port every tps_bench test shares, and would fail the tests after
    this one. SIGINT first, the run's own way out; then SIGKILL for both
    groups."""
    if proc.poll() is not None:
        return
    table = subprocess.run(["ps", "-axo", "pgid=,ppid="], capture_output=True,
                           text=True).stdout.split()
    groups = {proc.pid} | {int(pgid) for pgid, ppid in zip(table[::2], table[1::2])
                           if int(ppid) == proc.pid}
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        pass
    for pgid in groups:                 # nothing is left of it after SIGINT
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# These two share tps_bench with the tests above. What they need from it:
# baseline-mock is still the best (a promoted fast-1 would end both runs
# at the screen), and nobody holds the lock or the port. And, as same-1
# does, a machine as fast as when the fixture measured the baseline: the
# mock's throughput moves in steps of a whole answer per window (682.7 or
# 512.0 tok/s on TPS_TARGET), so a run that comes out one step under the
# stored sentinel is REJECTED_SCREEN after two launches.
@pytest.mark.slow
def test_the_callers_sigterm_does_not_end_the_run(tps_bench):
    """What Claude Code sends to a background task when the researcher's
    session ends: one SIGTERM to the task's process group. The run must
    finish on its own, with a verdict and a transcript."""
    root, _ = tps_bench
    assert _best(root) == "baseline-mock"
    proc = _start_detached(tps_bench, ["orphan-1"])
    try:
        _wait_for(root / "results" / "orphan-1" / "launches")
        os.killpg(proc.pid, signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=600)
    finally:
        _end_detached(proc)
    assert proc.returncode == 0, stdout[-3000:]
    results = results_of(root, "orphan-1")
    # all four launches were measured: the same answer as same-1 gets
    assert results["verdict"] in ("REJECTED", "INCONCLUSIVE"), results["reasons"]
    assert sorted(results["launches"]) == ["C1", "C2", "I1", "I2"]
    assert "SIGTERM from the caller ignored" in stdout
    assert f"VERDICT {results['verdict']}" in stdout
    transcript = (root / "results" / "orphan-1" / "run.log").read_text()
    assert "SIGTERM from the caller ignored" in transcript
    assert f"VERDICT {results['verdict']}" in transcript


@pytest.mark.slow
def test_sigint_aborts_the_run_and_frees_the_lock(tps_bench):
    root, _ = tps_bench
    assert _best(root) == "baseline-mock"
    proc = _start_detached(tps_bench, ["ctrlc-1"])
    try:
        _wait_for(root / "results" / "ctrlc-1" / "launches")
        os.killpg(proc.pid, signal.SIGINT)
        stdout, _ = proc.communicate(timeout=60)
    finally:
        _end_detached(proc)
    assert proc.returncode == 130, stdout[-3000:]
    assert "SIGINT: aborting" in stdout
    assert not (root / "results" / "ctrlc-1" / "results.json").exists()
    # the lock is free again: the next run is not BUSY
    after = _tps_run(tps_bench, ["after-ctrlc-1"])
    assert after["verdict"] in ("REJECTED", "INCONCLUSIVE"), after["reasons"]


# Promotes, so it runs after every test that expects baseline-mock as best.
@pytest.mark.slow
def test_tps_a_faster_candidate_promotes_and_confirms(tps_bench):
    root, _ = tps_bench
    res = _tps_run(tps_bench, ["fast-1"])
    assert res["verdict"] == "PROMOTED", res["reasons"]
    assert res["stats"] and res["metric"]["kind"] == "tps"
    assert _best(root) == "fast-1"
    conf = _tps_run(tps_bench, ["--confirm", "fast-1"])
    assert conf["kind"] == "confirm"
    assert conf["verdict"] == "PROMOTED", conf["reasons"]


# ------------------------------------------------------ the seal, for real --
SEAL_TARGET = """\
id: seal-test
runnable: true
accelerator: {kind: none, chips: 1, hosts: 1}
model: {served_name: mock, revision: "0", local_dir: "", tokenizer: "ws"}
engine: {sources: [vllm, sglang], name: mock, image: "python:3.12-slim", docker_flags: [], port: 8000}
metric:
  kind: tps
  isl: 64
  osl: 32
  concurrency: [4]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 0.3
  window_s: 1.5
  design: abba
  blocks: 2
budgets: {build_s: 120, compile_s: 0, health_s: 15, run_s: 300}
reference: {dir: "reference/seal-test"}
"""
# launch.sh is the candidate's own code: here it tries to get out and says how
# far it got. It never serves, so the run ends BUILD_FAILED with the report in
# the container's log.
SEAL_PROBE = r"""#!/usr/bin/env bash
python3 - <<'PY'
import json, os, socket
res = {"net": {}, "write": {}}
for host, port in (("1.1.1.1", 53), ("pypi.org", 443)):
    try:
        socket.create_connection((host, port), timeout=3).close()
        res["net"][host] = "open"
    except Exception as exc:
        res["net"][host] = "blocked"
for d in ("/ro/exp", "/ro/src/vllm", "/ro/src/sglang",
          "/opt/inference-judge/mock_server.py", "/out"):
    if not os.path.exists(d):
        res["write"][d] = "absent"
        continue
    try:
        if os.path.isfile(d):
            open(d, "a").close()
        else:
            open(os.path.join(d, ".probe"), "w").write("x")
        res["write"][d] = "writable"
    except Exception:
        res["write"][d] = "refused"
print("SEALPROBE " + json.dumps(res, sort_keys=True), flush=True)
PY
exit 3
"""


def _have_image(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image],
                              capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _have_image("python:3.12-slim"),
                    reason="needs docker and python:3.12-slim")
def test_a_sealed_launch_has_no_network_and_nothing_to_write(tmp_path_factory):
    """A real Docker launch of the cpu-mock seal: from inside the candidate's
    container the network is gone and every mount refuses a write, and the
    ruler's own probe records the same in results.json."""
    base = Path(os.path.realpath(tmp_path_factory.mktemp("seal")))
    root = make_bench_root(base, allow_baseline=True)
    (root / "targets" / "seal-test.yaml").write_text(SEAL_TARGET)
    exp = base / "src" / "research" / "experiments" / "baseline-probe"
    exp.mkdir(parents=True)
    (exp / "launch.sh").write_text(SEAL_PROBE)
    out = run_sh(root, ["baseline-probe"], extra_env={
        "BENCH_TARGET": "seal-test", "IJ_RULER_IMAGE": "python:3.12-slim",
        "RESEARCH_ROOT": str(base / "src" / "research"),
        "IJ_SRC_ROOT": str(base / "src")})
    assert out.returncode == 0, out.stderr[-3000:]
    results = results_of(root, "baseline-probe")
    assert results["verdict"] == "BUILD_FAILED"       # it never served
    iso = results["isolation"]
    assert iso["network"] == "none" and iso["writable_mounts"] == []
    assert iso["exfil_probe"] == "blocked" and iso["seal"] == "full"
    logs = "".join(p.read_text(errors="replace") for p in
                   (root / "results" / "baseline-probe").rglob("*.log"))
    # an exited container ends the health wait at the next poll, not at its end
    assert "exited before becoming healthy" in logs + out.stderr
    line = next(l for l in logs.splitlines() if l.startswith("SEALPROBE "))
    seen = json.loads(line.split(" ", 1)[1])
    assert set(seen["net"].values()) == {"blocked"}, seen
    assert seen["write"].pop("/out") == "absent", seen
    assert set(seen["write"].values()) == {"refused"}, seen


# ------------------------------------------- the TPS path, sealed in Docker --
# The same ruler steps as the --local block above, through the real launcher:
# every launch is a sealed container and every measurement a sidecar in its
# network namespace. One test, because the steps build on each other.
TPS_DOCKER_TARGET = TPS_TARGET.replace("tps-test", "tps-docker")


@pytest.mark.slow
@pytest.mark.skipif(not (_have_image("inference-judge-mock:latest")
                         and _have_image("inference-judge-ruler:latest")),
                    reason="needs docker, inference-judge-mock:latest and inference-judge-ruler:latest")
def test_tps_through_the_sealed_docker_launcher(tmp_path_factory):
    """Baseline, calibration, a faster candidate that is PROMOTED from inside
    the seal and confirmed, and a cheat that is not."""
    assert 'image: "inference-judge-mock:latest"' in TPS_DOCKER_TARGET
    base = Path(os.path.realpath(tmp_path_factory.mktemp("tpsdocker")))
    root = make_bench_root(base, allow_baseline=True)
    (root / "targets" / "tps-docker.yaml").write_text(TPS_DOCKER_TARGET)
    exps = base / "src" / "research" / "experiments"
    launch = BENCH.parent / "research" / "experiments" / "baseline-mock" / "launch.sh"
    for exp_id, behaviour in (("baseline-mock", None), ("fast-1", "fast"),
                              ("junk-1", "junk")):
        (exps / exp_id).mkdir(parents=True)
        shutil.copy(launch, exps / exp_id / "launch.sh")
        if behaviour:
            (exps / exp_id / "behaviour").write_text(behaviour + "\n")
    env = {"BENCH_TARGET": "tps-docker", "RESEARCH_ROOT": str(base / "src" / "research"),
           "IJ_SRC_ROOT": str(base / "src"), "CALIBRATE_LAUNCHES": "3"}
    seen, started = {}, time.time()

    def step(args):
        t0 = time.time()
        r = run_sh(root, args, extra_env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        sub = "/confirm" if "--confirm" in args else ""
        res = json.loads((root / "results" / (args[-1] + sub) / "results.json").read_text())
        seen[" ".join(args)] = (res["verdict"], round(time.time() - t0))
        print(f"docker tps: {' '.join(args)} -> {res['verdict']} "
              f"in {time.time() - t0:.0f}s {res.get('reasons')}")
        return res

    def sealed(res):
        iso = res["isolation"]
        return (iso["network"] == "none" and iso["writable_mounts"] == []
                and iso["exfil_probe"] == "blocked" and iso["seal"] == "full")

    try:
        assert step(["baseline-mock"])["verdict"] == "BASELINE"
        assert step(["--calibrate", "cal-1"])["verdict"] == "CALIBRATED"
        fast = step(["fast-1"])
        assert fast["verdict"] == "PROMOTED", fast["reasons"]
        assert sealed(fast), fast["isolation"]
        board = json.loads((root / "leaderboard" / "tps-docker.json").read_text())
        assert board["best"]["id"] == "fast-1"
        conf = step(["--confirm", "fast-1"])
        assert conf["kind"] == "confirm"
        assert conf["verdict"] == "PROMOTED", conf["reasons"]
        assert sealed(conf), conf["isolation"]
        junk = step(["junk-1"])
        assert junk["verdict"] in ("INVALID", "REJECTED", "REJECTED_SCREEN",
                                   "CANARY_FAILED"), junk
        assert junk["reasons"] and sealed(junk)
        board = json.loads((root / "leaderboard" / "tps-docker.json").read_text())
        assert board["best"]["id"] == "fast-1"
    finally:
        print(f"docker tps: {seen} total {time.time() - started:.0f}s")
        for exp_id in ("baseline-mock", "fast-1", "junk-1"):
            subprocess.run(["docker", "rmi", f"inference-judge-inc:tps-docker-{exp_id}"],
                           capture_output=True, timeout=60)


CACHE_LAUNCH = """\
#!/usr/bin/env bash
# the mock, saying what is in its compile cache and then "compiling" a shape
set -euo pipefail
who="$(cat "$(dirname -- "$0")/who")"
echo "CACHE_SEEN $(ls /cache | sort | tr '\\n' ' ')"
touch "/cache/$who-$(ls /cache | grep -c "^$who-" || true)"
exec python3 -I /opt/inference-judge/mock_server.py --behaviour honest --port "${PORT:-8000}" \\
  --host 0.0.0.0 --model "${IJ_SERVED_NAME:-mock}"
"""


@pytest.mark.slow
@pytest.mark.skipif(not (_have_image("inference-judge-mock:latest")
                         and _have_image("inference-judge-ruler:latest")),
                    reason="needs docker, inference-judge-mock:latest and inference-judge-ruler:latest")
def test_the_candidate_keeps_its_compile_cache_from_s_to_c2(tmp_path_factory):
    """What the candidate compiled in S is there for C1, and both for C2: it
    compiles its shapes once in a run. None of it reaches the seed or an
    incumbent's launch; what the incumbent compiled does reach the seed. And no
    prompt goes to the candidate twice."""
    base = Path(os.path.realpath(tmp_path_factory.mktemp("kept")))
    root = make_bench_root(base, allow_baseline=True)
    seed = base / "seed"
    seed.mkdir()
    (seed / "seed-shape").write_text("x\n")
    port = TWO_TREES
    assert port in TPS_TARGET
    (root / "targets" / "tps-kept.yaml").write_text(
        TPS_TARGET.replace("tps-test", "tps-kept").replace(port, port[:-1] + (
            f", compile_cache: {{seed: {seed}, mount: /cache}}}}")).replace(
            "  design: abba\n", "  design: abba\n  screen_window_s: 1\n"))
    exps = base / "src" / "research" / "experiments"
    for exp_id, who in (("baseline-mock", "base"), ("cand-1", "cand")):
        (exps / exp_id).mkdir(parents=True)
        (exps / exp_id / "launch.sh").write_text(CACHE_LAUNCH)
        (exps / exp_id / "who").write_text(who + "\n")
    env = {"BENCH_TARGET": "tps-kept", "RESEARCH_ROOT": str(base / "src" / "research"),
           "IJ_SRC_ROOT": str(base / "src")}

    def step(exp_id):
        r = run_sh(root, [exp_id], extra_env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return results_of(root, exp_id)

    def saw(exp_id, label):
        log = (root / "results" / exp_id / "launches" / label / "server.log").read_text()
        return next(l for l in log.splitlines() if l.startswith("CACHE_SEEN")).split()[1:]

    try:
        assert step("baseline-mock")["verdict"] == "BASELINE"
        assert saw("baseline-mock", "B") == ["seed-shape"]
        # a calibration, so the run is launched at all; a real one would add
        # its own incumbent launches to the seed
        (root / "calibration").mkdir(exist_ok=True)
        (root / "calibration" / "tps-kept.json").write_text(json.dumps(
            {"sigma_L": 0.05, "mde": 0.16, "n_launches": 10}))
        res = step("cand-1")
        assert res["verdict"] in ("REJECTED", "INCONCLUSIVE"), res["reasons"]
        assert sorted(res["launches"]) == ["C1", "C2", "I1", "I2", "S"]
        assert saw("cand-1", "S") == ["base-0", "seed-shape"]
        assert saw("cand-1", "C1") == ["base-0", "cand-0", "seed-shape"]
        assert saw("cand-1", "C2") == ["base-0", "cand-0", "cand-1", "seed-shape"]
        assert saw("cand-1", "I1") == ["base-0", "seed-shape"]
        assert saw("cand-1", "I2") == ["base-0", "base-1", "seed-shape"]
        assert sorted(p.name for p in seed.iterdir()) == [
            "base-0", "base-1", "base-2", "seed-shape"]
        assert res["cleanup_ok"] is True

        def prompts(label):
            return {json.loads(l)["prompt_id"]
                    for f in (root / "results" / "cand-1" / "launches" / label).glob(
                        "*/tps_records.jsonl") for l in f.read_text().splitlines()}
        assert prompts("S") and prompts("C1") and prompts("C2")
        assert not prompts("S") & prompts("C1")
        assert not (prompts("S") | prompts("C1")) & prompts("C2")
    finally:
        for exp_id in ("baseline-mock",):
            subprocess.run(["docker", "rmi", f"inference-judge-inc:tps-kept-{exp_id}"],
                           capture_output=True, timeout=60)


# ------------------------------- the target's trees, mix, screen and cache --
MIX_TARGET = """\
id: tps-mix
runnable: true
accelerator: {kind: none, chips: 1, hosts: 1}
model: {served_name: mock, revision: "0", local_dir: "", tokenizer: "ws"}
engine:
  name: mock
  image: "inference-judge-mock:latest"
  docker_flags: []
  port: 8000
  sources: [tpu_inference, vllm]
metric:
  kind: tps
  isl: [16, 32, 64]
  osl: 32
  concurrency: [1]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 0.3
  window_s: 3
  screen_window_s: 2
  design: abba
  blocks: 2
budgets: {build_s: 600, compile_s: 60, health_s: 60, run_s: 1800}
reference: {dir: "reference/tps-mix"}
quality: {allow_missing_logprobs: false}
"""


def _plan(tmp_path, target: str, args=("exp-alpha",)) -> str:
    root = make_bench_root(tmp_path)
    (root / "targets" / "tps-mix.yaml").write_text(target)
    out = run_sh(root, ["--dry-run", *args], {"BENCH_TARGET": "tps-mix"})
    assert out.returncode == 0, out.stderr
    return out.stdout


def _launch_lines(plan: str) -> list[str]:
    return [l for l in plan.splitlines() if "docker run -d" in l]


def _mounts(line: str) -> list[str]:
    flags = line.split("docker run -d", 1)[1].split(" inference-judge-mock:")[0].split()
    # the plan shell-quotes every flag (%q): <private-tmp> comes out escaped
    return [flags[i + 1].replace("\\", "") for i, f in enumerate(flags) if f == "-v"]


def test_dry_run_mounts_the_engine_trees_the_target_names(tmp_path):
    plan = _plan(tmp_path, MIX_TARGET)
    for expected in ("/home/researcher/tpu_inference-src/", "/home/researcher/vllm-src/",
                     "tpu_inference.patch", "tpu_inference.untracked", "vllm.head",
                     "/ro/src/tpu_inference:ro", "/ro/src/vllm:ro",
                     "write into /ro/exp /ro/src/tpu_inference /ro/src/vllm"):
        assert expected in plan, f"dry run never mentions {expected!r}"
    assert "sglang" not in plan


def test_dry_run_gives_the_incumbent_the_trees_it_was_promoted_with(tmp_path):
    """An engine edit is only measured if the incumbent does not get it too."""
    by_label = {l.split("--name ", 1)[1].split()[0].rsplit("-", 1)[1]: _mounts(l)
                for l in _launch_lines(_plan(tmp_path, MIX_TARGET))}
    assert "<private-tmp>/src/vllm:/ro/src/vllm:ro" in by_label["C1"]
    assert "<private-tmp>/best-src/vllm:/ro/src/vllm:ro" in by_label["I1"]
    assert "<private-tmp>/best-src/tpu_inference:/ro/src/tpu_inference:ro" in by_label["I2"]


def test_dry_run_screens_the_candidate_alone_before_any_incumbent(tmp_path):
    plan = _plan(tmp_path, MIX_TARGET)
    names = [l.split("--name ", 1)[1].split()[0] for l in _launch_lines(plan)]
    assert [n.rsplit("-", 1)[1] for n in names] == ["S", "I1", "C1", "C2", "I2"]
    lines = plan.splitlines()
    screen = next(i for i, l in enumerate(lines) if "compare.py tps-screen" in l)
    assert "launches/S/launch.json" in lines[screen]
    assert screen < next(i for i, l in enumerate(lines) if names[1] in l)
    # one short window, after a warm pass over every size; no canary, no gate
    s_part = "\n".join(lines[:screen])
    assert "--phase block1" in s_part and "--window-s 2 " in s_part
    assert "--phase block2" not in s_part
    assert "--warm-cycles 1" in s_part and "--isl 16,32,64" in s_part
    assert "--mode canary" not in s_part and "quality2.py gate" not in s_part
    assert plan.count("quality2.py gate") == 2 and plan.count("--mode canary") == 1


def test_dry_run_checks_the_stamps_before_the_first_launch(tmp_path):
    """A ruler mismatch with the calibration or the incumbent is known as soon
    as the stamps are: it ends the run there, not after five launches."""
    lines = _plan(tmp_path, MIX_TARGET).splitlines()
    at = {key: next(i for i, l in enumerate(lines) if key in l)
          for key in ("compare.py stamps", "compare.py tps-preflight", "docker run -d")}
    assert at["compare.py stamps"] < at["compare.py tps-preflight"] < at["docker run -d"]
    pre = lines[at["compare.py tps-preflight"]]
    assert "--calibration" in pre and "--leaderboard" in pre and "stamps.json" in pre


def test_a_calibration_gets_a_budget_for_all_its_launches(tmp_path):
    """budgets.run_s is one candidate run: the design's full launches (abba
    four, ici three). A calibration is CALIBRATE_LAUNCHES full launches under
    the same watchdog and gets the same share for each, never less than one
    run's; an override still wins."""
    root = make_bench_root(tmp_path)
    (root / "targets" / "tps-mix.yaml").write_text(MIX_TARGET)
    (root / "targets" / "tps-ici.yaml").write_text(
        MIX_TARGET.replace("id: tps-mix", "id: tps-ici").replace("design: abba",
                                                                 "design: ici"))

    def budget(args, target="tps-mix", **env):
        out = run_sh(root, ["--dry-run", *args],
                     {"BENCH_TARGET": target, "HARD_TIMEOUT_S": "", **env})
        assert out.returncode == 0, out.stderr
        return int(re.search(r"kill the whole run after (\d+)s", out.stdout).group(1))

    assert budget(["exp-alpha"]) == 1800                           # run_s
    assert budget(["--confirm", "exp-alpha"]) == 1800
    assert budget(["--calibrate", "cal-9"]) == 4500                # 10 of 4
    assert budget(["--calibrate", "cal-9"], "tps-ici") == 6000     # 10 of 3
    assert budget(["--calibrate", "cal-9"], CALIBRATE_LAUNCHES="6") == 2700
    assert budget(["--calibrate", "cal-9"], CALIBRATE_LAUNCHES="3") == 1800
    assert budget(["--calibrate", "cal-9"], HARD_TIMEOUT_S="600") == 600
    assert budget(["--calibrate", "cal-9"], IJ_BUDGET_S="900") == 900


def test_a_baseline_gets_a_budget_for_its_reference_launches(tmp_path):
    """A real engine's baseline is B and then reference.launches - 1 more cold
    launches that answer the quality prompts. Each of those adds its own two
    limits (the ready wait, the quality limit) to the run's budget; the mock's
    baseline has none and keeps run_s; an override still wins."""
    root = make_bench_root(tmp_path, allow_baseline=True)
    line = 'reference: {dir: "reference/tps-mix"}'
    assert line in MIX_TARGET
    (root / "targets" / "tps-mix.yaml").write_text(MIX_TARGET)
    (root / "targets" / "tps-cold.yaml").write_text(
        MIX_TARGET.replace("id: tps-mix", "id: tps-cold").replace(
            line, 'reference: {dir: "reference/tps-cold", launches: 3}'))

    def budget(args, target="tps-cold", **env):
        out = run_sh(root, ["--dry-run", *args],
                     {"BENCH_TARGET": target, "HARD_TIMEOUT_S": "", **env})
        assert out.returncode == 0, out.stderr
        return int(re.search(r"kill the whole run after (\d+)s", out.stdout).group(1))

    assert budget(["baseline-mock"], "tps-mix") == 1800            # run_s
    # run_s and twice (ready 60 + 60, quality 900 + compile 60)
    assert budget(["baseline-mock"]) == 1800 + 2 * (120 + 960)
    assert budget(["baseline-mock"], QUALITY_TIMEOUT_S="100") == 1800 + 2 * (120 + 100)
    assert budget(["baseline-mock"], HEALTH_TIMEOUT_S="30") == 1800 + 2 * (30 + 960)
    assert budget(["exp-alpha"]) == 1800                           # not a baseline
    assert budget(["--calibrate", "baseline-mock"]) == 4500        # nor is this
    assert budget(["baseline-mock"], HARD_TIMEOUT_S="600") == 600


def test_dry_run_waits_for_ready_as_long_as_the_target_says(tmp_path):
    """The ready wait is compile_s + health_s unless the target's server
    answers before it compiles (health.ready_timeout_s); the compile budget
    still reaches the warm sweep, where such a server spends it."""
    def waits(plan):
        return set(re.findall(r"until 200 \(max (\d+)s\)", plan))

    for sub in "abc":
        (tmp_path / sub).mkdir()
    assert waits(_plan(tmp_path / "a", MIX_TARGET)) == {"120"}
    lazy = MIX_TARGET + "health: {ready_timeout_s: 45}\n"
    plan = _plan(tmp_path / "b", lazy)
    assert waits(plan) == {"45"}
    warm = next(l for l in plan.splitlines() if "--phase warm" in l)
    assert "--warm-cycles 1" in warm
    root = make_bench_root(tmp_path / "c")
    (root / "targets" / "tps-mix.yaml").write_text(lazy)
    out = run_sh(root, ["--dry-run", "exp-alpha"],
                 {"BENCH_TARGET": "tps-mix", "HEALTH_TIMEOUT_S": "7"})
    assert waits(out.stdout) == {"7"}


@pytest.mark.slow
def test_a_server_that_exits_at_start_does_not_use_the_ready_wait(tmp_path):
    """A launch.sh that dies is seen at the next poll: the run ends there,
    not when the ready wait (here 20 minutes) runs out."""
    from conftest import free_port
    base = Path(os.path.realpath(tmp_path))
    root = make_bench_root(base, allow_baseline=True)
    (root / "targets" / "tps-mix.yaml").write_text(
        MIX_TARGET.replace("compile_s: 60, health_s: 60", "compile_s: 600, health_s: 600"))
    exp = base / "src" / "research" / "experiments" / "baseline-dead"
    exp.mkdir(parents=True)
    (exp / "launch.sh").write_text("#!/usr/bin/env bash\necho no such flag >&2\nexit 3\n")
    for name in ("tpu_inference-src", "vllm-src"):
        (base / "src" / name).mkdir()
    out = run_sh(root, ["--local", "baseline-dead"], extra_env={
        "BENCH_TARGET": "tps-mix", "RESEARCH_ROOT": str(base / "src" / "research"),
        "IJ_SRC_ROOT": str(base / "src"),
        "PORT": str(free_port())})
    assert out.returncode == 0, out.stderr[-3000:]
    res = results_of(root, "baseline-dead")
    assert res["verdict"] == "BUILD_FAILED", res
    assert res["duration_s"] < 60, res["duration_s"]


def test_dry_run_without_a_screen_window_is_the_four_launch_plan(tmp_path):
    plan = _plan(tmp_path,
                 MIX_TARGET.replace("  screen_window_s: 2\n", ""))
    assert len(_launch_lines(plan)) == 4
    assert "launches/C1/launch.json" in next(
        l for l in plan.splitlines() if "compare.py tps-screen" in l)


def test_dry_run_shows_the_private_compile_cache(tmp_path):
    """The one writable mount is a copy of the ruler's seed. An incumbent
    launch has its own, and what it compiled is added to the seed. The
    candidate has one for its launches of this run, so what it compiled in S
    is there for C1 and C2; that copy is never added to the seed."""
    target = MIX_TARGET.replace(
        "  port: 8000\n",
        "  port: 8000\n  compile_cache: {seed: /opt/bench/cache/t, mount: /root/.cache/x}\n")
    plan = _plan(tmp_path, target)
    for line in _launch_lines(plan):
        label = line.split("--name ", 1)[1].split()[0].rsplit("-", 1)[1]
        stage = "candidate" if label in ("S", "C1", "C2") else label
        rw = [m for m in _mounts(line) if not m.endswith(":ro")]
        assert rw == [f"<private-tmp>/cache/{stage}:/root/.cache/x:rw"], rw
    # copied from the seed once, not once per launch
    assert plan.count("/opt/bench/cache/t/ <private-tmp>/cache/candidate/") == 1
    assert "<private-tmp>/cache/S" not in plan and "<private-tmp>/cache/C1" not in plan
    assert "--ignore-existing <private-tmp>/cache/I1/ /opt/bench/cache/t/" in plan
    assert "--ignore-existing <private-tmp>/cache/candidate/" not in plan
    assert "--ignore-existing <private-tmp>/cache/C1/" not in plan
    assert "--ignore-existing <private-tmp>/cache/S/" not in plan
    # an incumbent's copy goes after its launch; the candidate's stays to the end
    assert "rm -rf /s/I1" in plan and "rm -rf /s/candidate" not in plan


def _sweeps(plan: str, label: str) -> list[str]:
    return [l for l in plan.splitlines()
            if "--mode tps" in l
            and f"/launches/{label}/" in l.split("--out ", 1)[1].split()[0]]


def _excluded(line: str) -> set[str]:
    """launch/sweep of every records file a sweep leaves out."""
    return {"/".join(path.split("/")[-3:-1])
            for path in re.findall(r"--pool-exclude (\S+)", line)}


def test_dry_run_sends_the_candidate_no_prompt_twice(tmp_path):
    """The candidate keeps a writable cache from launch to launch, so a prompt
    it has answered is not sent to it again: I1 and C1 leave out what S was
    sent, C2 and I2 also what C1 was sent. Each pair of the ABBA still answers
    the same prompts (same seed, same exclusions)."""
    plan = _plan(tmp_path, MIX_TARGET)
    seen_s = {"S/warm", "S/b1"}
    seen_c1 = {"C1/warm", "C1/b1", "C1/b2"}
    assert [_excluded(l) for l in _sweeps(plan, "S")] == [set(), {"S/warm"}]
    for a, b, earlier in (("I1", "C1", seen_s), ("I2", "C2", seen_s | seen_c1)):
        mine, theirs = _sweeps(plan, a), _sweeps(plan, b)
        assert len(mine) == len(theirs) == 3
        for x, y, own in zip(mine, theirs, ([], ["warm"], ["b1"])):
            assert _excluded(x) == earlier | {f"{a}/{s}" for s in own}
            assert _excluded(y) == earlier | {f"{b}/{s}" for s in own}
            seed = re.search(r"--seed (\d+)", x).group(1)
            assert f"--seed {seed} " in y


def test_dry_run_ici_keeps_both_incumbent_launches_on_the_same_prompts(tmp_path):
    """With one candidate launch after the screen nothing follows C1 that
    could have seen its prompts, so I2 answers what I1 and C1 answered."""
    plan = _plan(tmp_path, MIX_TARGET.replace("design: abba",
                                                           "design: ici"))
    for label in ("I1", "C1", "I2"):
        assert _excluded(_sweeps(plan, label)[0]) == {"S/warm", "S/b1"}


def test_dry_run_gives_the_canary_and_the_gate_the_compile_budget(tmp_path):
    """The canary's and the quality prompts need not fall on a shape the warm
    sweep compiled: one request may wait as long as a compile (budgets.compile_s,
    60 here), and one such timeout is a dead server. An override still wins."""
    plan = _plan(tmp_path, MIX_TARGET)
    canary = next(l for l in plan.splitlines() if "--mode canary" in l)
    assert "--canary-timeout 180 " in canary and "--canary-max-timeouts 1 " in canary
    gates = [l for l in plan.splitlines() if "quality2.py gate" in l]
    assert gates and all("--request-timeout 180 " in l for l in gates)
    (tmp_path / "o").mkdir()
    root = make_bench_root(tmp_path / "o")
    (root / "targets" / "tps-mix.yaml").write_text(MIX_TARGET)
    out = run_sh(root, ["--dry-run", "exp-alpha"],
                 {"BENCH_TARGET": "tps-mix", "CANARY_TIMEOUT_S": "5"})
    canary = next(l for l in out.stdout.splitlines() if "--mode canary" in l)
    assert "--canary-timeout 5 " in canary and "--canary-max-timeouts" not in canary


def _seeds(plan: str, label: str) -> list[int]:
    return [int(re.search(r"--seed (\d+)", l).group(1)) for l in _sweeps(plan, label)]


def test_dry_run_a_calibration_draws_its_prompts_as_a_run_does(tmp_path):
    """sigma_L is the scale of the drift check, I2 against I1. In an ABBA run
    those two answer different prompts (I2 leaves out what C1, so I1, was
    sent), so a calibration's launches do too: each has its own draw and
    leaves out what the launch before it was sent. Replaying one set would
    leave the prompts out of sigma_L and void honest runs."""
    for sub in "ab":
        (tmp_path / sub).mkdir()
    root = make_bench_root(tmp_path / "a")
    (root / "targets" / "tps-mix.yaml").write_text(MIX_TARGET)
    out = run_sh(root, ["--dry-run", "--calibrate", "cal-9"],
                 {"BENCH_TARGET": "tps-mix", "CALIBRATE_LAUNCHES": "4"})
    assert out.returncode == 0, out.stderr
    plan = out.stdout
    warm_seeds = []
    for i in range(1, 5):
        me, before = f"K{i}", {f"K{i - 1}/{s}" for s in ("warm", "b1", "b2")} if i > 1 else set()
        sweeps = _sweeps(plan, me)
        assert [_excluded(l) for l in sweeps] == [
            before, before | {f"{me}/warm"}, before | {f"{me}/b1"}], me
        seeds = _seeds(plan, me)
        assert seeds[1:] == [seeds[0] + 1, seeds[0] + 2]
        warm_seeds.append(seeds[0])
    # four draws: no two launches share a seed, on any sweep
    assert len({s + b for s in warm_seeds for b in range(3)}) == 12
    # a candidate run is as before: one seed for all of its launches
    plan = _plan(tmp_path / "b", MIX_TARGET)
    assert len({_seeds(plan, l)[0] for l in ("S", "I1", "C1", "C2", "I2")}) == 1


def test_dry_run_an_ici_calibration_replays_one_draw(tmp_path):
    """In an ICI run I1, C1 and I2 answer the same prompts, so the drift check
    there sees launch noise alone and its calibration replays one set."""
    root = make_bench_root(tmp_path)
    (root / "targets" / "tps-mix.yaml").write_text(
        MIX_TARGET.replace("design: abba", "design: ici"))
    out = run_sh(root, ["--dry-run", "--calibrate", "cal-9"],
                 {"BENCH_TARGET": "tps-mix", "CALIBRATE_LAUNCHES": "3"})
    assert out.returncode == 0, out.stderr
    for i in range(1, 4):
        me = f"K{i}"
        assert [_excluded(l) for l in _sweeps(out.stdout, me)] == [
            set(), {f"{me}/warm"}, {f"{me}/b1"}]
    assert len({_seeds(out.stdout, f"K{i}")[0] for i in range(1, 4)}) == 1


def _baseline_plan(tmp_path, target: str) -> str:
    root = make_bench_root(tmp_path, allow_baseline=True)
    (root / "targets" / "tps-mix.yaml").write_text(target)
    out = run_sh(root, ["--dry-run", "baseline-mock"], {"BENCH_TARGET": "tps-mix"})
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_dry_run_the_mock_baseline_writes_r_from_its_one_server(tmp_path):
    """One mock server measured three times stands in for three launches; the
    length of a continuation and the wait for a compile come from the target."""
    plan = _baseline_plan(tmp_path, MIX_TARGET)
    make = [l for l in plan.splitlines() if "quality2.py make-reference" in l]
    assert len(make) == 1 and "quality2.py collect" not in plan
    assert "--base-url " in make[0] and "--launches-from" not in make[0]
    assert "--max-tokens 64 " in make[0] and "--request-timeout 180 " in make[0]
    assert [l.split("--name ", 1)[1].split()[0].rsplit("-", 1)[1]
            for l in _launch_lines(plan)] == ["B"]


def test_dry_run_a_real_baseline_writes_r_from_cold_launches(tmp_path):
    """R's floors are the reference's own launch-to-launch noise, so it is
    launched as often as the target says: the first is the baseline's measured
    launch, the others only answer the quality prompts, anchored to the first.
    They start from the seed the baseline's launch has just added to."""
    target = MIX_TARGET.replace(
        'reference: {dir: "reference/tps-mix"}',
        'reference: {dir: "reference/tps-mix", launches: 3, max_tokens: 128}').replace(
        "  port: 8000\n",
        "  port: 8000\n  compile_cache: {seed: /opt/bench/cache/t, mount: /root/.cache/x}\n")
    assert "launches: 3" in target
    lines = _baseline_plan(tmp_path, target).splitlines()
    plan = "\n".join(lines)
    starts = {l.split("--name ", 1)[1].split()[0].rsplit("-", 1)[1]: i
              for i, l in enumerate(lines) if "docker run -d" in l}
    assert list(starts) == ["B", "R1", "R2"]
    collects = [(i, l) for i, l in enumerate(lines) if "quality2.py collect" in l]
    assert len(collects) == 3
    for (i, line), label in zip(collects, starts):
        assert f"-{label} " in line.split("--network container:", 1)[1].split(" -u ")[0] + " "
        assert "--max-tokens 128 " in line and "--request-timeout 180 " in line
    dumps = [re.search(r"--out (\S+)", l).group(1) for _, l in collects]
    assert "--anchor" not in collects[0][1]
    assert all(f"--anchor {dumps[0]} " in l for _, l in collects[1:])
    make = [l for l in lines if "quality2.py make-reference" in l]
    assert len(make) == 1 and "--base-url" not in make[0]
    assert f"--launches-from {','.join(dumps)} " in make[0]
    assert "network container:" not in make[0]         # no server is needed for it
    # the baseline's compiled shapes are in the seed before R1 copies it
    merged = next(i for i, l in enumerate(lines)
                  if "--ignore-existing <private-tmp>/cache/B/ /opt/bench/cache/t/" in l)
    assert merged < starts["R1"]
    assert "--ignore-existing <private-tmp>/cache/R1/" not in plan
    for l in lines:
        if "--mode tps" in l or "--mode canary" in l:
            assert "/launches/B" in l                   # only B is measured


@pytest.mark.slow
def test_tps_a_sized_mix_end_to_end(tmp_path_factory):
    """Baseline, calibration, a faster candidate promoted through the screen
    launch with every size reported, and then an honest one (now twice as
    slow as the best) that costs one launch and never starts the incumbent."""
    from conftest import free_port
    base = Path(os.path.realpath(tmp_path_factory.mktemp("mix")))
    root = make_bench_root(base, allow_baseline=True)
    (root / "targets" / "tps-mix.yaml").write_text(MIX_TARGET)
    exps = base / "src" / "research" / "experiments"
    launch = BENCH.parent / "research" / "experiments" / "baseline-mock" / "launch.sh"
    for exp_id, behaviour in (("baseline-mock", None), ("fast-1", "fast"),
                              ("same-1", None)):
        (exps / exp_id).mkdir(parents=True)
        shutil.copy(launch, exps / exp_id / "launch.sh")
        if behaviour:
            (exps / exp_id / "behaviour").write_text(behaviour + "\n")
    tree = base / "src" / "tpu_inference-src"
    (tree / ".git").mkdir(parents=True)
    (tree / ".git" / "big").write_text("history\n")
    (tree / "runner.py").write_text("# the engine as the baseline saw it\n")
    env = {"BENCH_TARGET": "tps-mix", "RESEARCH_ROOT": str(base / "src" / "research"),
           "IJ_SRC_ROOT": str(base / "src"),
           "PORT": str(free_port()), "CALIBRATE_LAUNCHES": "2"}
    bench = (root, env)

    res = _tps_run(bench, ["baseline-mock"])
    assert res["verdict"] == "BASELINE", res["reasons"]
    m = res["metric"]
    assert set(m["by_isl"]) == {"16", "32", "64"}
    assert 60 < m["decode_tps"] < 140              # the mock: 8 ms per token
    # the score is the whole answer: 32 tokens in 150 ms + 31 x 8 ms, ~80 tok/s
    assert 50 < m["stream_tps"] < 95 and m["stream_tps"] < m["decode_tps"]
    assert m["headline"] == pytest.approx(m["stream_tps"], rel=1e-3)   # one chip
    kept = root / "results" / "baseline-mock" / "snapshot" / "src" / "tpu_inference"
    assert (kept / "runner.py").exists() and not (kept / ".git").exists()
    assert (root / "results" / "baseline-mock" / "snapshot" / "tpu_inference.head").exists()
    warm = [json.loads(l) for l in (root / "results" / "baseline-mock" / "launches"
                                    / "B" / "warm" / "tps_records.jsonl").open()]
    assert [r["isl"] for r in warm[:3]] == [16, 32, 64]

    assert _tps_run(bench, ["--calibrate", "cal-1"])["verdict"] == "CALIBRATED"

    fast = _tps_run(bench, ["fast-1"])
    assert fast["verdict"] == "PROMOTED", fast["reasons"]
    assert list(fast["launches"]) == ["C1", "C2", "I1", "I2", "S"]
    assert fast["screen"]["passed"] is True
    assert fast["metric"]["guards"]["incumbent_by_isl"]["passed"] is True
    assert fast["launches"]["C1"]["by_isl"]["64"]["p50_ttft_ms"] > 0
    assert fast["metric"]["stream_tps"] > 1.5 * m["stream_tps"]
    assert fast["metric"]["decode_tps"] > 1.5 * m["decode_tps"]
    summary = (root / "results" / "fast-1" / "summary.txt").read_text()
    assert "one stream" in summary and "isl 64" in summary
    assert "for the whole answer" in summary and "whole-answer speed per chip" in summary
    board = json.loads((root / "leaderboard" / "tps-mix.json").read_text())
    assert board["best"]["id"] == "fast-1" and set(board["best"]["by_isl"]) == set(m["by_isl"])

    same = _tps_run(bench, ["same-1"])
    assert same["verdict"] == "REJECTED_SCREEN", same
    assert list(same["launches"]) == ["S"]
    assert same["duration_s"] < fast["duration_s"] / 2


SEAL2_TARGET = """\
id: seal2-test
runnable: true
accelerator: {kind: none, chips: 1, hosts: 1}
model: {served_name: mock, revision: "0", local_dir: "", tokenizer: "ws"}
engine:
  name: mock
  image: "python:3.12-slim"
  docker_flags: [FLAGS]
  port: 8000
  sources: [tpu_inference, vllm]
  compile_cache: {seed: SEED, mount: /cache}
metric:
  kind: tps
  isl: 64
  osl: 32
  concurrency: [4]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 0.3
  window_s: 1.5
  design: abba
  blocks: 2
budgets: {build_s: 120, compile_s: 0, health_s: 15, run_s: 300}
reference: {dir: "reference/seal2-test"}
"""
SEAL2_PROBE = SEAL_PROBE.replace(
    '"/opt/inference-judge/mock_server.py", "/out"',
    '"/ro/src/tpu_inference", "/cache", "/plugin"').replace(
    'print("SEALPROBE', 'res["cache"] = sorted(f for f in os.listdir("/cache") if f[0] != ".")\n'
    'open("/cache/from-the-candidate", "w").write("x")\nprint("SEALPROBE')


def _seal2(tmp_path_factory, flags: str):
    base = Path(os.path.realpath(tmp_path_factory.mktemp("seal2")))
    root = make_bench_root(base, allow_baseline=True)
    seed = base / "seed"
    seed.mkdir()
    (seed / "compiled-yesterday").write_text("x")
    (base / "plugin").mkdir()
    (root / "targets" / "seal2-test.yaml").write_text(
        SEAL2_TARGET.replace("SEED", str(seed)).replace("FLAGS", flags.replace(
            "PLUGIN", str(base / "plugin"))))
    exp = base / "src" / "research" / "experiments" / "baseline-probe"
    exp.mkdir(parents=True)
    (exp / "launch.sh").write_text(SEAL2_PROBE)
    (base / "src" / "tpu_inference-src").mkdir()
    (base / "src" / "tpu_inference-src" / "runner.py").write_text("x = 1\n")
    out = run_sh(root, ["baseline-probe"], extra_env={
        "BENCH_TARGET": "seal2-test", "IJ_RULER_IMAGE": "python:3.12-slim",
        "RESEARCH_ROOT": str(base / "src" / "research"),
        "IJ_SRC_ROOT": str(base / "src")})
    assert out.returncode == 0, out.stderr[-3000:]
    results = results_of(root, "baseline-probe")
    logs = "".join(p.read_text(errors="replace") for p in
                   (root / "results" / "baseline-probe").rglob("*.log"))
    line = next(l for l in logs.splitlines() if l.startswith("SEALPROBE "))
    return results, json.loads(line.split(" ", 1)[1]), base


@pytest.mark.slow
@pytest.mark.skipif(not _have_image("python:3.12-slim"),
                    reason="needs docker and python:3.12-slim")
def test_a_sealed_launch_gets_the_targets_trees_and_a_private_cache(
        tmp_path_factory):
    """The cache is the one mount a launch may write: a copy of the seed that
    the launch can fill and the seed never sees. A read-only mount from the
    target changes nothing; a host device is named, the seal is not called
    full, and the clean probe proves nothing about what the device reaches."""
    import socket
    results, seen, base = _seal2(tmp_path_factory,
                                 '--device, /dev/null, -v, "PLUGIN:/plugin:ro"')
    assert seen["write"] == {"/ro/exp": "refused", "/ro/src/vllm": "refused",
                             "/ro/src/sglang": "absent", "/plugin": "refused",
                             "/ro/src/tpu_inference": "refused", "/cache": "writable"}
    assert seen["cache"] == ["compiled-yesterday"]
    assert sorted(p.name for p in (base / "seed").iterdir()) == ["compiled-yesterday"]
    iso = results["isolation"]
    assert iso["network"] == "none", iso
    assert iso["seal"] == "weaker: device /dev/null", iso
    assert iso["exfil_probe"].startswith("not_proven: device /dev/null"), iso
    assert iso["probe"]["net"] == "blocked"
    assert iso["writable_mounts"] == ["/cache"]
    assert iso["private_mounts"]["/cache"]["write_reached_seed"] is False
    assert iso["host_sockets"] == []
    assert results["cleanup_ok"] is True


@pytest.mark.slow
@pytest.mark.skipif(not _have_image("python:3.12-slim"),
                    reason="needs docker and python:3.12-slim")
def test_a_privileged_launch_is_not_called_sealed(tmp_path_factory):
    """The probe cannot see what a privileged container can reach, so a
    blocked probe proves nothing; a socket the target mounts is named, and
    its being there is enough for the seal not to be called full (nobody
    was listening on this one: a host service can start at any time)."""
    import socket
    base = Path(os.path.realpath(tmp_path_factory.mktemp("sock")))
    cwd = os.getcwd()
    os.chdir(base)                       # AF_UNIX paths are short
    try:
        s = socket.socket(socket.AF_UNIX)
        s.bind("vbar.sock")
    finally:
        os.chdir(cwd)
    results, seen, _ = _seal2(tmp_path_factory,
                              f'--privileged, -v, "{base}:/plugin:ro"')
    s.close()
    iso = results["isolation"]
    assert iso["seal"] == "weaker: privileged, host socket /plugin/vbar.sock"
    assert iso["exfil_probe"].startswith(
        "not_proven: privileged, host socket /plugin/vbar.sock"), iso
    assert iso["host_sockets"] == ["/plugin/vbar.sock"]
    assert iso["probe"]["sockets"]["/plugin/vbar.sock"] != "connected"

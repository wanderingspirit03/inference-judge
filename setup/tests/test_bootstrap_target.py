"""bootstrap.sh is driven by the target file (DESIGN 2.1).

The interface under test is bootstrap's command line: `--target FILE --plan`
prints the steps the target would run or skip, and nothing else happens.  The
plan is the same list main() executes (plan_steps), so what is asserted here is
what a real box would do.  Two internals are tested in library mode
(IJ_BOOTSTRAP_LIB=1): the Hub-manifest verifier and the revision gate.
Every run is shimmed (conftest); target.py is read with the bench venv's python.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import GPU_BUILT_TARGET, REPO, make_repo

pytestmark = pytest.mark.skipif(os.getuid() == 0, reason="bootstrap would proceed past preflight as root")

def repo_with_targets(sb) -> Path:
    repo = make_repo(sb, with_git=False)
    shutil.copytree(REPO / "targets", repo / "targets")
    shutil.copy(REPO / "bench" / "target.py", repo / "bench" / "target.py")
    (repo / "targets" / "gpu-built.yaml").write_text(GPU_BUILT_TARGET)
    return repo


def boot(sb, repo: Path, *args: str, **env) -> subprocess.CompletedProcess:
    return sb.run([str(repo / "setup" / "bootstrap.sh"), *args], IJ_REPO=str(repo),
                  IJ_PINS_FILE=str(sb.root / "pins"), IJ_TARGET_PYTHON=sys.executable, **env)


def plan(sb, target: str, *extra: str) -> tuple[list[str], list[str]]:
    repo = repo_with_targets(sb)
    r = boot(sb, repo, "--target", f"targets/{target}.yaml", "--plan", *extra)
    assert r.returncode == 0, r.stderr
    run = [l[6:] for l in r.stdout.splitlines() if l.startswith("RUN ")]
    skip = [l[6:] for l in r.stdout.splitlines() if l.startswith("SKIP ")]
    assert run and skip
    return run, skip


def has(lines: list[str], text: str) -> bool:
    return any(text in l for l in lines)


def test_cpu_target_skips_every_accelerator_and_engine_build_step(sb):
    run, skip = plan(sb, "cpu-mock")
    assert has(skip, "accelerator driver") and has(skip, "nsys")
    assert not has(run, "gpu") and not has(run, "GPU") and not has(run, "nsys")
    assert has(skip, "engine build") and not has(run, "clone vllm") and not has(run, "inference-judge-base")
    assert has(run, "engine image inference-judge-mock:latest (targets/cpu-mock.Dockerfile)")
    assert has(skip, "model download  -- target names no model.hf_id")
    assert has(run, "harden") and has(run, "install the target") and has(run, "research clock")


# ------------------------------------------------------------------ the agent
def plan_out(sb, *extra: str, **env) -> subprocess.CompletedProcess:
    return boot(sb, repo_with_targets(sb), "--target", "targets/cpu-mock.yaml", "--plan", *extra, **env)


def test_the_agent_defaults_to_claude_and_the_plan_names_it(sb):
    r = plan_out(sb)
    assert r.returncode == 0, r.stderr
    assert "agent: claude" in r.stdout
    assert "RUN   agent claude (adapters/claude/setup.sh install)" in r.stdout
    assert "RUN   node 22 + npm" in r.stdout


@pytest.mark.parametrize("agent", ["codex", "pi"])
def test_the_plan_installs_the_agent_the_operator_picks(sb, agent):
    repo = repo_with_targets(sb)
    args = ("--target", "targets/cpu-mock.yaml", "--plan")
    r = boot(sb, repo, *args, "--agent", agent)
    assert r.returncode == 0, r.stderr
    assert f"agent: {agent}" in r.stdout
    assert f"RUN   agent {agent} (adapters/{agent}/setup.sh install)" in r.stdout
    assert "claude" not in "".join(l for l in r.stdout.splitlines() if "agent" in l)
    # the same through the environment, and the flag wins over it
    assert f"agent: {agent}" in boot(sb, repo, *args, IJ_AGENT=agent).stdout
    assert f"agent: {agent}" in boot(sb, repo, *args, f"--agent={agent}", IJ_AGENT="claude").stdout


def test_an_agent_already_on_the_path_is_not_installed_and_needs_no_node(sb):
    stub = sb.shims / "codex"
    stub.write_text("#!/bin/bash\necho codex-stub 1.0\n")
    stub.chmod(0o755)
    r = plan_out(sb, "--agent", "codex")
    assert r.returncode == 0, r.stderr
    assert f"SKIP  agent codex  -- codex already on PATH: {stub}" in r.stdout
    assert "SKIP  node 22 + npm  -- codex already on PATH" in r.stdout


@pytest.mark.parametrize("name", ["gemini", "TEMPLATE/../claude", "-x", ""])
def test_an_agent_with_no_adapter_is_refused_and_the_adapters_are_listed(sb, name):
    r = plan_out(sb, "--agent", name)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "claude codex pi" in r.stderr and "TEMPLATE" not in r.stderr.split("adapters here:")[1]
    assert "RUN " not in r.stdout


def test_an_adapter_without_setup_sh_is_planned_as_a_step_the_operator_does(sb):
    repo = repo_with_targets(sb)
    shutil.copytree(repo / "research" / "adapters" / "TEMPLATE", repo / "research" / "adapters" / "bare")
    r = boot(sb, repo, "--target", "targets/cpu-mock.yaml", "--plan", "--agent", "bare")
    assert r.returncode == 0, r.stderr
    assert "RUN   agent bare (no adapters/bare/setup.sh: install the `bare` command yourself)" in r.stdout
    assert "SKIP  node 22 + npm  -- the bare adapter has no setup.sh" in r.stdout


def test_help_names_the_agent_flag(sb):
    r = boot(sb, repo_with_targets(sb), "--help")
    assert r.returncode == 0 and "--agent NAME" in r.stdout and "IJ_AGENT" in r.stdout


def test_gpu_target_keeps_the_gpu_branch(sb):
    run, _ = plan(sb, "gpu-built")
    for step in ("gpu: nvidia-smi works", "GPU passthrough", "nsight-systems", "clone vllm and sglang",
                 "build inference-judge-base:latest", "Hub sha256 manifest", "research clock"):
        assert has(run, step), step


def test_tpu_target_runs_the_tpu_steps_and_pulls_its_image(sb):
    run, skip = plan(sb, "sketch-tpu-8chip")
    assert has(run, "tpu: the host shows all 8 chips") and has(run, "tpu: chip gate")
    assert has(run, "engine image (docker load from the disk's tar, else pull; digest checked)")
    assert not has(run, "(pull)") and has(skip, "engine build")      # never an unchecked pull on a tpu
    assert has(run, "Hub sha256 manifest")             # this target names model.hf_id
    assert not has(run, "nsight") and not has(run, "TODO")


def test_a_prebuilt_image_off_a_tpu_is_still_a_plain_pull(sb):
    repo = repo_with_targets(sb)
    assert '"inference-judge-base:latest"' in GPU_BUILT_TARGET
    (repo / "targets" / "gpu-pulled.yaml").write_text(
        GPU_BUILT_TARGET.replace('"inference-judge-base:latest"', '"vllm/vllm-openai:v0.10.0"'))
    r = boot(sb, repo, "--target", "targets/gpu-pulled.yaml", "--plan")
    assert r.returncode == 0, r.stderr
    run = [l[6:] for l in r.stdout.splitlines() if l.startswith("RUN ")]
    assert has(run, "engine image vllm/vllm-openai:v0.10.0 (pull)") and not has(run, "docker load")


def plan_lines(sb, target: str, *extra: str, **env) -> list[str]:
    repo = repo_with_targets(sb)
    r = boot(sb, repo, "--target", f"targets/{target}.yaml", "--plan", *extra, **env)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_the_plain_gpu_target_pulls_its_image_and_gates_on_it(sb):
    lines = plan_lines(sb, "gpu-gemma4-26b-tps")
    run = [l[6:] for l in lines if l.startswith("RUN ")]
    skip = [l[6:] for l in lines if l.startswith("SKIP ")]
    assert has(run, "gpu: nvidia-smi works and shows 1 GPU(s)")
    assert has(run, "docker: daemon") and not has(run, "GPU passthrough")      # proved in the engine image instead
    assert has(skip, "engine build") and not has(run, "inference-judge-base") and not has(run, "engine channel")
    assert has(run, "engine image vllm/vllm-openai@sha256:") and has(run, "(pull)")
    assert has(run, "gpu: gate (1 GPU(s) visible in the engine image)")
    assert has(run, "engine source trees (vllm) at the target's pinned refs") and not has(run, "sglang")
    assert has(run, "model google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52 + Hub sha256 manifest")
    assert not has(run, "tpu") and not has(run, "cache disk") and not has(run, "compile-cache")
    # a sketch stops after the gate, with every later step listed, like a tpu sketch
    order = [l for l in lines if l.startswith(("RUN ", "STOP "))]
    gate = [i for i, l in enumerate(order) if "gpu: gate" in l][0]
    assert order[gate + 1].startswith("STOP  target is runnable: false")
    assert has(run, "research clock")


def test_the_plain_gpu_target_needs_no_engine_pins(sb):
    repo = repo_with_targets(sb)
    r = boot(sb, repo, "--target", "targets/gpu-gemma4-26b-tps.yaml")
    assert "IJ_VLLM_TAG" not in r.stderr and "IJ_SGLANG_TAG" not in r.stderr
    assert "preflight" in r.stdout and "must run as root" in r.stderr and r.returncode == 1


def test_a_runnable_gpu_target_with_a_prebuilt_image_has_no_stop(sb):
    repo = repo_with_targets(sb)
    (repo / "targets" / "gpu-pulled.yaml").write_text(
        GPU_BUILT_TARGET.replace('"inference-judge-base:latest"', '"vllm/vllm-openai:v0.10.0"'))
    r = boot(sb, repo, "--target", "targets/gpu-pulled.yaml", "--plan")
    assert r.returncode == 0, r.stderr
    assert "STOP" not in r.stdout and "RUN   gpu: gate" in r.stdout and "RUN   research clock" in r.stdout


def test_a_built_engine_that_names_one_tree_clones_one_and_needs_one_pin(sb):
    repo = repo_with_targets(sb)
    assert "sources: [vllm, sglang]" in GPU_BUILT_TARGET
    (repo / "targets" / "gpu-built-one.yaml").write_text(
        GPU_BUILT_TARGET.replace("sources: [vllm, sglang]", "sources: [vllm]"))
    r = boot(sb, repo, "--target", "targets/gpu-built-one.yaml", "--plan")
    assert r.returncode == 0, r.stderr
    assert "RUN   clone vllm at release tags" in r.stdout and "sglang at" not in r.stdout
    r = boot(sb, repo, "--target", "targets/gpu-built-one.yaml", IJ_VLLM_TAG="v0.29.0")
    assert "IJ_SGLANG_TAG" not in r.stderr and "preflight" in r.stdout      # past the pin check, into step 1


def test_the_gpu_target_is_judged_like_the_tpu_target():
    """Same model at the same revision, same metric block, pools and quality settings."""
    import yaml
    gpu = yaml.safe_load((REPO / "targets" / "gpu-gemma4-26b-tps.yaml").read_text())
    tpu = yaml.safe_load((REPO / "targets" / "v6e4-gemma4-26b-tps.yaml").read_text())
    assert gpu["metric"] == tpu["metric"] and gpu["quality"] == tpu["quality"]
    for k in ("prompts", "tasks", "max_tokens", "launches"):
        assert gpu["reference"][k] == tpu["reference"][k], k
    assert gpu["reference"]["dir"] == "reference/gpu-gemma4-26b-tps"
    assert (gpu["model"]["hf_id"], gpu["model"]["revision"]) == (tpu["model"]["hf_id"], tpu["model"]["revision"])
    assert gpu["accelerator"]["kind"] == "gpu" and gpu["accelerator"]["chips"] == 1
    assert gpu["runnable"] is False
    assert gpu["engine"]["name"] == "vllm" and "@sha256:" in gpu["engine"]["image"]
    assert "MOE_REQUANTIZE_WEIGHT_DTYPE" not in gpu["engine"].get("env", {})     # full precision
    assert "compile_cache" not in gpu["engine"] and "cache_disk" not in gpu.get("provision", {})


def test_skip_model_and_no_clock_are_honoured(sb):
    run, skip = plan(sb, "sketch-tpu-8chip", "--skip-model", "--no-clock")
    assert has(skip, "model download  -- --skip-model")
    assert has(skip, "research clock  -- --no-clock") and not has(run, "research clock")


def test_a_target_that_is_not_runnable_never_gets_the_clock(sb):
    repo = repo_with_targets(sb)
    mock = (repo / "targets" / "cpu-mock.yaml").read_text()
    assert "\nrunnable: true\n" in mock
    (repo / "targets" / "cpu-sketch.yaml").write_text(mock.replace("\nrunnable: true\n", "\nrunnable: false\n"))
    r = boot(sb, repo, "--target", "targets/cpu-sketch.yaml", "--plan")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert has(lines, "SKIP  research clock  -- target is runnable: false") and not has(lines, "RUN   research clock")
    assert not has(lines, "STOP")                      # only a tpu target is finished on the machine


def test_a_target_must_be_named(sb):
    repo = repo_with_targets(sb)
    r = boot(sb, repo)
    assert r.returncode == 2
    assert "--target is required" in r.stderr and "cpu-mock.yaml" in r.stderr
    assert "preflight" not in r.stdout


def test_engine_pins_are_required_only_when_the_engine_is_built_here(sb):
    repo = repo_with_targets(sb)
    built = boot(sb, repo, "--target", "targets/gpu-built.yaml")
    assert built.returncode == 2 and "IJ_VLLM_TAG" in built.stderr
    assert "preflight" not in built.stdout
    mock = boot(sb, repo, "--target", "targets/cpu-mock.yaml")
    assert "IJ_VLLM_TAG" not in mock.stderr
    assert "preflight" in mock.stdout and "must run as root" in mock.stderr   # step 1 ran, as a non-root user
    assert mock.returncode == 1


def test_an_invalid_target_is_refused_before_any_step(sb, tmp_path):
    repo = repo_with_targets(sb)
    bad = repo / "targets" / "broken.yaml"
    bad.write_text("id: broken\naccelerator: {kind: quantum}\n")
    r = boot(sb, repo, "--target", "targets/broken.yaml", "--plan")
    assert r.returncode == 2 and "does not validate" in r.stderr
    assert "RUN" not in r.stdout


# ------------------------------------------------------------ library mode
def lib(script: str, cwd: Path) -> subprocess.CompletedProcess:
    """Source bootstrap.sh without running it; only hashlib/python3 and coreutils are used."""
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(cwd), "IJ_BOOTSTRAP_LIB": "1",
           "LC_ALL": "C"}
    return subprocess.run(["/bin/bash", "-c", f'. "{REPO}/setup/bootstrap.sh"; {script}'],
                          env=env, cwd=cwd, capture_output=True, text=True, timeout=60)


def write_model(tmp_path: Path) -> tuple[Path, Path]:
    d = tmp_path / "model"
    d.mkdir()
    shard = os.urandom(1 << 16)
    (d / "model-00001.safetensors").write_bytes(shard)
    (d / "config.json").write_bytes(b'{"a": 1}\n')
    git_sha1 = hashlib.sha1(b"blob %d\0" % 9 + b'{"a": 1}\n').hexdigest()
    m = tmp_path / "model.sha256"
    m.write_text(f"sha256 {hashlib.sha256(shard).hexdigest()} model-00001.safetensors\n"
                 f"gitsha1 {git_sha1} config.json\n")
    return d, m


def test_manifest_verifier_passes_an_intact_download(tmp_path):
    d, m = write_model(tmp_path)
    r = lib(f'verify_model_manifest "{d}" "{m}" full', tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK 2/2 files verified" in r.stdout


def test_manifest_verifier_fails_a_truncated_shard(tmp_path):
    d, m = write_model(tmp_path)
    shard = d / "model-00001.safetensors"
    shard.write_bytes(shard.read_bytes()[:-100])
    r = lib(f'verify_model_manifest "{d}" "{m}" full', tmp_path)
    assert r.returncode == 1
    assert "FAIL hash mismatch: model-00001.safetensors" in r.stdout


def test_manifest_quick_check_catches_a_missing_file(tmp_path):
    d, m = write_model(tmp_path)
    (d / "config.json").unlink()
    r = lib(f'verify_model_manifest "{d}" "{m}" quick', tmp_path)
    assert r.returncode == 1 and "FAIL missing: config.json" in r.stdout


@pytest.mark.parametrize("rev", ["unpinned", "main", "abc123", ""])
def test_model_download_requires_a_40_hex_revision(tmp_path, rev):
    script = (f'T_ID=t T_MODEL_ID=org/m T_MODEL_REV="{rev}" T_MODEL_DIR="{tmp_path}/m"; '
              f'( set -e; s_model )')
    r = lib(script, tmp_path)
    assert r.returncode == 1
    assert "40-hex commit sha is required" in r.stderr
    assert not (tmp_path / "m").exists()               # refused before touching the disk


# ------------------------------------------- what the ruler's python imports
def test_tokenizers_is_declared_for_the_bench_venv_and_for_the_ruler_image():
    """A tps target counts tokens with the model's tokenizer.json: without the library the
    first measured run is a hard stop, hours into a paid VM."""
    venv = (REPO / "bench" / "pyproject.toml").read_text().split("[project.optional-dependencies]")[0]
    image = (REPO / "targets" / "cpu-mock.Dockerfile").read_text()
    assert '"tokenizers>=' in venv
    assert '"tokenizers>=' in [l for l in image.split("\nRUN ")[1:] if "pip install" in l][0]


def test_this_venv_has_every_library_the_ruler_imports(tmp_path):
    r = lib(f'ruler_libs "{sys.executable}"', tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tokenizers" in r.stdout


def test_a_python_without_the_rulers_libraries_is_named_with_what_it_lacks(tmp_path):
    r = lib(f'ruler_libs "{sys.executable}" -S', tmp_path)       # -S: no site-packages at all
    assert r.returncode == 1
    assert "missing: httpx numpy yaml tokenizers" in r.stderr


def test_check_asks_the_same_question_of_the_venv_and_of_the_ruler_image():
    check = (REPO / "setup" / "check.sh").read_text()
    assert check.count("ruler_libs ") >= 2
    assert '"$BENCH/.venv/bin/python"' in check and "inference-judge-ruler:latest python3" in check


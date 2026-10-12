"""The NVIDIA GPU steps of bootstrap (setup/gpu.sh) and what a gpu target decides.

The interface under test is each step's function as bootstrap runs it: in a
`( set -e; fn )` subshell, with the target loaded from a file.  nvidia-smi,
docker, git, sudo and chown are small fakes in a bin directory that record
their argv and play a scripted outcome, so what is asserted is what the step
decided and told the operator, never what a real GPU would do.  NOTHING HERE
HAS RUN ON A GPU: the whole gpu path is unproven on real hardware (see
setup/README.md).  /bin/bash is the macOS 3.2 on a laptop, so this also proves
the library has no bash 4 in it.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import GPU_BUILT_TARGET, REPO

pytestmark = pytest.mark.skipif(os.getuid() == 0, reason="the fakes stand in for root-only commands")

DIGEST = "sha256:" + "b" * 64
IMAGE = "vllm/vllm-openai@" + DIGEST
SHA = "db9527a46873454610df6dbedf79a36d6bf1a7f6"

FAKES = {
    "nvidia-smi": r"""#!/bin/bash
echo "nvidia-smi $*" >> "$FAKE/smi.log"
n=${FAKE_GPUS:-1}; i=0
while [ "$i" -lt "$n" ]; do
  case "$*" in
    -L) echo "GPU $i: ${FAKE_GPU_NAME:-NVIDIA H100 80GB HBM3} (UUID: GPU-0000-$i)" ;;
    *driver_version*) echo "${FAKE_DRIVER:-580.126.09}" ;;
    *--query-gpu=name*) echo "${FAKE_GPU_NAME:-NVIDIA H100 80GB HBM3}" ;;
    *memory.total*) echo "${FAKE_GPU_MIB:-81559}" ;;
  esac
  i=$((i + 1))
done
exit 0
""",
    "docker": r"""#!/bin/bash
echo "docker $*" >> "$FAKE/docker.log"
case "$1" in
  image) [ -f "$FAKE/image_present" ] || exit 1
         case "$*" in *image.revision*) cat "$FAKE/image_revision" 2>/dev/null ;; esac
         exit 0 ;;
  run) rc=${FAKE_GATE_RC:-0}
       [ "$rc" = 0 ] || { echo "${FAKE_GATE_ERR:-docker: Error response from daemon: could not select device driver}" >&2; exit "$rc"; }
       n=${FAKE_GATE_GPUS:-1}; i=0
       while [ "$i" -lt "$n" ]; do echo "GPU $i: NVIDIA H100 80GB HBM3 (UUID: GPU-0000-$i)"; i=$((i + 1)); done
       exit 0 ;;
esac
exit 0
""",
    # A clone or a fetch+checkout leaves a tree with a .git; nothing is fetched.
    "git": r"""#!/bin/bash
echo "git $*" >> "$FAKE/git.log"
case "$1" in
  clone) mkdir -p "${!#}/.git" ;;
  init) mkdir -p "${!#}/.git" ;;
  fetch) [ -f "$FAKE/fetch_fails" ] && { echo "fatal: remote error: upload-pack: not our ref" >&2; exit 128; } ;;
  describe) echo fake-describe ;;
  status) : ;;
esac
exit 0
""",
    "chown": r"""#!/bin/bash
echo "chown $*" >> "$FAKE/chown.log"
""",
    # sudo -u USER -H bash -lc CMD  ->  bash -c CMD (no login shell: path_helper would reorder PATH)
    "sudo": r"""#!/bin/bash
while [ $# -gt 0 ]; do case "$1" in -u) shift 2 ;; -H|-n) shift ;; *) break ;; esac; done
[ "$1" = bash ] && [ "$2" = -lc ] && { shift 2; exec bash -c "$@"; }
exec "$@"
""",
}


class Box:
    """A directory tree standing in for the GPU box, and a way to run a step on it."""

    def __init__(self, root: Path):
        self.root = root
        self.fake = root / "fake"
        self.bin = root / "bin"
        self.claude = root / "home-claude"
        for d in (self.fake, self.bin, self.claude):
            d.mkdir(parents=True)
        for name, body in FAKES.items():
            p = self.bin / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        py = self.bin / "python3"                      # a venv python must be run by its own path
        py.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        py.chmod(py.stat().st_mode | stat.S_IXUSR)
        self.target = root / "target.yaml"
        self.pulled()

    def write(self, t: dict) -> None:
        self.target.write_text(yaml.safe_dump(t))

    def pulled(self, **edit) -> dict:
        """One GPU and a prebuilt engine image pinned by digest: the plain gpu target."""
        t = yaml.safe_load(GPU_BUILT_TARGET)
        t["id"] = "gpu-pulled"
        t["engine"].update(image=IMAGE, sources=["vllm"], docker_flags=["--entrypoint="],
                           env={"HF_HUB_OFFLINE": "1"}, shm_size="8g")
        t["provision"] = {"sources": {"vllm": {"ref": SHA}}}
        t["model"]["local_dir"] = str(self.root / "models" / "some-model")
        for dotted, v in edit.items():
            node = t
            *path, last = dotted.split("__")
            for k in path:
                node = node.setdefault(k, {})
            if v is None:
                node.pop(last, None)
            else:
                node[last] = v
        self.write(t)
        return t

    def built(self, **edit) -> dict:
        t = self.pulled(engine__image="inference-judge-base:latest", engine__docker_flags=[], engine__env={},
                        provision=None, **edit)
        return t

    def run(self, script: str, **env) -> subprocess.CompletedProcess:
        e = {"PATH": f"{self.bin}:/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(self.claude), "LC_ALL": "C",
             "FAKE": str(self.fake), "IJ_BOOTSTRAP_LIB": "1", "IJ_TARGET_PYTHON": sys.executable,
             "IJ_GPU_DRIVER_FILE": str(self.root / "driver-version"),
             "IJ_GPU_CHANNEL_FILE": str(self.root / "channel"),
             "IJ_PINS_FILE": str(self.root / "pins"), "GIT_CONFIG_NOSYSTEM": "1", **env}
        body = (f'. "{REPO}/setup/bootstrap.sh"; RESEARCHER_HOME="{self.claude}"; target_load "{self.target}"; '
                f'( set -e; {script} )')
        return subprocess.run(["/bin/bash", "-c", body], env=e, cwd=self.root, capture_output=True,
                              text=True, timeout=120)

    def log(self, name: str) -> list[str]:
        p = self.fake / f"{name}.log"
        return p.read_text().splitlines() if p.exists() else []


@pytest.fixture
def box(tmp_path: Path) -> Box:
    return Box(tmp_path)


def said(r: subprocess.CompletedProcess) -> str:
    return r.stdout + r.stderr


# ------------------------------------------------------------ the GPUs on the host
def test_one_gpu_for_a_one_gpu_target_passes_and_records_the_driver(box):
    r = box.run("gpu_devices")
    assert r.returncode == 0, said(r)
    assert (box.root / "driver-version").read_text().strip() == "580.126.09"


def test_more_gpus_than_the_target_names_is_refused(box):
    r = box.run("gpu_devices", FAKE_GPUS="2")
    assert r.returncode == 1
    assert "shows 2 GPU(s)" in r.stderr and "wants 1" in r.stderr and "accelerator.chips" in r.stderr


def test_fewer_gpus_than_the_target_names_is_refused(box):
    box.pulled(accelerator__chips=4)
    r = box.run("gpu_devices", FAKE_GPUS="2")
    assert r.returncode == 1 and "shows 2 GPU(s)" in r.stderr and "wants 4" in r.stderr


def test_a_host_without_a_gpu_is_refused(box):
    r = box.run("gpu_devices", FAKE_GPUS="0")
    assert r.returncode == 1 and "shows 0 GPU(s)" in r.stderr


@pytest.mark.parametrize("name", ["NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5090", "NVIDIA L40S"])
def test_a_target_that_names_no_type_takes_any_card(box, name):
    r = box.run("gpu_devices", FAKE_GPU_NAME=name)
    assert r.returncode == 0, said(r)
    assert "H100" not in said(r) and "expected" not in said(r)


def test_a_named_type_must_be_the_card_on_the_host(box):
    box.pulled(accelerator__type="L40S")
    r = box.run("gpu_devices", FAKE_GPU_NAME="NVIDIA A100-SXM4-80GB")
    assert r.returncode == 1
    assert "accelerator.type" in r.stderr and "L40S" in r.stderr and "A100" in r.stderr


def test_a_named_type_matches_whatever_the_case(box):
    box.pulled(accelerator__type="l40s")
    assert box.run("gpu_devices", FAKE_GPU_NAME="NVIDIA L40S").returncode == 0


def test_a_card_smaller_than_the_target_says_is_refused(box):
    box.pulled(accelerator__hbm_gb_per_chip=80)
    assert box.run("gpu_devices").returncode == 0                       # 81559 MiB is an 80 GB card
    r = box.run("gpu_devices", FAKE_GPU_MIB="24564")
    assert r.returncode == 1 and "hbm_gb_per_chip" in r.stderr and "24564" in r.stderr


def test_a_prebuilt_image_asks_for_no_driver_series(box):
    r = box.run("gpu_devices", FAKE_DRIVER="550.90.07")
    assert r.returncode == 0, said(r)


def test_an_engine_built_here_states_its_minimum_driver_and_why(box):
    box.built()
    r = box.run("gpu_devices", FAKE_DRIVER="570.124.06")
    assert r.returncode == 1
    assert "r580" in r.stderr and "CUDA 13" in r.stderr
    assert "ubuntu-24.04-cuda" not in r.stderr          # the requirement, not one cloud's image name
    assert box.run("gpu_devices", FAKE_DRIVER="580.65.06").returncode == 0


def test_the_target_can_name_its_own_minimum_driver(box):
    box.pulled(provision__min_driver_major=590)
    r = box.run("gpu_devices", FAKE_DRIVER="580.126.09")
    assert r.returncode == 1 and "r590" in r.stderr and "provision.min_driver_major" in r.stderr


# ---------------------------------------------------------- docker reaches the GPU
def test_a_prebuilt_image_is_not_probed_with_another_image(box):
    r = box.run("gpu_passthrough")
    assert r.returncode == 0 and "gpu gate" in said(r)
    assert not [l for l in box.log("docker") if l.startswith("docker run")]


def test_an_engine_built_here_is_probed_with_the_cuda_base_it_builds_on(box):
    box.built()
    r = box.run('gpu_passthrough; echo "probe=$GPU_PROBE_IMAGE"')
    assert r.returncode == 0, said(r)
    runs = [l for l in box.log("docker") if l.startswith("docker run")]
    assert len(runs) == 1 and "--gpus all" in runs[0] and "nvidia-smi -L" in runs[0]
    # the probe is the same CUDA release Dockerfile.base builds on
    release = [l for l in (REPO / "setup" / "Dockerfile.base").read_text().splitlines()
               if l.startswith("ARG CUDA_IMAGE=")][0].split(":")[1].split("-")[0]
    assert f"probe=nvidia/cuda:{release}-base-" in r.stdout


def test_the_gate_launches_the_engine_image_as_the_ruler_will(box):
    r = box.run("gpu_gate")
    assert r.returncode == 0, said(r)
    run = [l for l in box.log("docker") if l.startswith("docker run")][0]
    for flag in ("--network none", "--gpus all", "--shm-size 8g", "--entrypoint=", "-e HF_HUB_OFFLINE=1", IMAGE):
        assert flag in run, flag
    assert "nvidia/cuda" not in run


def test_the_gate_fails_when_the_image_sees_another_number_of_gpus(box):
    r = box.run("gpu_gate", FAKE_GATE_GPUS="0")
    assert r.returncode == 1 and "0 GPU(s)" in r.stderr and "wants 1" in r.stderr


def test_the_gate_says_what_a_container_that_cannot_start_usually_means(box):
    r = box.run("gpu_gate", FAKE_GATE_RC="125")
    assert r.returncode == 1
    assert "could not select device driver" in r.stderr and "NVIDIA container toolkit" in r.stderr
    old = box.run("gpu_gate", FAKE_GATE_RC="125", FAKE_GATE_ERR="nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0")
    assert old.returncode == 1 and "driver" in old.stderr and "too old for this image" in old.stderr


def test_a_sketch_is_told_what_to_do_after_the_gate(box):
    box.pulled(runnable=False)
    r = box.run("gpu_not_runnable")
    assert r.returncode == 0
    for text in ("runnable: false", "runnable: true", "baseline-vllm", "bootstrap.sh --target"):
        assert text in r.stdout, text


# ------------------------------------------------------------- engine source trees
def test_a_prebuilt_image_gets_only_the_trees_the_target_names_at_its_pinned_commit(box):
    r = box.run("s_engine_sources")
    assert r.returncode == 0, said(r)
    git = " ".join(box.log("git"))
    assert SHA in git and "vllm-project/vllm" in git
    assert "sglang" not in git
    assert (box.claude / "vllm-src" / ".git").is_dir() and not (box.claude / "sglang-src").exists()


def test_a_tree_with_no_pinned_ref_is_refused_before_anything_is_fetched(box):
    box.pulled(engine__sources=["vllm", "sglang"])        # provision.sources names only vllm
    r = box.run("s_engine_sources")
    assert r.returncode == 1 and "provision.sources.sglang.ref" in r.stderr
    assert box.log("git") == []


@pytest.mark.parametrize("ref", ["main", "latest", "nightly", "abc123", "v0.31.0; reboot"])
def test_a_floating_ref_is_refused(box, ref):
    box.pulled(provision__sources={"vllm": {"ref": ref}})
    r = box.run("s_engine_sources")
    assert r.returncode == 1 and "40-hex commit or a release tag" in r.stderr
    assert box.log("git") == []


def test_a_tree_from_somewhere_else_needs_its_url_in_the_target(box):
    box.pulled(engine__sources=["my_engine"], provision__sources={"my_engine": {"ref": SHA}})
    r = box.run("s_engine_sources")
    assert r.returncode == 1 and "provision.sources.my_engine.url" in r.stderr
    box.pulled(engine__sources=["my_engine"],
               provision__sources={"my_engine": {"ref": SHA, "url": "https://example.org/my_engine.git"}})
    r = box.run("s_engine_sources")
    assert r.returncode == 0, said(r)
    assert "https://example.org/my_engine.git" in " ".join(box.log("git"))


def test_a_failed_fetch_leaves_no_half_made_tree_behind(box):
    (box.fake / "fetch_fails").write_text("")
    r = box.run("s_engine_sources")
    assert r.returncode != 0
    assert not (box.claude / "vllm-src").exists()         # the next run fetches again


def test_an_existing_tree_is_left_alone(box):
    (box.claude / "vllm-src" / ".git").mkdir(parents=True)
    r = box.run("s_engine_sources")
    assert r.returncode == 0 and "leaving it alone" in said(r)
    assert not [l for l in box.log("git") if l.startswith(("git clone", "git fetch", "git checkout"))]


def test_a_tree_at_another_commit_than_the_image_was_built_from_is_reported(box):
    (box.fake / "image_present").write_text("")
    (box.fake / "image_revision").write_text("f" * 40 + "\n")
    r = box.run("s_engine_sources")
    assert r.returncode == 0 and "the image says it was built from" in said(r)


def test_an_engine_built_here_clones_only_the_named_tree_at_its_release_tag(box):
    box.built()                                           # engine vllm, sources [vllm]
    r = box.run("VLLM_TAG=v0.29.0 SGLANG_TAG=; s_engine_sources")
    assert r.returncode == 0, said(r)
    git = " ".join(box.log("git"))
    assert "--branch v0.29.0" in git and "vllm-project/vllm" in git and "sglang" not in git


def test_an_engine_built_here_with_both_trees_clones_both(box):
    box.built(engine__sources=["vllm", "sglang"])
    r = box.run("VLLM_TAG=v0.29.0 SGLANG_TAG=v0.5.19; s_engine_sources")
    assert r.returncode == 0, said(r)
    git = " ".join(box.log("git"))
    assert "--branch v0.29.0" in git and "--branch v0.5.19" in git and "sgl-project/sglang" in git


# -------------------------------------------------------- pins for a built engine
def test_only_the_engines_the_target_names_need_a_release_tag(box):
    box.built()                                           # vllm only
    r = box.run("require_engine_pins", IJ_VLLM_TAG="v0.29.0")
    assert r.returncode == 0, said(r)
    box.built(engine__sources=["vllm", "sglang"])
    r = box.run("require_engine_pins", IJ_VLLM_TAG="v0.29.0")
    assert r.returncode == 2 and "IJ_SGLANG_TAG is required" in r.stderr


def test_an_engine_built_here_must_list_its_own_tree(box):
    box.built(engine__sources=None)                       # no engine.sources at all
    r = box.run("require_engine_pins", IJ_VLLM_TAG="v0.29.0")
    assert r.returncode == 2 and "engine.sources: [vllm]" in r.stderr


# ------------------------------------------------------- a target with no tree
def test_a_target_that_lists_no_engine_tree_fetches_none(box):
    """No engine is assumed: without engine.sources nothing is cloned, the plan
    says so, and the file the edit hook reads is empty."""
    box.pulled(engine__sources=None, provision=None)
    r = box.run("plan_steps")
    assert r.returncode == 0, said(r)
    assert "skip|engine source trees|the target lists none" in r.stdout
    assert "s_engine_sources" not in r.stdout
    r = box.run("s_engine_sources")
    assert r.returncode == 0, said(r)
    assert not (box.fake / "git.log").exists() and not list(box.claude.glob("*-src"))


def test_the_channel_records_only_the_engines_it_builds(box):
    box.built()
    (box.root / "driver-version").write_text("580.126.09\n")
    r = box.run("gpu_channel", IJ_VLLM_TAG="v0.29.0", IJ_SGLANG_TAG="v0.5.19")
    assert r.returncode == 0, said(r)
    pins = (box.root / "pins").read_text()
    assert "VLLM_TAG=v0.29.0" in pins and "SGLANG_TAG=\n" in pins and "TORCH_BACKEND=cu130" in pins


def test_a_tree_the_base_image_cannot_build_is_refused(box):
    box.built(engine__sources=["vllm", "my_engine"])
    r = box.run("require_engine_pins", IJ_VLLM_TAG="v0.29.0")
    assert r.returncode == 2 and "my_engine" in r.stderr


# --------------------------------------------------------------- the disk gate
@pytest.mark.parametrize("edit,want", [
    ({}, "100"),                                          # a gpu target that says nothing
    ({"provision__min_free_gb": 120}, "120"),
    ({"provision__min_free_gb": 30}, "30"),
    ({"accelerator__kind": "none"}, "5"),
])
def test_free_disk_needed_comes_from_the_target(box, edit, want):
    box.pulled(**edit)
    r = box.run("disk_need_gb")
    assert r.returncode == 0 and r.stdout.strip() == want, said(r)


def test_a_rerun_with_the_weights_and_the_image_only_needs_headroom(box):
    t = box.pulled(provision__min_free_gb=120)
    Path(t["model"]["local_dir"]).mkdir(parents=True)
    assert box.run("disk_need_gb").stdout.strip() == "120"          # weights, but no engine image yet
    (box.fake / "image_present").write_text("")
    assert box.run("disk_need_gb").stdout.strip() == "40"
    assert any(IMAGE in l for l in box.log("docker"))               # this target's image, not another


def test_the_tpu_and_cpu_targets_keep_their_disk_gate():
    for name, want in (("cpu-mock", "5"), ("v6e4-gemma4-26b-tps", "40"), ("sketch-tpu-8chip", "40")):
        r = lib(f'target_load "{REPO}/targets/{name}.yaml"; disk_need_gb')
        assert r.stdout.strip() == want, name + said(r)


# ------------------------------------------------------- the baseline experiment
def lib(script: str) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": "/nonexistent", "IJ_BOOTSTRAP_LIB": "1",
           "LC_ALL": "C", "IJ_TARGET_PYTHON": sys.executable}
    return subprocess.run(["/bin/bash", "-c", f'. "{REPO}/setup/bootstrap.sh"; {script}'],
                          env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("target,baseline", [
    ("cpu-mock", "baseline-mock"), ("v6e4-gemma4-26b-tps", "baseline-tpu"),
    ("sketch-tpu-8chip", "baseline-tpu"), ("gpu-gemma4-26b-tps", "baseline-vllm"),
])
def test_every_shipped_target_resolves_to_a_baseline_that_exists(target, baseline):
    r = lib(f'target_load "{REPO}/targets/{target}.yaml"; echo "$T_BASELINE"')
    assert r.stdout.strip() == baseline, said(r)
    assert (REPO / "research" / "experiments" / baseline / "launch.sh").is_file()


def test_no_shipped_target_is_left_without_a_baseline():
    for f in sorted((REPO / "targets").glob("*.yaml")):
        r = lib(f'target_load "{f}"; echo "$T_BASELINE"')
        name = r.stdout.strip()
        assert name.startswith("baseline-"), f.name + said(r)
        assert (REPO / "research" / "experiments" / name / "launch.sh").is_file(), f"{f.name} -> {name}"


def test_a_target_can_name_its_own_baseline(box):
    box.pulled(provision__baseline="baseline-sglang")
    assert box.run('echo "$T_BASELINE"').stdout.strip() == "baseline-sglang"
    box.pulled(engine__name="sglang")
    assert box.run('echo "$T_BASELINE"').stdout.strip() == "baseline-sglang"


def test_check_verifies_the_baseline_launcher_of_every_target():
    check = (REPO / "setup" / "check.sh").read_text()
    assert 'files+=("experiments/$T_BASELINE/launch.sh")' in check
    assert '[ -n "$T_BASELINE" ] && files+=' not in check              # no target is exempt


# ------------------------------------------------- nothing names one machine
@pytest.mark.parametrize("path", ["setup/bootstrap.sh", "setup/gpu.sh", "setup/check.sh", "setup/target.sh",
                                  "setup/Dockerfile.base", "setup/README.md",
                                  "research/experiments/baseline-vllm/launch.sh"])
def test_the_gpu_path_names_no_cloud_card_or_model(path):
    text = (REPO / path).read_text()
    for word in ("ubuntu-24.04-cuda", "H100", "gpt-oss", "120b"):
        assert word not in text, f"{path} still names {word}"


# ----------------------------------------------------------- the plain launcher
def test_the_plain_vllm_launcher_serves_from_an_image_that_has_only_python3(tmp_path):
    """The upstream image has python3 and no `python`; the base image built here has both."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in (("python3", 'echo "python3 $*" >> "$LOG"'), ("vllm", 'echo "vllm $*" >> "$LOG"'),
                       ("date", "echo now")):
        p = bin_ / name
        p.write_text(f"#!/bin/bash\n{body}\n")
        p.chmod(0o755)
    log = tmp_path / "log"
    r = subprocess.run(["/bin/bash", str(REPO / "research/experiments/baseline-vllm/launch.sh")],
                       env={"PATH": str(bin_), "LOG": str(log), "PORT": "8000", "IJ_SERVED_NAME": "m",
                            "IJ_MODEL_DIR": "/opt/models/m"},
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    calls = log.read_text().splitlines()
    assert calls[0].startswith("python3 -c") and calls[-1] == "vllm serve /opt/models/m --served-model-name m --host 0.0.0.0 --port 8000"

"""The TPU VM steps of bootstrap (setup/tpu.sh), one behaviour per test.

The interface under test is each step's function as bootstrap runs it: in a
`( set -e; fn )` subshell, with the target loaded from a file.  Everything the
steps call on the machine (docker, mount, findfs, chown, sudo, install) is a
small fake in a bin directory; the fakes only record their argv and play a
scripted outcome, so what is asserted is what the step decided and told the
operator, never what a real TPU would do (that stays unproven; see the notes in
the final report).  Files are real: the Hugging Face cache layout, hardlinks,
git repos, rsync.  /bin/bash is the macOS 3.2 on a laptop, so this also proves
the library has no bash 4 in it.
"""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import REPO

pytestmark = pytest.mark.skipif(os.getuid() == 0, reason="the fakes stand in for root-only commands")

DIGEST = "sha256:" + "a" * 64
IMAGE = "vllm/vllm-tpu@" + DIGEST
REV = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"

FAKES = {
    "docker": r"""#!/bin/bash
echo "docker $*" >> "$FAKE/docker.log"
case "$1" in
  image) [ -f "$FAKE/loaded" ] || exit 1; echo "[\"x@$(cat "$FAKE/loaded")\"] sha256:abc"; exit 0 ;;
  load) [ -f "$FAKE/load_gives_digest" ] && cp "$FAKE/digest" "$FAKE/loaded"; exit 0 ;;
  pull) [ -f "$FAKE/pull_ok" ] || { echo "pull failed" >&2; exit 1; }; cp "$FAKE/digest" "$FAKE/loaded"; exit 0 ;;
  rm) exit 0 ;;
  run) case "$*" in *"--network host"*) rc=${FAKE_GATE2_RC:-0} ;; *) rc=${FAKE_GATE1_RC:-0} ;; esac
       [ "$rc" = 0 ] || echo "RuntimeError: no TPU chips" >&2
       exit "$rc" ;;
  create) echo cid1; exit 0 ;;
  cp) src=${2#*:}; name=$(basename "$(dirname "$src")")
      grep -qx "$name" "$FAKE/image_trees" 2>/dev/null || { echo "no such path $src" >&2; exit 1; }
      mkdir -p "$3/pkg"; echo "x = 1" > "$3/pkg/mod.py"
      if [ -f "$FAKE/image_dirty_git" ]; then   # a shipped checkout the image build patched
        git -C "$3" init -q && git -C "$3" add -A && git -C "$3" -c user.name=i -c user.email=i@i commit -qm shipped
        echo "x = 99  # patched by the image build" > "$3/pkg/mod.py"
      fi
      exit 0 ;;
esac
exit 0
""",
    "findfs": r"""#!/bin/bash
[ "$1" = "LABEL=$(cat "$FAKE/label" 2>/dev/null)" ] && { echo /dev/sdb; exit 0; }
exit 1
""",
    "mountpoint": r"""#!/bin/bash
grep -qx "${!#}" "$FAKE/mounts" 2>/dev/null
""",
    # `mount --bind A B` makes B show A's contents, as the real one does (B becomes a link to A).
    "mount": r"""#!/bin/bash
echo "mount $*" >> "$FAKE/mount.log"; echo "${!#}" >> "$FAKE/mounts"
if [ "$1" = --bind ]; then rmdir "$3" && /bin/ln -s "$2" "$3" && echo "$3" >> "$FAKE/binds"; fi
""",
    # link(2) through a bind mount is EXDEV even when both ends are one filesystem (checked
    # on Linux 6.x with a real bind mount).  The fake refuses what the kernel refuses.
    "ln": r"""#!/bin/bash
echo "ln $*" >> "$FAKE/ln.log"
dst="${!#}"
while IFS= read -r bound; do
  case "$dst" in "$bound"/*) echo "ln: failed to create hard link '$dst': Invalid cross-device link" >&2; exit 1 ;; esac
done < <(cat "$FAKE/binds" 2>/dev/null)
exec /bin/ln "$@"
""",
    "chown": r"""#!/bin/bash
echo "chown $*" >> "$FAKE/chown.log"
""",
    "install": r"""#!/bin/bash
echo "install $*" >> "$FAKE/install.log"
case "$*" in *" -d "*|"-d "*) mkdir -p "${!#}" ;; esac
""",
    # sudo -u USER -H bash -lc CMD  ->  bash -c CMD (no login shell: path_helper would reorder PATH)
    "sudo": r"""#!/bin/bash
while [ $# -gt 0 ]; do case "$1" in -u) shift 2 ;; -H|-n) shift ;; *) break ;; esac; done
[ "$1" = bash ] && [ "$2" = -lc ] && { shift 2; exec bash -c "$@"; }
exec "$@"
""",
}


class Vm:
    """A directory tree standing in for the TPU VM, and a way to run a step on it."""

    def __init__(self, root: Path):
        self.root = root
        self.fake = root / "fake"
        self.bin = root / "bin"
        self.mnt = root / "disk"                       # where the cache disk is "mounted"
        self.models_root = root / "opt" / "models"
        self.claude = root / "home-claude"
        self.fstab = root / "fstab"
        self.dev = root / "dev"
        for d in (self.fake, self.bin, self.mnt, self.claude, self.dev):
            d.mkdir(parents=True)
        self.fstab.write_text("")
        for name, body in FAKES.items():
            p = self.bin / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        py = self.bin / "python3"                      # a venv python must be run by its own path
        py.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
        py.chmod(py.stat().st_mode | stat.S_IXUSR)
        (self.fake / "digest").write_text(DIGEST)
        (self.fake / "label").write_text("gemma4-cache")
        (self.fake / "image_trees").write_text("tpu_inference\nvllm\n")
        self.target = root / "target.yaml"
        self.write_target()

    def write_target(self, runnable: bool = True, **engine_extra) -> None:
        t = yaml.safe_load((REPO / "targets" / "v6e4-gemma4-26b-tps.yaml").read_text())
        t["runnable"] = runnable                       # the file in the repo is a sketch until its first VM
        t["engine"]["image"] = IMAGE
        t["engine"]["docker_flags"] = ["--privileged"]
        t["engine"]["sources"] = ["tpu_inference", "vllm"]
        t["engine"]["env"] = {"MODEL_IMPL_TYPE": "flax_nnx", "USE_BATCHED_RPA_KERNEL": "1"}
        t["engine"].update(engine_extra)
        t["engine"]["compile_cache"] = {"seed": str(self.root / "bench-cache"), "mount": "/root/.cache/vllm"}
        t["model"]["local_dir"] = str(self.models_root / "gemma-4-26b-a4b-it")
        t["probes"] = {"present": "echo 4", "idle": "true"}
        t["provision"]["cache_disk"] = {
            "label": "gemma4-cache", "mount": str(self.mnt), "image_tar": "runtime/vllm-tpu-image.tar",
            "hf_cache": "artifacts/gemma4-hf", "compile_cache": "artifacts/gemma4-vllm"}
        self.target.write_text(yaml.safe_dump(t))

    def run(self, script: str, **env) -> subprocess.CompletedProcess:
        e = {"PATH": f"{self.bin}:/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(self.claude), "LC_ALL": "C",
             "FAKE": str(self.fake), "IJ_BOOTSTRAP_LIB": "1", "IJ_FSTAB": str(self.fstab),
             "IJ_DEV_DIR": str(self.dev), "IJ_TARGET_PYTHON": sys.executable,
             "IJ_DEADMAN_DIR": str(self.root / "deadman"), "GIT_CONFIG_NOSYSTEM": "1",
             "IJ_TPU_GATE_VERDICT": str(self.root / "gate-verdict"), **env}
        body = (f'. "{REPO}/setup/bootstrap.sh"; RESEARCHER_HOME="{self.claude}"; target_load "{self.target}"; '
                f'( set -e; {script} )')
        return subprocess.run(["/bin/bash", "-c", body], env=e, cwd=self.root, capture_output=True,
                              text=True, timeout=120)

    def log(self, name: str) -> list[str]:
        p = self.fake / f"{name}.log"
        return p.read_text().splitlines() if p.exists() else []

    # --- the disk -----------------------------------------------------------
    def put_image_tar(self) -> Path:
        p = self.mnt / "runtime" / "vllm-tpu-image.tar"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"not a real tar")
        return p

    def put_hf_cache(self, files: dict[str, bytes]) -> Path:
        """The Hugging Face hub layout: blobs named by hash, snapshot entries are symlinks to them."""
        hub = self.mnt / "artifacts" / "gemma4-hf" / "models--google--gemma-4-26B-A4B-it"
        snap = hub / "snapshots" / REV
        (hub / "blobs").mkdir(parents=True)
        snap.mkdir(parents=True)
        for rel, data in files.items():
            name = hashlib.sha256(data).hexdigest() if len(data) > 100 else hashlib.sha1(
                b"blob %d\0" % len(data) + data).hexdigest()
            (hub / "blobs" / name).write_bytes(data)
            (snap / rel).parent.mkdir(parents=True, exist_ok=True)
            (snap / rel).symlink_to(Path("..") / ".." / "blobs" / name)
        return hub


@pytest.fixture
def vm(tmp_path: Path) -> Vm:
    return Vm(tmp_path)


def said(r: subprocess.CompletedProcess) -> str:
    return r.stdout + r.stderr


# ------------------------------------------------------------------ the chips
def test_the_probe_count_must_match_the_chips(vm):
    assert vm.run("tpu_devices").returncode == 0


def test_a_probe_that_counts_the_wrong_path_is_reported_but_the_real_nodes_decide(vm):
    vm.write_target()
    t = yaml.safe_load(vm.target.read_text())
    t["probes"]["present"] = "ls /dev/accel* 2>/dev/null | wc -l"
    vm.target.write_text(yaml.safe_dump(t))
    (vm.dev / "vfio").mkdir()
    for n in "0123":
        (vm.dev / "vfio" / n).write_text("")
    (vm.dev / "vfio" / "vfio").write_text("")            # the control node is not a chip
    r = vm.run("tpu_devices")
    assert r.returncode == 0 and "fix the probe" in said(r)


def test_too_few_chips_says_the_node_is_wrong(vm):
    t = yaml.safe_load(vm.target.read_text())
    t["probes"]["present"] = "echo 2"
    vm.target.write_text(yaml.safe_dump(t))
    r = vm.run("tpu_devices")
    assert r.returncode == 1 and "wants 4" in r.stderr and "recreate" in r.stderr


# ------------------------------------------------------------------ the disk
def test_the_disk_is_found_by_label_mounted_and_remembered_for_reboots(vm):
    r = vm.run("tpu_mount_disk")
    assert r.returncode == 0, said(r)
    assert vm.log("mount") == [f"mount -o noatime /dev/sdb {vm.mnt}"]
    assert vm.fstab.read_text() == f"LABEL=gemma4-cache {vm.mnt} ext4 defaults,nofail,noatime 0 2\n"


def test_mounting_twice_changes_nothing(vm):
    vm.run("tpu_mount_disk")
    r = vm.run("tpu_mount_disk")
    assert r.returncode == 0 and "already mounted" in r.stdout
    assert len(vm.log("mount")) == 1 and len(vm.fstab.read_text().splitlines()) == 1


def test_the_spend_guard_is_told_where_the_disk_is_so_the_results_can_outlive_the_vm(vm):
    r = vm.run("tpu_mount_disk")
    assert r.returncode == 0, said(r)
    assert (vm.root / "deadman" / "save-to").read_text() == f"{vm.mnt}\n"     # tools/tpu_deadman.sh reads this
    assert "tpu_deadman.sh arm" in r.stderr                                    # and nobody has armed it yet
    (vm.root / "deadman" / "deadline").write_text("1800000000\n")
    r = vm.run("tpu_mount_disk")                                               # a re-run, on an armed VM
    assert r.returncode == 0 and "tpu_deadman.sh arm" not in said(r)
    assert (vm.root / "deadman" / "save-to").read_text() == f"{vm.mnt}\n"


def test_a_missing_disk_tells_the_operator_how_to_attach_it(vm):
    (vm.fake / "label").write_text("some-other-disk")
    r = vm.run("tpu_mount_disk")
    assert r.returncode == 1 and "gemma4-cache" in r.stderr and "attach-disk" in r.stderr
    assert vm.fstab.read_text() == ""


# ------------------------------------------------------------------ the image
def test_an_image_already_at_its_digest_is_left_alone(vm):
    (vm.fake / "loaded").write_text(DIGEST)
    r = vm.run("tpu_engine_image")
    assert r.returncode == 0
    assert not [l for l in vm.log("docker") if l.startswith(("docker load", "docker pull"))]


def test_the_image_is_loaded_from_the_disk_tar_and_verified_by_digest(vm):
    tar = vm.put_image_tar()
    (vm.fake / "load_gives_digest").write_text("")
    r = vm.run("tpu_engine_image")
    assert r.returncode == 0, said(r)
    assert f"docker load --input {vm.mnt}/runtime/vllm-tpu-image.tar" in vm.log("docker")
    assert not [l for l in vm.log("docker") if l.startswith("docker pull")]
    assert tar.exists()


def test_a_load_that_loses_the_digest_falls_back_to_a_pull(vm):
    vm.put_image_tar()                                    # docker load leaves no RepoDigest behind
    (vm.fake / "pull_ok").write_text("")
    r = vm.run("tpu_engine_image")
    assert r.returncode == 0 and "pulling it" in r.stderr
    assert f"docker pull {IMAGE}" in vm.log("docker")


def test_no_tar_and_no_pull_fails_and_says_which_digest(vm):
    r = vm.run("tpu_engine_image")
    assert r.returncode == 1 and DIGEST in r.stderr


def test_an_image_pinned_by_tag_is_refused(vm):
    t = yaml.safe_load(vm.target.read_text())
    t["engine"]["image"] = "vllm/vllm-tpu:latest"
    vm.target.write_text(yaml.safe_dump(t))
    r = vm.run("tpu_engine_image")
    assert r.returncode == 1 and "not pinned by digest" in r.stderr
    assert not vm.log("docker")


# ---------------------------------------------------------------- chip gate
def gate_run_line(vm) -> str:
    return next(l for l in vm.log("docker") if l.startswith("docker run"))


def test_the_gate_launches_the_way_the_ruler_does(vm):
    r = vm.run("tpu_gate")
    assert r.returncode == 0, said(r)
    line = gate_run_line(vm)
    for want in ("--network none", "--shm-size 16g", "--privileged", "-e MODEL_IMPL_TYPE=flax_nnx",
                 f"{IMAGE} python -c", "jax.devices()", "== 4"):
        assert want in line, want
    assert "--entrypoint" not in line                  # through the image's entrypoint, as a real launch goes
    assert len([l for l in vm.log("docker") if l.startswith("docker run")]) == 1


def test_a_target_that_names_a_network_gets_no_seal(vm):
    vm.write_target(docker_flags=["--privileged", "--network", "host"])
    assert vm.run("tpu_gate").returncode == 0
    assert "--network none" not in gate_run_line(vm)


def test_flags_that_break_a_healthy_node_say_do_not_recreate_it(vm):
    r = vm.run("tpu_gate", FAKE_GATE1_RC="1", FAKE_GATE2_RC="0")
    assert r.returncode == 1
    assert "node is healthy" in r.stderr and "Do NOT recreate" in r.stderr and "engine.docker_flags" in r.stderr
    assert len([l for l in vm.log("docker") if l.startswith("docker run")]) == 2
    assert "--privileged --network host" in [l for l in vm.log("docker") if l.startswith("docker run")][1]
    assert f"tools/tpu_seal_probe.sh '{IMAGE}' 4" in r.stderr     # the one place the working flags come from


def test_on_a_sketch_target_unproven_flags_are_a_finding_not_a_failure(vm):
    vm.write_target(runnable=False)
    r = vm.run("tpu_gate", FAKE_GATE1_RC="1", FAKE_GATE2_RC="0")
    assert r.returncode == 0, said(r)
    assert "node is healthy" in said(r) and "Do NOT recreate" in said(r)
    assert (vm.root / "gate-verdict").read_text().strip() == "node"
    assert vm.run("tpu_gate").returncode == 0
    assert (vm.root / "gate-verdict").read_text().strip() == "flags"


def test_a_broken_node_fails_the_gate_on_a_sketch_target_too(vm):
    vm.write_target(runnable=False)
    r = vm.run("tpu_gate", FAKE_GATE1_RC="1", FAKE_GATE2_RC="1")
    assert r.returncode == 1 and "DELETE AND RECREATE THE NODE" in r.stderr
    assert not (vm.root / "gate-verdict").exists()


def test_a_node_that_fails_every_way_says_delete_and_recreate_it(vm):
    r = vm.run("tpu_gate", FAKE_GATE1_RC="1", FAKE_GATE2_RC="1")
    assert r.returncode == 1
    assert "DELETE AND RECREATE THE NODE" in r.stderr and "no TPU chips" in r.stderr


def test_a_gate_that_times_out_says_so(vm):
    r = vm.run("tpu_gate", FAKE_GATE1_RC="124", FAKE_GATE2_RC="124", IJ_TPU_GATE_TIMEOUT_S="7")
    assert r.returncode == 1 and "timed out after 7s" in r.stderr


@pytest.mark.skipif(not any(os.access(f"{d}/{n}", os.X_OK) for d in os.environ["PATH"].split(":")
                            for n in ("timeout", "gtimeout")), reason="needs GNU timeout (not on macOS)")
def test_a_gate_that_hangs_is_killed(vm):
    (vm.bin / "docker").write_text('#!/bin/bash\n[ "$1" = run ] && exec sleep 30\nexit 0\n')
    t0 = __import__("time").time()
    r = vm.run("tpu_gate", IJ_TPU_GATE_TIMEOUT_S="1")
    assert r.returncode == 1 and __import__("time").time() - t0 < 25


# ------------------------------------------------------------------ the model
FILES = {"config.json": b'{"a": 1}\n', "tokenizer.json": b'{"t": 2}\n',
         "model-00001-of-00002.safetensors": os.urandom(4096),
         "model-00002-of-00002.safetensors": os.urandom(4096)}


def model_dir(vm) -> Path:
    return vm.models_root / "gemma-4-26b-a4b-it"


def test_the_model_dir_is_flat_real_files_hardlinked_from_the_cache(vm):
    hub = vm.put_hf_cache(FILES)
    r = vm.run("tpu_models")
    assert r.returncode == 0, said(r)
    d = model_dir(vm)
    assert sorted(p.name for p in d.iterdir()) == sorted(FILES)
    assert not any(p.is_symlink() for p in d.rglob("*"))
    for name, data in FILES.items():
        assert (d / name).read_bytes() == data
    blob = next((hub / "blobs").iterdir())
    assert any((d / n).stat().st_ino == blob.stat().st_ino for n in FILES)       # linked, not copied
    assert f"mount --bind {vm.mnt}/models {vm.models_root}" in vm.log("mount")
    assert f"{vm.mnt}/models {vm.models_root} none bind,nofail 0 0" in vm.fstab.read_text()


def test_the_links_are_made_on_the_disk_never_through_the_bind_mount(vm):
    vm.put_hf_cache(FILES)
    r = vm.run("tpu_models")
    assert r.returncode == 0, said(r)
    made = [l.split()[-1] for l in vm.log("ln")]
    assert len(made) == len(FILES)
    for dst in made:                                      # the disk's own path: one mount, so link(2) works
        assert dst.startswith(f"{vm.mnt}/models/gemma-4-26b-a4b-it/"), dst
    for name in FILES:                                    # and the ruler's path shows the same files
        assert (model_dir(vm) / name).samefile(vm.mnt / "models" / "gemma-4-26b-a4b-it" / name)


def test_the_manifest_comes_from_the_blob_names_and_is_verified(vm):
    vm.put_hf_cache(FILES)
    r = vm.run("tpu_models")
    assert r.returncode == 0, said(r)
    manifest = (model_dir(vm).parent / "gemma-4-26b-a4b-it.sha256").read_text().splitlines()
    assert len(manifest) == len(FILES)
    assert f"sha256 {hashlib.sha256(FILES['model-00001-of-00002.safetensors']).hexdigest()} " \
           "model-00001-of-00002.safetensors" in manifest
    assert any(l.startswith("gitsha1 ") and l.endswith(" config.json") for l in manifest)
    assert "OK 4/4 files verified" in r.stdout


def test_a_second_run_does_not_rehash_or_remount(vm):
    vm.put_hf_cache(FILES)
    vm.run("tpu_models")
    r = vm.run("tpu_models")
    assert r.returncode == 0, said(r)
    assert "present" in r.stdout and "verified" not in r.stdout
    assert len(vm.log("mount")) == 1 and len(vm.fstab.read_text().splitlines()) == 1


def test_a_damaged_blob_fails_the_hash_check(vm):
    hub = vm.put_hf_cache(FILES)
    blob = hub / "blobs" / hashlib.sha256(FILES["model-00001-of-00002.safetensors"]).hexdigest()
    blob.write_bytes(os.urandom(4096))                    # same name, other bytes
    r = vm.run("tpu_models")
    assert r.returncode == 1 and "hash mismatch: model-00001-of-00002.safetensors" in r.stdout


def test_a_snapshot_entry_that_points_outside_the_blobs_is_refused(vm):
    hub = vm.put_hf_cache(FILES)
    outside = vm.root / "elsewhere.bin"
    outside.write_bytes(b"x")
    snap = hub / "snapshots" / REV
    (snap / "config.json").unlink()
    (snap / "config.json").symlink_to(outside)
    r = vm.run("tpu_models")
    assert r.returncode == 1 and "outside" in r.stderr and "config.json" in r.stderr


def test_a_missing_revision_is_an_error_naming_it(vm):
    vm.put_hf_cache(FILES)
    t = yaml.safe_load(vm.target.read_text())
    t["model"]["revision"] = "0" * 40
    vm.target.write_text(yaml.safe_dump(t))
    r = vm.run("tpu_models")
    assert r.returncode == 1 and "0" * 40 in r.stderr


# ----------------------------------------------------------- the compile seed
def test_the_seed_is_filled_from_the_disk_and_handed_to_bench(vm):
    src = vm.mnt / "artifacts" / "gemma4-vllm"
    (src / "xla").mkdir(parents=True)
    (src / "xla" / "a.bin").write_bytes(b"compiled")
    r = vm.run("tpu_compile_seed")
    assert r.returncode == 0, said(r)
    seed = vm.root / "bench-cache"
    assert (seed / "xla" / "a.bin").read_bytes() == b"compiled"
    assert any("bench:bench" in l and str(seed) in l for l in vm.log("chown"))
    assert seed.stat().st_mode & 0o077 == 0 and (seed / "xla" / "a.bin").stat().st_mode & 0o077 == 0


def test_a_rerun_adds_to_the_seed_and_never_takes_from_it(vm):
    src = vm.mnt / "artifacts" / "gemma4-vllm"
    src.mkdir(parents=True)
    (src / "new.bin").write_bytes(b"1")
    (src / "shape.bin").write_bytes(b"the disk's older copy")
    seed = vm.root / "bench-cache"
    seed.mkdir()
    (seed / "compiled.bin").write_bytes(b"hours")         # what the baseline launch compiled since
    (seed / "shape.bin").write_bytes(b"recompiled here")
    r = vm.run("tpu_compile_seed")
    assert r.returncode == 0, said(r)
    assert (seed / "new.bin").read_bytes() == b"1"
    assert (seed / "compiled.bin").read_bytes() == b"hours"
    assert (seed / "shape.bin").read_bytes() == b"recompiled here"
    assert "kept" in r.stdout


def test_a_disk_without_a_compile_cache_is_an_error(vm):
    r = vm.run("tpu_compile_seed")
    assert r.returncode == 1 and "no compile cache" in r.stderr


# ------------------------------------------------------------- engine sources
def head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def test_each_source_tree_is_copied_from_the_image_and_gets_a_baseline_commit(vm):
    r = vm.run("tpu_engine_sources")
    assert r.returncode == 0, said(r)
    for name in ("tpu_inference", "vllm"):
        tree = vm.claude / f"{name}-src"
        assert (tree / "pkg" / "mod.py").read_text() == "x = 1\n"
        assert head(tree) and (tree / ".git").is_dir()
        status = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True)
        assert status.stdout == ""                        # the researcher starts from a clean diff
    assert any("researcher:researcher" in l and "tpu_inference-src" in l for l in vm.log("chown"))
    assert "docker rm -f cid1" in vm.log("docker")
    assert not (vm.claude / "sglang-src").exists()


def test_an_existing_tree_is_not_overwritten(vm):
    vm.run("tpu_engine_sources")
    mine = vm.claude / "vllm-src" / "pkg" / "mod.py"
    mine.write_text("x = 2\n")                            # the researcher's edit
    n = len(vm.log("docker"))
    r = vm.run("tpu_engine_sources")
    assert r.returncode == 0 and "left alone" in r.stdout
    assert mine.read_text() == "x = 2\n"
    assert not [l for l in vm.log("docker")[n:] if l.startswith("docker create")]


def test_the_trees_are_re_extracted_on_request(vm):
    vm.run("tpu_engine_sources")
    mine = vm.claude / "vllm-src" / "pkg" / "mod.py"
    mine.write_text("x = 2\n")
    r = vm.run("tpu_engine_sources", IJ_REINSTALL_ENGINES="1")
    assert r.returncode == 0, said(r)
    assert mine.read_text() == "x = 1\n"


def test_an_image_without_a_listed_tree_names_the_target_field(vm):
    (vm.fake / "image_trees").write_text("vllm\n")
    r = vm.run("tpu_engine_sources")
    assert r.returncode == 1 and "tpu_inference" in r.stderr and "engine.sources" in r.stderr
    assert not (vm.claude / "tpu_inference-src").exists()                # no half-copied tree left behind


def test_a_tree_that_already_has_history_keeps_it(vm):
    tree = vm.claude / "vllm-src"
    tree.mkdir()
    (tree / "a.py").write_text("1\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "theirs"]):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True, env=env)
    before = head(tree)
    r = vm.run("tpu_engine_sources")
    assert r.returncode == 0, said(r)
    assert head(tree) == before


def test_a_tree_the_image_shipped_patched_gets_the_patch_into_the_baseline(vm):
    (vm.fake / "image_dirty_git").write_text("")
    r = vm.run("tpu_engine_sources")
    assert r.returncode == 0, said(r)
    tree = vm.claude / "vllm-src"
    status = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True)
    assert status.stdout == "" and "patched by the image build" in (tree / "pkg" / "mod.py").read_text()
    log = subprocess.run(["git", "-C", str(tree), "log", "--format=%s"], capture_output=True, text=True).stdout.split("\n")
    assert any(l.startswith("baseline:") for l in log) and "shipped" in log     # history kept, patch on top


def test_a_researchers_edit_is_never_swept_into_the_baseline_on_a_rerun(vm):
    vm.run("tpu_engine_sources")
    tree = vm.claude / "vllm-src"
    (tree / "pkg" / "mod.py").write_text("x = 2\n")
    before = head(tree)
    assert vm.run("tpu_engine_sources").returncode == 0
    assert head(tree) == before and (tree / "pkg" / "mod.py").read_text() == "x = 2\n"


def test_the_seed_parent_directory_is_made_for_bench_too(vm):
    (vm.mnt / "artifacts" / "gemma4-vllm").mkdir(parents=True)
    vm.run("tpu_compile_seed")
    assert f"install -d -o bench -g bench -m 0750 {vm.root}" in vm.log("install")[0]


# --------------------------------------------------------- the target's list
def test_the_target_file_reads_into_the_arrays_bootstrap_uses(vm):
    r = vm.run('echo "src=${T_SOURCES[*]} flags=${T_DOCKER_FLAGS[*]} env=${T_ENV[*]} shm=$T_SHM"; '
               'echo "disk=$T_DISK_LABEL seed=$T_COMPILE_SEED base=$T_BASELINE"; engine_src_dirs')
    assert r.returncode == 0, said(r)
    assert "src=tpu_inference vllm flags=--privileged" in r.stdout
    env_line = next(l for l in r.stdout.splitlines() if " env=" in l)
    assert sorted(env_line.split(" env=")[1].rsplit(" shm=", 1)[0].split()) == [
        "MODEL_IMPL_TYPE=flax_nnx", "USE_BATCHED_RPA_KERNEL=1"]
    assert env_line.endswith("shm=16g")
    assert "disk=gemma4-cache" in r.stdout and "base=baseline-tpu" in r.stdout
    assert r.stdout.splitlines()[-2:] == [f"{vm.claude}/tpu_inference-src", f"{vm.claude}/vllm-src"]


def test_a_target_without_engine_sources_has_no_tree(vm):
    t = yaml.safe_load(vm.target.read_text())
    t["engine"].pop("sources")
    vm.target.write_text(yaml.safe_dump(t))
    r = vm.run('echo "n=${#T_SOURCES[@]}"; engine_src_dirs; plan_steps')
    assert r.returncode == 0, said(r)
    assert r.stdout.splitlines()[0] == "n=0"
    assert "skip|engine source trees|the target lists none" in r.stdout
    assert "tpu_engine_sources" not in r.stdout


def test_s_target_writes_the_source_list_the_hooks_read(vm):
    bench = vm.root / "opt-bench"
    (bench / "targets").mkdir(parents=True)
    (bench / "target.py").write_text("")                 # `validate` of the stub: exits 0 (the schema is bench/tests')
    r = vm.run(f'BENCH_DIR="{bench}"; TARGET_BENCH="{bench}/targets/$T_ID.yaml"; TARGET_ID_FILE="{bench}/TARGET"; '
               f'TARGET_PUBLIC="{vm.root}/etc/target.yaml"; ENGINE_SOURCES_FILE="{vm.root}/etc/engine-sources"; '
               f'IJ_REPO="{vm.root}"; s_target')
    assert r.returncode == 0, said(r)
    assert (vm.root / "etc" / "engine-sources").read_text() == "tpu_inference\nvllm\n"
    assert any("root:root" in l and "engine-sources" in l for l in vm.log("chown"))


# ------------------------------------------------------------------ the plan
def plan_of(vm) -> list[str]:
    r = subprocess.run(["/bin/bash", str(REPO / "setup" / "bootstrap.sh"), "--target", str(vm.target), "--plan"],
                       env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "HOME": str(vm.root), "LC_ALL": "C",
                            "IJ_TARGET_PYTHON": sys.executable},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_the_plan_for_the_gemma_target_runs_the_tpu_steps_in_order_and_no_gpu_ones(vm):
    lines = plan_of(vm)
    run = [l[6:] for l in lines if l.startswith("RUN ")]
    order = ["preflight", "tpu: the host shows all 4 chips", "docker: daemon", "users bench", "tpu: cache disk",
             "engine image (docker load", "tpu: chip gate", "engine source trees (tpu_inference vllm)",
             "from the disk's Hub cache", "install /opt/bench", "compile-cache seed", "install the target"]
    at = [next(i for i, l in enumerate(run) if want in l) for want in order]
    assert at == sorted(at), dict(zip(order, at))
    text = "\n".join(lines)
    for gone in ("nvidia", "nsight (", "clone vllm", "baseline-vllm", "gpu:"):
        assert gone not in text.replace("accelerator driver (nvidia-smi, CUDA)", ""), gone
    assert "TODO" not in text
    assert "SKIP  reference + calibration" in text


# ------------------------------------------------- a target not yet runnable
def test_the_plan_for_a_sketch_target_stops_after_the_gate_and_before_anything_it_cannot_finish(vm):
    vm.write_target(runnable=False)
    lines = plan_of(vm)
    at = {want: next(i for i, l in enumerate(lines) if want in l)
          for want in ("tpu: cache disk", "engine image (docker load", "tpu: chip gate", "STOP ",
                       "engine source trees", "research clock")}
    assert list(at.values()) == sorted(at.values()), at
    assert lines[at["STOP "]].startswith("STOP  ") and "runnable: false" in lines[at["STOP "]]
    vm.write_target()
    assert not any(l.startswith("STOP") for l in plan_of(vm))


# run_plan with every step but the gate replaced by a line saying it ran.
RUN_THE_PLAN = ('while IFS="|" read -r kind name fn mode; do '
                'if [ "$kind" = step ] && [ "$fn" != tpu_gate ]; then eval "$fn() { echo RAN:$fn; }"; fi; '
                'done < <(plan_steps); run_plan; checklist; echo FINISHED')


def ran(r: subprocess.CompletedProcess) -> list[str]:
    return [l[4:] for l in r.stdout.splitlines() if l.startswith("RAN:")]


def test_a_sketch_target_stops_before_the_clock_and_says_what_to_do_next(vm):
    vm.write_target(runnable=False)
    r = vm.run(RUN_THE_PLAN, FAKE_GATE1_RC="1", FAKE_GATE2_RC="0")
    assert r.returncode == 3, said(r)
    did = ran(r)
    assert did[-2:] == ["tpu_mount_disk", "tpu_engine_image"] and "s_docker" in did
    assert "FINISHED" not in r.stdout
    todo = r.stdout[r.stdout.index("stopped here on purpose"):]
    at = [todo.index(want) for want in (f"sudo bash tools/tpu_seal_probe.sh '{IMAGE}' 4", "engine.docker_flags",
                                        "runnable: true", f"sudo bash setup/bootstrap.sh --target {vm.target}")]
    assert at == sorted(at), todo
    assert str(vm.target) in todo and "--network none" in todo and "VBAR_CONTROL_SERVICE_URL" in todo
    assert "clock is not started" in todo
    checklist = r.stdout[r.stdout.index("bootstrap checklist"):r.stdout.index("stopped here on purpose")]
    assert "PASS   tpu: chip gate" in checklist and "STOP   target is runnable: false" in checklist
    assert "FAIL" not in checklist                     # and the advice is the last thing on the screen


def test_when_the_gate_already_reached_the_chips_there_is_nothing_to_probe(vm):
    vm.write_target(runnable=False)
    r = vm.run(RUN_THE_PLAN)
    assert r.returncode == 3, said(r)
    todo = r.stdout[r.stdout.index("stopped here on purpose"):]
    assert "tpu_seal_probe.sh" not in todo and "own engine.docker_flags" in todo
    assert "runnable: true" in todo and f"sudo bash setup/bootstrap.sh --target {vm.target}" in todo


def test_once_the_target_is_runnable_the_same_command_runs_to_the_clock(vm):
    r = vm.run(RUN_THE_PLAN)
    assert r.returncode == 0, said(r)
    did = ran(r)
    assert did[-1] == "s_clock" and "tpu_engine_sources" in did and "tpu_compile_seed" in did
    assert "FINISHED" in r.stdout and "STOP" not in r.stdout

"""The target file: the one place that names a chip, an engine or a model.

Everything here goes through target.py's interface: load_target, target_hash,
ruler_hash, seal_notes and the `get` / `validate` / `ruler-hash` CLI that the
shell callers use.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import target

BENCH = Path(__file__).resolve().parent.parent
TARGETS = BENCH.parent / "targets"


def write(tmp_path: Path, data: dict, name: str = "t.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def cpu_mock() -> dict:
    return yaml.safe_load((TARGETS / "cpu-mock.yaml").read_text())


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(BENCH / "target.py"), *args],
                          capture_output=True, text=True, timeout=30)


# ------------------------------------------------------------ shipped files --
@pytest.mark.parametrize("name", ["cpu-mock.yaml", "sketch-tpu-8chip.yaml",
                                  "v6e4-gemma4-26b-tps.yaml",
                                  "gpu-gemma4-26b-tps.yaml"])
def test_shipped_targets_validate(name):
    t = target.load_target(TARGETS / name)
    assert t["id"] == name.removesuffix(".yaml")


def test_cpu_mock_is_the_runnable_tps_mock():
    t = target.load_target(TARGETS / "cpu-mock.yaml")
    assert t["engine"]["name"] == "mock"
    assert t["accelerator"]["kind"] == "none"
    assert t["metric"]["kind"] == "tps"
    assert t["runnable"] is True
    # quality2's built-in default for the mock must be what the file says
    assert t["model"]["served_name"] == "mock"
    assert t["model"]["tokenizer"] == "ws"


def test_tpu_sketch_validates_but_is_not_runnable():
    t = target.load_target(TARGETS / "sketch-tpu-8chip.yaml")
    assert t["accelerator"]["kind"] == "tpu"
    assert t["runnable"] is False
    assert "privileged" in target.seal_notes(t)


# ---------------------------------------------------------------- validation --
@pytest.mark.parametrize("mutate,needle", [
    (lambda t: t.pop("id"), "id"),
    (lambda t: t.update(id="Bad_Id"), "id"),
    (lambda t: t["accelerator"].update(kind="Not A Kind"), "accelerator.kind"),
    (lambda t: t["accelerator"].update(kind=""), "accelerator.kind"),
    (lambda t: t["accelerator"].update(chips=0), "accelerator.chips"),
    (lambda t: t["metric"].update(kind="latency"), "metric.kind"),
    (lambda t: t["metric"].update(concurrency=[]), "metric.concurrency"),
    (lambda t: t["metric"].update(concurrency=[4, "8"]), "metric.concurrency"),
    (lambda t: t["metric"].pop("osl"), "metric.osl"),
    (lambda t: t["metric"]["slo"].pop("p99_tpot_ms"), "metric.slo.p99_tpot_ms"),
    (lambda t: t["metric"].update(design="latin-square"), "metric.design"),
    (lambda t: t["budgets"].update(health_s=-1), "budgets.health_s"),
    (lambda t: t.pop("reference"), "reference"),
    (lambda t: t["engine"].update(docker_flags="--privileged"), "engine.docker_flags"),
    (lambda t: t.update(quality={"allow_missing_logprobs": "yes"}),
     "quality.allow_missing_logprobs"),
    (lambda t: t["metric"].update(windw_s=3), "metric.windw_s"),
])
def test_invalid_targets_are_refused_with_the_field_named(tmp_path, mutate, needle):
    t = cpu_mock()
    mutate(t)
    with pytest.raises(target.TargetError, match=needle.replace(".", r"\.")):
        target.load_target(write(tmp_path, t))


@pytest.mark.parametrize("flags,needle", [
    (["-p", "8000:8000"], "publish"),
    (["--publish=8000:8000"], "publish"),
    (["-v", "/data:/data"], "writable"),
    (["-v", "/data:/data:rw"], "writable"),
    (["--volume=/data:/data"], "writable"),
    (["--mount", "type=bind,src=/a,dst=/b"], "writable"),
    (["--network", "bridge"], "network"),
    (["--gpus", "all"], "accelerator"),
])
def test_docker_flags_cannot_unseal_the_launch(tmp_path, flags, needle):
    t = cpu_mock()
    t["engine"]["docker_flags"] = flags
    with pytest.raises(target.TargetError, match=needle):
        target.load_target(write(tmp_path, t))


def test_read_only_mounts_privileged_and_host_network_are_allowed_but_noted(tmp_path):
    t = cpu_mock()
    t["engine"]["docker_flags"] = ["--privileged", "--net=host", "-v",
                                   "/mnt/models:/opt/models:ro",
                                   "--mount", "type=bind,src=/a,dst=/b,readonly"]
    loaded = target.load_target(write(tmp_path, t))
    assert target.seal_notes(loaded) == ["privileged", "network=host"]
    assert target.seal_notes(target.load_target(TARGETS / "cpu-mock.yaml")) == []


def test_unknown_top_level_sections_belong_to_other_tools(tmp_path):
    """provision/probes/price are bootstrap's; the loader keeps them."""
    t = cpu_mock()
    t["provision"] = {"zone": "europe-west4-a"}
    assert target.load_target(write(tmp_path, t))["provision"]["zone"] == "europe-west4-a"


def test_defaults_are_filled_in(tmp_path):
    t = cpu_mock()
    t.pop("runnable", None)
    t["metric"].pop("design", None)
    t["metric"].pop("blocks", None)
    loaded = target.load_target(write(tmp_path, t))
    assert loaded["runnable"] is True
    assert loaded["metric"]["design"] == "abba"
    assert loaded["metric"]["blocks"] >= 1
    assert loaded["quality"]["allow_missing_logprobs"] is False


def test_not_a_mapping_and_bad_yaml_are_refused(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(target.TargetError):
        target.load_target(p)
    p.write_text("id: [unclosed\n")
    with pytest.raises(target.TargetError):
        target.load_target(p)
    with pytest.raises(target.TargetError):
        target.load_target(tmp_path / "missing.yaml")


# --------------------------------------------------------------------- hashes --
def test_target_hash_is_canonical_json_not_file_bytes(tmp_path):
    t = cpu_mock()
    a = target.load_target(write(tmp_path, t, "a.yaml"))
    reordered = dict(reversed(list(t.items())))
    b_path = tmp_path / "b.yaml"
    b_path.write_text("# a comment changes the bytes, not the target\n"
                      + yaml.safe_dump(reordered, sort_keys=False))
    b = target.load_target(b_path)
    assert target.target_hash(a) == target.target_hash(b)
    c = copy.deepcopy(a)
    c["metric"]["osl"] += 1
    assert target.target_hash(c) != target.target_hash(a)
    assert target.target_hash(a) == hashlib.sha256(json.dumps(
        a, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ruler_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    for name in target.RULER_FILES:
        (root / name).write_text(f"# {name}\n")
    (root / "scratch.py").write_text("# not part of the ruler\n")
    t = write(tmp_path, cpu_mock(), "cpu-mock.yaml")
    return root, t


def test_ruler_hash_formula_is_pinned(tmp_path):
    """DESIGN 2.6, in the exact byte layout the scorecard re-implements:
    files sorted by name, each fed as name NUL bytes NUL."""
    root, t = ruler_tree(tmp_path)
    h = hashlib.sha256()
    files = [root / n for n in target.RULER_FILES] + [t]
    for f in sorted(files, key=lambda p: p.name):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    assert target.ruler_hash(root, t) == h.hexdigest()


def test_ruler_hash_moves_with_every_ruler_file_and_the_target(tmp_path):
    root, t = ruler_tree(tmp_path)
    before = target.ruler_hash(root, t)
    (root / "scratch.py").write_text("# edited: not a ruler file\n")
    assert target.ruler_hash(root, t) == before
    for name in target.RULER_FILES:
        p = root / name
        old = p.read_bytes()
        p.write_bytes(old + b"# edit\n")
        assert target.ruler_hash(root, t) != before, name
        p.write_bytes(old)
    t.write_text(t.read_text() + "# edit\n")
    assert target.ruler_hash(root, t) != before


def test_ruler_files_cover_the_contract_list():
    assert set(target.RULER_FILES) == {
        "harness.py", "metrics.py", "compare.py", "quality2.py",
        "stats.py", "tokenizer.py", "target.py", "run.sh"}


# ------------------------------------------------------------------------ CLI --
def test_cli_get_prints_scalars_and_json():
    f = str(TARGETS / "cpu-mock.yaml")
    assert cli("get", f, "metric.kind").stdout.strip() == "tps"
    assert json.loads(cli("get", f, "metric.concurrency").stdout) == \
        target.load_target(TARGETS / "cpu-mock.yaml")["metric"]["concurrency"]
    assert json.loads(cli("get", f, "metric.slo").stdout)["p99_tpot_ms"] > 0
    assert cli("get", f, "runnable").stdout.strip() == "true"
    assert cli("get", f, "engine.docker_flags").stdout.strip() == "[]"


def test_cli_get_missing_key_fails():
    p = cli("get", str(TARGETS / "cpu-mock.yaml"), "metric.nope")
    assert p.returncode != 0
    assert "metric.nope" in p.stderr


def test_cli_validate(tmp_path):
    assert cli("validate", str(TARGETS / "cpu-mock.yaml")).returncode == 0
    t = cpu_mock()
    t["accelerator"]["kind"] = "Not A Kind"
    p = cli("validate", str(write(tmp_path, t)))
    assert p.returncode != 0
    assert "accelerator.kind" in p.stderr


def test_cli_hashes(tmp_path):
    root, t = ruler_tree(tmp_path)
    assert cli("ruler-hash", str(root), str(t)).stdout.strip() == target.ruler_hash(root, t)
    assert cli("hash", str(t)).stdout.strip() == target.target_hash(target.load_target(t))


# ----------------------------------------------------------------- the lock --
def test_lock_round_trip_and_refusal_after_an_edit(tmp_path):
    root, t = ruler_tree(tmp_path)
    lock = tmp_path / "RULER_LOCK"
    h = target.write_lock(root, t, lock)
    assert h == target.ruler_hash(root, t)
    assert target.check_lock(root, t, lock)[0] is True
    (root / "compare.py").write_text("# PROMOTE_EVERYTHING\n")
    ok, msg = target.check_lock(root, t, lock)
    assert ok is False and "ruler" in msg


def test_the_lock_is_never_overwritten(tmp_path):
    root, t = ruler_tree(tmp_path)
    lock = tmp_path / "RULER_LOCK"
    target.write_lock(root, t, lock)
    with pytest.raises(target.TargetError, match="already exists"):
        target.write_lock(root, t, lock)


def test_a_lock_covers_the_target_it_names(tmp_path):
    root, t = ruler_tree(tmp_path)
    lock = tmp_path / "RULER_LOCK"
    target.write_lock(root, t, lock)
    t.write_text(t.read_text() + "# edited\n")
    assert target.check_lock(root, t, lock)[0] is False
    other = write(tmp_path, cpu_mock(), "other.yaml")
    ok, msg = target.check_lock(root, other, lock)
    assert ok is False and "target" in msg


def test_a_bare_hash_lock_covers_the_bench_files_only(tmp_path):
    """The scorecard writes the hash and nothing else."""
    root, t = ruler_tree(tmp_path)
    lock = tmp_path / "RULER_LOCK"
    lock.write_text(target.ruler_hash(root) + "\n")
    assert target.check_lock(root, t, lock)[0] is True
    (root / "run.sh").write_text("# edited\n")
    assert target.check_lock(root, t, lock)[0] is False


def test_cli_lock_and_check_lock(tmp_path):
    root, t = ruler_tree(tmp_path)
    lock = tmp_path / "RULER_LOCK"
    assert cli("lock", str(root), str(t), str(lock)).returncode == 0
    assert cli("lock", str(root), str(t), str(lock)).returncode != 0
    assert cli("check-lock", str(root), str(t), str(lock)).returncode == 0
    (root / "stats.py").write_text("# edited\n")
    p = cli("check-lock", str(root), str(t), str(lock))
    assert p.returncode == 1 and "ruler" in p.stderr


def test_cli_shell_is_eval_safe(tmp_path):
    t = cpu_mock()
    t["engine"]["env"] = {"XLA_FLAGS": "--a=1 --b='x'", "Q": "$(touch pwned)"}
    t["engine"]["docker_flags"] = ["--privileged"]
    f = write(tmp_path, t)
    out = cli("shell", str(f), "/opt/bench").stdout
    script = out + ('printf "%s|" "$T_ID" "$T_ACCEL" "$T_CONCURRENCY" "$T_REF_DIR" '
                    '"${T_ENV[@]}" "${T_DOCKER_FLAGS[@]}" "$T_SEAL_NOTES"')
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split("|")[:-1] == [
        "cpu-mock", "none", "4,8,16", "/opt/bench/reference/cpu-mock",
        "Q=$(touch pwned)", "XLA_FLAGS=--a=1 --b='x'", "--privileged", "privileged"]
    assert not (tmp_path / "pwned").exists()


# ------------------------------------- engine trees, sized mix, cache (TPU) --
def shell_eval(tmp_path: Path, t: dict, expr: str) -> list[str]:
    """The target's shell form, eval'd by bash, then `expr` printed |-joined."""
    out = cli("shell", str(write(tmp_path, t)), "/opt/bench").stdout
    p = subprocess.run(["bash", "-c", out + f'printf "%s|" {expr}'],
                       capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    return p.stdout.split("|")[:-1]


@pytest.mark.parametrize("kind", ["fpga", "npu", "apple-silicon", "lpu"])
def test_any_plain_label_is_an_accelerator_kind(tmp_path, kind):
    """The ruler treats one kind specially (gpu: it passes the GPUs to the
    container). Every other label is the target's own word for its hardware."""
    t = cpu_mock()
    t["accelerator"]["kind"] = kind
    assert target.load_target(write(tmp_path, t))["accelerator"]["kind"] == kind
    assert shell_eval(tmp_path, t, '"$T_ACCEL"') == [kind]


def test_a_target_that_lists_no_engine_sources_has_none(tmp_path):
    """No engine is assumed: a target whose researcher edits no engine tree
    leaves engine.sources out, or writes []."""
    t = cpu_mock()
    t["engine"].pop("sources", None)
    assert target.load_target(write(tmp_path, t))["engine"]["sources"] == []
    assert shell_eval(tmp_path, t, '"${#T_SOURCES[@]}"') == ["0"]
    t["engine"]["sources"] = []
    assert target.load_target(write(tmp_path, t))["engine"]["sources"] == []
    assert shell_eval(tmp_path, t, '"${#T_SOURCES[@]}"') == ["0"]


def test_engine_sources_come_from_the_target(tmp_path):
    t = cpu_mock()
    t["engine"]["sources"] = ["tpu_inference", "vllm"]
    assert shell_eval(tmp_path, t, '"${T_SOURCES[@]}"') == ["tpu_inference", "vllm"]


@pytest.mark.parametrize("sources", [["Vllm"], ["../etc"], ["a b"], ["vllm", "vllm"],
                                     "vllm", ["9lives"], ["x;rm"]])
def test_engine_source_names_are_plain_and_distinct(tmp_path, sources):
    t = cpu_mock()
    t["engine"]["sources"] = sources
    with pytest.raises(target.TargetError, match=r"engine\.sources"):
        target.load_target(write(tmp_path, t))


def test_isl_is_one_size_or_a_list_of_sizes(tmp_path):
    t = cpu_mock()
    one = shell_eval(tmp_path, t, '"$T_ISL"')
    assert one == [str(t["metric"]["isl"])]          # a single size: as before
    t["metric"]["isl"] = [128, 1024, 8000]
    assert shell_eval(tmp_path, t, '"$T_ISL"') == ["128,1024,8000"]
    for bad in ([], [128, 128], [0, 5], [128, "x"], "128,256"):
        t["metric"]["isl"] = bad
        with pytest.raises(target.TargetError, match=r"metric\.isl"):
            target.load_target(write(tmp_path, t))


def test_screen_window_and_length_tol_reach_the_shell(tmp_path):
    t = cpu_mock()
    assert shell_eval(tmp_path, t, '"[$T_SCREEN_WINDOW_S]" "$T_LENGTH_TOL"') == ["[]", "0.0"]
    t["metric"].update(screen_window_s=45, length_tol=0.01)
    assert shell_eval(tmp_path, t, '"$T_SCREEN_WINDOW_S" "$T_LENGTH_TOL"') == ["45", "0.01"]
    t["metric"]["screen_window_s"] = 0
    with pytest.raises(target.TargetError, match=r"metric\.screen_window_s"):
        target.load_target(write(tmp_path, t))


def test_compile_cache_is_a_seed_and_a_mount_point(tmp_path):
    t = cpu_mock()
    assert shell_eval(tmp_path, t, '"[$T_CACHE_SEED]" "[$T_CACHE_MOUNT]"') == ["[]", "[]"]
    t["engine"]["compile_cache"] = {"seed": "/opt/bench/cache/x", "mount": "/root/.cache/vllm"}
    assert shell_eval(tmp_path, t, '"$T_CACHE_SEED" "$T_CACHE_MOUNT"') == [
        "/opt/bench/cache/x", "/root/.cache/vllm"]
    for bad in ({"seed": "/a"}, {"seed": "rel", "mount": "/m"},
                {"seed": "/a", "mount": "/m:rw"}, {"seed": "/a", "mount": "/ro/exp"},
                {"seed": "/a", "mount": "/src"}, {"seed": "/a", "mount": "/m", "x": 1},
                {"seed": "/a b", "mount": "/m"}, "/a:/m"):
        t["engine"]["compile_cache"] = bad
        with pytest.raises(target.TargetError, match=r"engine\.compile_cache"):
            target.load_target(write(tmp_path, t))


def test_the_ready_wait_is_the_targets_own_when_it_names_one(tmp_path):
    """How long a launch may take to answer /v1/models. By default the compile
    budget and the health budget together: a server that compiles before it
    serves needs both. A server that answers first and compiles on its first
    requests says so in health.ready_timeout_s, and a hung start then costs
    that, not the compile budget on top."""
    t = cpu_mock()
    b = t["budgets"]
    assert shell_eval(tmp_path, t, '"$T_READY_S"') == [str(b["compile_s"] + b["health_s"])]
    t["health"] = {"url": "http://127.0.0.1:8000/v1/models", "ready_timeout_s": 300}
    assert shell_eval(tmp_path, t, '"$T_READY_S" "$T_COMPILE_S"') == [
        "300", str(b["compile_s"])]
    t["health"] = {"url": "http://127.0.0.1:8000/v1/models"}
    assert shell_eval(tmp_path, t, '"$T_READY_S"') == [str(b["compile_s"] + b["health_s"])]
    for bad in ({"ready_timeout_s": 0}, {"ready_timeout_s": "soon"},
                {"ready_timeout": 300}, "300"):
        t["health"] = bad
        with pytest.raises(target.TargetError, match="health"):
            target.load_target(write(tmp_path, t))
    gemma = target.load_target(TARGETS / "v6e4-gemma4-26b-tps.yaml")
    assert gemma["health"]["ready_timeout_s"] < (gemma["budgets"]["compile_s"]
                                                 + gemma["budgets"]["health_s"])


def test_host_devices_are_named_in_the_seal_notes(tmp_path):
    """The strong TPU seal: the chips as devices, the control socket read-only,
    no network. It is the strongest launch that reaches the chips and it is
    not a full seal: every host device handed in is a seal note (the socket
    is found by the probe, on the host). A writable mount is still refused
    even when it follows a two-token --device."""
    t = cpu_mock()
    t["engine"]["docker_flags"] = ["--device", "/dev/vfio/vfio", "--device=/dev/vfio/0",
                                   "-v", "/var/run/tpu-plugin:/var/run/tpu-plugin:ro"]
    loaded = target.load_target(write(tmp_path, t))
    assert target.seal_notes(loaded) == ["device /dev/vfio/vfio", "device /dev/vfio/0"]
    t["engine"]["docker_flags"] = ["--privileged", "--device=/dev/fuse:/dev/fuse:rwm"]
    loaded = target.load_target(write(tmp_path, t))
    assert target.seal_notes(loaded) == ["privileged", "device /dev/fuse"]
    t["engine"]["docker_flags"] = ["--device", "-v", "/data:/data"]
    with pytest.raises(target.TargetError, match="--device"):
        target.load_target(write(tmp_path, t))
    t["engine"]["docker_flags"] = ["--device", "/dev/vfio/0", "-v", "/data:/data"]
    with pytest.raises(target.TargetError, match="writable"):
        target.load_target(write(tmp_path, t))


def test_gemma_target_is_the_agreed_shape():
    t = target.load_target(TARGETS / "v6e4-gemma4-26b-tps.yaml")
    e, m = t["engine"], t["metric"]
    assert t["runnable"] is False
    assert e["sources"] == ["tpu_inference", "vllm"]
    assert e["compile_cache"] == {"seed": "/opt/bench/cache/v6e4-gemma4-26b-tps",
                                  "mount": "/root/.cache/vllm"}
    assert t["model"]["local_dir"] == "/opt/models/gemma-4-26b-a4b-it"
    assert t["model"]["tokenizer"].startswith("/opt/models/gemma-4-26b-a4b-it/")
    # the strong seal is the default: nothing privileged, no host network, and
    # the chips, which are host devices, named
    assert target.seal_notes(t) == ["device /dev/vfio/vfio"] + [
        f"device /dev/vfio/{i}" for i in range(4)]
    assert not any("accel" in str(p) for p in t.get("probes", {}).values())
    assert "compile_cache" not in t                  # the stale top-level block
    assert isinstance(m["isl"], list) and max(m["isl"]) <= 8192 and m["osl"] == 512
    assert m["concurrency"] == [1] and m["screen_window_s"] > 0
    assert t["provision"]["cache_disk"] == {
        "label": "gemma4-cache", "mount": "/mnt/gemma4-cache",
        "image_tar": "runtime/vllm-tpu-image.tar", "hf_cache": "artifacts/gemma4-hf",
        "compile_cache": "artifacts/gemma4-vllm"}


def test_the_reference_of_a_real_engine_is_three_cold_launches_of_128_tokens(tmp_path):
    """R's floors are its own launch-to-launch noise, so a real engine is
    launched three times for it and each continuation is 128 tokens. One mock
    server measured three times at 64 tokens stands in for that in the tests.
    A target may name both."""
    t = cpu_mock()
    assert shell_eval(tmp_path, t, '"$T_REF_MAX_TOKENS" "[$T_REF_LAUNCHES]"') == ["64", "[]"]
    t["reference"].update(max_tokens=32, launches=2)
    assert shell_eval(tmp_path, t, '"$T_REF_MAX_TOKENS" "$T_REF_LAUNCHES"') == ["32", "2"]
    for bad in ({"max_tokens": 0}, {"launches": 0}, {"launches": "three"}):
        t = cpu_mock()
        t["reference"].update(bad)
        with pytest.raises(target.TargetError, match="reference"):
            target.load_target(write(tmp_path, t))
    gemma = target.load_target(TARGETS / "v6e4-gemma4-26b-tps.yaml")
    gemma["runnable"] = True
    assert shell_eval(tmp_path, gemma, '"$T_REF_MAX_TOKENS" "$T_REF_LAUNCHES"') == ["128", "3"]

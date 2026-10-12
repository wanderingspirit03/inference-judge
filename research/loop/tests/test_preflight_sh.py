"""loop/preflight.sh <id>: CPU-only checks that turn a doomed
ruler slot into a PREFLIGHT_FAILED line.  The last line of output is the
verdict; exit 0 OK, 1 FAILED, 2 usage."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from conftest import LOOP, Lab


def pf(lab: Lab, *args: str, **env) -> tuple[int, str, str]:
    r = lab.run(["bash", str(LOOP / "preflight.sh"), *args], **env)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return r.returncode, (lines[-1] if lines else ""), r.stdout


def test_valid_experiment_passes(lab: Lab):
    lab.experiment("exp-ok")
    rc, last, _ = pf(lab, "exp-ok")
    assert (rc, last) == (0, "PREFLIGHT_OK exp-ok")


def test_usage(lab: Lab):
    assert pf(lab)[0] == 2
    assert pf(lab, "a", "b")[0] == 2


@pytest.mark.parametrize("bad_id", ["ab", "Exp-1", "-rf", "x" * 42, "a/../../b", "a;b"])
def test_bad_ids(lab: Lab, bad_id):
    rc, last, _ = pf(lab, bad_id)
    assert rc == 1 and last.startswith("PREFLIGHT_FAILED") and "does not match" in last
    assert ";" not in last.split(":")[0]       # the id is sanitised in the verdict line


def test_41_chars_is_the_bench_limit(lab: Lab):
    ok_id = "a" * 41
    lab.experiment(ok_id)
    assert pf(lab, ok_id)[0] == 0


def test_reserved_and_used_ids(lab: Lab):
    lab.experiment("baseline-x")
    assert "reserved" in pf(lab, "baseline-x")[1]
    lab.experiment("exp-used")
    (lab.results / "exp-used").mkdir()
    assert "already used" in pf(lab, "exp-used")[1]
    lab.experiment("exp-ledger")
    (lab.state / "runs.jsonl").write_text(json.dumps({"id": "exp-ledger", "verdict": "REJECTED"}) + "\n")
    assert "already used" in pf(lab, "exp-ledger")[1]


def test_previous_preflight_failure_does_not_burn_the_id(lab: Lab):
    lab.experiment("exp-retry")
    (lab.state / "runs.jsonl").write_text(json.dumps({"id": "exp-retry", "verdict": "PREFLIGHT_FAILED"}) + "\n")
    assert pf(lab, "exp-retry")[0] == 0


def test_launch_sh_missing_or_unparseable(lab: Lab):
    (lab.experiments / "exp-empty").mkdir()
    assert "missing experiments/exp-empty/launch.sh" in pf(lab, "exp-empty")[1]
    lab.experiment("exp-syntax", launch="#!/usr/bin/env bash\nif [ x ]; then\n")
    rc, last, _ = pf(lab, "exp-syntax")
    assert rc == 1 and "does not parse" in last


@pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not installed")
def test_shellcheck_error_level_fails(lab: Lab):
    # parses with bash -n, but shellcheck flags it at error severity (SC2148: no shebang)
    lab.experiment("exp-noshebang", launch="echo hi\n")
    rc, last, _ = pf(lab, "exp-noshebang")
    assert rc == 1 and "shellcheck error" in last
    # warnings alone do not fail
    lab.experiment("exp-warn", launch="#!/usr/bin/env bash\necho $UNQUOTED\n")
    assert pf(lab, "exp-warn")[0] == 0


def _git_tree(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "engine.py").write_text("a = 1\nb = 2\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=True)


GOOD_PATCH = """--- a/engine.py
+++ b/engine.py
@@ -1,2 +1,2 @@
 a = 1
-b = 2
+b = 3
"""
BAD_PATCH = GOOD_PATCH.replace("-b = 2", "-b = 9")


def test_patches_must_apply_to_some_engine_tree(lab: Lab, tmp_path):
    tree = tmp_path / "vllm-src"
    _git_tree(tree)
    trees = f"{tmp_path / 'absent'}:{tree}"
    d = lab.experiment("exp-patch")
    (d / "patches").mkdir()
    (d / "patches" / "001.diff").write_text(GOOD_PATCH)
    rc, last, out = pf(lab, "exp-patch", IJ_PATCH_TREES=trees)
    assert rc == 0 and "applies to" in out
    (d / "patches" / "002.patch").write_text(BAD_PATCH)
    rc, last, _ = pf(lab, "exp-patch", IJ_PATCH_TREES=trees)
    assert rc == 1 and "002.patch applies to none" in last
    assert (tree / "engine.py").read_text() == "a = 1\nb = 2\n"     # --check only


def test_deps_txt_parses(lab: Lab):
    d = lab.experiment("exp-deps")
    (d / "deps.txt").write_text("# pinned\nnumpy==1.26.4\nflashinfer-python>=0.2 ; python_version>='3.10'\n"
                                "--extra-index-url https://example.invalid/simple\n"
                                "git+https://github.com/x/y@abc\n")
    assert pf(lab, "exp-deps")[0] == 0
    (d / "deps.txt").write_text("numpy==1.26\n--trusted-host evil\nnot a requirement !!\n")
    rc, last, _ = pf(lab, "exp-deps")
    assert rc == 1 and "deps.txt" in last and "--trusted-host" in last


def test_reports_every_failure_at_once(lab: Lab):
    d = lab.experiment("exp-many", launch="#!/usr/bin/env bash\nif then\n")
    (d / "deps.txt").write_text("-e .\n")
    rc, last, _ = pf(lab, "exp-many")
    assert rc == 1 and last.count(";") >= 1


def test_patch_trees_follow_the_targets_engine_sources(lab: Lab, tmp_path):
    """The Gemma target lists tpu_inference and vllm; sglang-src does not exist there."""
    home = tmp_path / "home"
    home.mkdir()
    tpu, vllm, sglang = (home / f"{n}-src" for n in ("tpu_inference", "vllm", "sglang"))
    for t in (tpu, vllm, sglang):
        _git_tree(t)
    sources = tmp_path / "engine-sources"
    d = lab.experiment("exp-tpu-patch")
    (d / "patches").mkdir()
    (d / "patches" / "001.diff").write_text(GOOD_PATCH)
    env = {"HOME": str(home), "IJ_PATCH_TREES": "", "IJ_ENGINE_SOURCES_FILE": str(sources)}
    sources.write_text("tpu_inference\nvllm\n")
    rc, _, out = pf(lab, "exp-tpu-patch", **env)
    assert rc == 0 and f"applies to {tpu}" in out                 # the first listed tree that takes it
    # a tree the target does not list is never tried, even though it exists and would apply
    sources.write_text("../escape\nsglang-jax\n$(touch x)\n")
    rc, last, _ = pf(lab, "exp-tpu-patch", **env)
    assert rc == 1 and "applies to none" in last and str(sglang) not in last
    # no file: no tree, so a patch has nothing to apply to
    rc, last, _ = pf(lab, "exp-tpu-patch", **{**env, "IJ_ENGINE_SOURCES_FILE": str(tmp_path / "absent")})
    assert rc == 1 and "applies to none" in last and str(vllm) not in last

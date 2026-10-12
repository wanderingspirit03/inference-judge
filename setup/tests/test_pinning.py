"""Engine versions must be pinned, never discovered.

bootstrap.sh is driven through its command line only. The pin check happens before
any step runs, so these runs never reach apt, docker or the network; `git` is not
shimmed here because nothing may call it, and the shim log proves nothing else ran.
"""
from __future__ import annotations

import os

import pytest

import shutil
import sys

from conftest import GPU_BUILT_TARGET, REPO, make_repo

pytestmark = pytest.mark.skipif(os.getuid() == 0, reason="bootstrap would proceed past preflight as root")


def boot(sb, **env):
    """Bootstrap for a target whose engine is built here: the one that needs the pins."""
    repo = make_repo(sb, with_git=False)
    shutil.copy(REPO / "bench" / "target.py", repo / "bench" / "target.py")
    (repo / "targets").mkdir()
    (repo / "targets" / "gpu-built.yaml").write_text(GPU_BUILT_TARGET)
    return sb.run([str(repo / "setup" / "bootstrap.sh"), "--target", "targets/gpu-built.yaml"],
                  IJ_PINS_FILE=str(sb.root / "pins"), IJ_REPO=str(repo),
                  IJ_TARGET_PYTHON=sys.executable, **env)


def test_refuses_to_start_without_pins(sb):
    r = boot(sb)
    assert r.returncode == 2
    assert "IJ_VLLM_TAG" in r.stderr and "IJ_SGLANG_TAG" in r.stderr
    assert "preflight" not in r.stdout          # no step ran
    assert sb.calls() == []


def test_one_missing_pin_is_named(sb):
    r = boot(sb, IJ_VLLM_TAG="v0.29.0")
    assert r.returncode == 2
    assert "IJ_SGLANG_TAG" in r.stderr and "IJ_VLLM_TAG is required" not in r.stderr


@pytest.mark.parametrize("bad", ["latest", "main", "0.29.0", "v0.29.0; reboot", "v0.29.0-foo", "$(id)"])
def test_floating_or_malformed_tags_are_refused(sb, bad):
    r = boot(sb, IJ_VLLM_TAG=bad, IJ_SGLANG_TAG="v0.5.19")
    assert r.returncode == 2
    assert "not a release tag" in r.stderr
    assert sb.calls() == []


@pytest.mark.parametrize("good", ["v0.29.0", "v0.5.19", "v0.6", "v0.6.0rc1", "v0.6.0.post1"])
def test_release_tags_pass_the_pin_check(sb, good):
    r = boot(sb, IJ_VLLM_TAG=good, IJ_SGLANG_TAG="v0.5.19")
    assert "IJ_VLLM_TAG" not in r.stderr and "release tag" not in r.stderr
    assert "preflight" in r.stdout              # got past the check and into step 1 (which stops: not root)


def test_help_does_not_need_pins(sb):
    repo = make_repo(sb, with_git=False)
    r = sb.run([str(repo / "setup" / "bootstrap.sh"), "--help"])
    assert r.returncode == 0 and "IJ_VLLM_TAG" in r.stdout

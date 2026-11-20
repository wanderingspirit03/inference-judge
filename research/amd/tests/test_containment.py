"""The judge's side of the containment pieces: the firewall check that
./submit runs before a judged run (fails closed), the image build from a
Dockerfile, the queue's container handling through the Docker launcher,
and the engine's stop counting containers that survive it.  Stdlib only;
nothing here needs Docker, sudo or root."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AMD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AMD / "lib"))
import engine as E  # noqa: E402
import gpuqueue as Q  # noqa: E402


def script(d: Path, name: str, body: str) -> str:
    p = d / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)
    return str(p)


class FirewallCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_ok_only_on_exit_zero_and_ok(self):
        self.assertIsNone(E.firewall_problem([script(self.tmp, "a", "echo ok")]))

    def test_fails_closed(self):
        cases = {
            "rule absent": script(self.tmp, "b", "echo 'ij-fw: the rule for cand is not in place: x'; exit 1"),
            "exit 0 but silent": script(self.tmp, "c", "exit 0"),
            "exit 0 but other words": script(self.tmp, "d", "echo maybe"),
            "sudo refused": script(self.tmp, "e", "echo 'sudo: a password is required' >&2; exit 1"),
            "no checker": str(self.tmp / "missing"),
        }
        for name, argv0 in cases.items():
            with self.subTest(name):
                why = E.firewall_problem([argv0])
                self.assertIsNotNone(why, name)
                self.assertTrue(why.startswith("the firewall"), why)

    def test_default_checker_is_the_box_copy_through_sudo(self):
        seen = []

        def fake_run(argv, **kw):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 1, "", "")
        with mock.patch.object(E.subprocess, "run", fake_run):
            self.assertIsNotNone(E.firewall_problem())
        self.assertEqual(seen, [["sudo", "-n", "/opt/ij-amd/amd/controller/box/ij-fw", "check"]])

    def test_submit_checks_before_the_judged_run(self):
        src = (AMD / "submit").read_text()
        i, j = src.index("E.firewall_problem()"), src.index("status, reasons, agg, sp = judged_run(")
        self.assertLess(i, j)
        self.assertIn("if not why and user:", src)


class ImageBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = {"engine": {"build_timeout_s": 60}}

    def test_no_dockerfile_no_image(self):
        self.assertIsNone(E.image_build(self.cfg, None, self.tmp, self.tmp, lambda m: None))

    def test_dockerfile_without_launcher_is_an_engine_failure(self):
        (self.tmp / "Dockerfile").write_text("FROM scratch\n")
        with mock.patch.object(Q, "LAUNCHER", str(self.tmp / "no-launcher")):
            with self.assertRaises(E.EngineFailed):
                E.image_build(self.cfg, None, self.tmp, self.tmp, lambda m: None)

    def test_judged_build_is_for_cand(self):
        (self.tmp / "Dockerfile").write_text("FROM scratch\n")
        seen = []

        def fake_run(hold, argv, **kw):
            seen.append(argv)
            return 0
        with mock.patch.object(Q, "LAUNCHER", script(self.tmp, "ij-docker", "exit 0")), \
                mock.patch.object(Q, "run", fake_run):
            self.assertIsNotNone(E.image_build(self.cfg, None, self.tmp, self.tmp, lambda m: None, user="cand"))
        self.assertEqual(seen[0][:3], ["sudo", "-n", str(self.tmp / "ij-docker")])
        self.assertEqual(seen[0][3:], ["build", "--for", "cand", str(self.tmp)])

    def test_image_name(self):
        self.assertEqual(E.image_name("cand"), "ij-cand/engine:latest")


class QueueThroughLauncherTest(unittest.TestCase):
    """builder has no Docker on a box with the launcher: the queue lists and
    removes its job containers through `sudo ij-docker jobs|reap`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.launcher = script(self.tmp, "ij-docker", "exit 0")
        self.calls = []

    def fake_run(self, argv, **kw):
        self.calls.append(argv)
        out = "abc ij-cand-engine tok1 cand\ndef ij-builder-ref tok2 -\n" if argv[3:4] == ["jobs"] else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    def patched(self):
        return [mock.patch.object(Q, "LAUNCHER", self.launcher),
                mock.patch.object(Q, "DOCKER_SOCK", str(self.tmp / "no-docker.sock")),
                mock.patch.object(Q.os, "geteuid", lambda: 1001),
                mock.patch.dict(os.environ, {"IJ_AMD_REAP_SUDO": "1"}),
                mock.patch.object(Q.subprocess, "run", self.fake_run)]

    def test_jobs_and_reap_go_through_the_launcher(self):
        ps = self.patched()
        for p in ps:
            p.start()
        try:
            os.environ.pop("DOCKER_HOST", None)
            self.assertTrue(Q.via_launcher())
            rows = Q.labelled_containers()
            self.assertEqual([(r["name"], r["token"], r["owner"]) for r in rows],
                             [("ij-cand-engine", "tok1", "cand"), ("ij-builder-ref", "tok2", "")])
            self.assertEqual([r["name"] for r in Q.owner_containers("cand")], ["ij-cand-engine"])
            Q._rm_labelled_containers(lambda t: True, "x", None, ["--end", "tok1"])
            Q._rm_labelled_containers(lambda t: True, "x", None, None)        # no args: nothing through the launcher
        finally:
            for p in reversed(ps):
                p.stop()
        self.assertEqual(self.calls[0], ["sudo", "-n", self.launcher, "jobs"])
        self.assertEqual(self.calls[-1], ["sudo", "-n", self.launcher, "reap", "--end", "tok1"])
        self.assertFalse(any(c[0] == "docker" for c in self.calls), self.calls)

    def test_without_the_launcher_docker_is_used(self):
        with mock.patch.object(Q, "LAUNCHER", str(self.tmp / "none")), \
                mock.patch.object(Q.subprocess, "run", self.fake_run):
            Q.labelled_containers()
        self.assertEqual(self.calls[0][:2], ["docker", "ps"])


class KfdUsersThroughReaperTest(unittest.TestCase):
    """judge cannot read the /proc fds of cand or builder: on the GPU machine
    kfd_users() asks the root-owned reaper, and falls back to its own view."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reaper = script(self.tmp, "ij-reap", "exit 0")
        self.calls = []

    def run_with(self, rc, out):
        def fake_run(argv, **kw):
            self.calls.append(argv)
            return subprocess.CompletedProcess(argv, rc, out, "")
        local = [{"pid": 1, "cmd": "own", "token": None}]
        with mock.patch.object(Q, "REAPER", self.reaper), mock.patch.object(Q.os, "geteuid", lambda: 1001), \
                mock.patch.dict(os.environ, {"IJ_AMD_REAP_SUDO": "1"}), \
                mock.patch.object(Q.subprocess, "run", fake_run), mock.patch.object(Q, "kfd_users_local", lambda: local):
            return Q.kfd_users()

    def test_the_reaper_sees_cand(self):
        got = self.run_with(0, '[{"pid": 42, "cmd": "python3 -m engine", "token": null}]')
        self.assertEqual(got, [{"pid": 42, "cmd": "python3 -m engine", "token": None}])
        self.assertEqual(self.calls, [["sudo", "-n", self.reaper, "--kfd-users"]])

    def test_a_refused_reaper_falls_back_to_the_own_view(self):
        self.assertEqual(self.run_with(2, ""), [{"pid": 1, "cmd": "own", "token": None}])
        self.assertEqual(self.run_with(0, "not json"), [{"pid": 1, "cmd": "own", "token": None}])

    def test_without_the_reap_flag_no_sudo(self):
        with mock.patch.object(Q, "REAPER", self.reaper), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(Q.subprocess, "run", lambda *a, **k: self.fail("sudo called")), \
                mock.patch.object(Q, "kfd_users_local", lambda: []):
            self.assertEqual(Q.kfd_users(), [])


class ServerStopTest(unittest.TestCase):
    def test_a_surviving_container_of_the_engine_account_is_counted(self):
        tmp = Path(tempfile.mkdtemp())
        cfg = {"engine": {"port": 18555, "health_path": "/v1/models"}, "check": {"engine_top": 5}}
        srv = E.Server(cfg, None, tmp, tmp, lambda m: None, user="cand")
        with mock.patch.object(E, "kill_user", lambda u, log=None: []), \
                mock.patch.object(Q, "rm_owner_containers", lambda u, log=None: [{"name": "ij-cand-engine"}]):
            self.assertEqual(srv.stop(), ["container ij-cand-engine"])
        with mock.patch.object(E, "kill_user", lambda u, log=None: []), \
                mock.patch.object(Q, "rm_owner_containers", lambda u, log=None: []):
            self.assertEqual(srv.stop(), [])


if __name__ == "__main__":
    unittest.main()

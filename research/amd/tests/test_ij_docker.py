"""controller/box/ij-docker, the Docker launcher of builder and cand, with a
fake docker on PATH (the stand-in machine has no Docker) in the launcher's
test mode (IJ_DOCKER_TEST=1, never as root).  Every refusal must leave the
fake docker uncalled; every run carries the forced flags.  Stdlib only."""
import fcntl
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parent.parent
LAUNCHER = AMD / "controller" / "box" / "ij-docker"

FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
d = os.environ["FAKE_DIR"]
with open(os.path.join(d, "calls.jsonl"), "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
if sys.argv[1:2] == ["build"]:
    open(os.path.join(d, "context.tar"), "wb").write(sys.stdin.buffer.read())
elif sys.argv[1:2] == ["inspect"]:
    print(os.environ.get("FAKE_OWNER", ""))
elif sys.argv[1:2] == ["ps"] and "label=ij.gpu.job" in sys.argv:
    sys.stdout.write(os.environ.get("FAKE_JOBS", ""))
'''

USERS = {"builder": [1001, 1001], "judge": [1002, 1002], "cand": [1003, 1003]}


class Launcher:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        (tmp / "q").mkdir()
        self.docker = tmp / "docker"
        self.docker.write_text(FAKE_DOCKER)
        self.docker.chmod(0o755)
        cfg = {"model": {"dir": "/srv/models/Qwen3.8-27B"},
               "ref": {"image": "ij-atom-ref:pinned", "served_model": "qwen3.8-27b", "gpu_memory_utilization": 0.5}}
        (tmp / "cfg.json").write_text(json.dumps(cfg))
        self.home = tmp / "home"
        for u in ("builder", "judge"):
            (self.home / u).mkdir(parents=True)
        self.lock = None

    def hold(self, token="tok1", pid=None):
        """Hold the queue (as a process of this test) with this token."""
        q = self.tmp / "q"
        self.lock = os.open(q / "lock", os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        (q / "holder.json").write_text(json.dumps({"pid": pid or os.getpid(), "token": token}))

    def release(self):
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def __call__(self, caller, *args, holder_uid=None, owner="", jobs="", stdin=None):
        env = {**os.environ, "IJ_DOCKER_TEST": "1", "IJ_DOCKER_TEST_CALLER": caller,
               "IJ_DOCKER_TEST_USERS": json.dumps(USERS), "IJ_DOCKER_TEST_BIN": str(self.docker),
               "IJ_DOCKER_TEST_QUEUE": str(self.tmp / "q"), "IJ_DOCKER_TEST_CONFIG": str(self.tmp / "cfg.json"),
               "IJ_DOCKER_TEST_CACHE": str(self.tmp / "cache"),
               "IJ_DOCKER_TEST_ROOTS": json.dumps({"builder": [str(self.home / "builder")],
                                                   "judge": [str(self.home / "judge")]}),
               "IJ_DOCKER_TEST_HOLDER_UID": "" if holder_uid is None else str(holder_uid),
               "FAKE_DIR": str(self.tmp), "FAKE_OWNER": owner, "FAKE_JOBS": jobs}
        (self.tmp / "cache").mkdir(exist_ok=True)
        return subprocess.run([sys.executable, str(LAUNCHER), *args], env=env, capture_output=True, text=True)

    def calls(self) -> list[list[str]]:
        p = self.tmp / "calls.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def clear(self):
        (self.tmp / "calls.jsonl").unlink(missing_ok=True)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.L = Launcher(Path(os.path.realpath(self.d.name)))

    def tearDown(self):
        self.L.release()
        self.d.cleanup()

    REFUSED = [
        ["-v", "/:/host"], ["--volume=/:/host"], ["-v/:/host"], ["--volume", "/home/judge:/j"], ["-dv", "/:/host"],
        ["--mount", "type=bind,src=/,dst=/host"], ["--privileged"], ["--network", "host"], ["--network=none"],
        ["--net=host"], ["--pid", "host"], ["--pid=host"], ["--userns", "host"], ["--userns=host"], ["-u", "0"],
        ["--user", "0:0"], ["--user=root"], ["--volumes-from", "ij-judge-ref"], ["--group-add", "0"],
        ["--ipc", "host"], ["--ipc=host"], ["--uts", "host"], ["--device", "/dev/mem"], ["--device=/dev/sda"],
        ["--cap-add", "SYS_ADMIN"], ["--cap-add=NET_ADMIN"], ["--security-opt", "seccomp=unconfined"],
        ["--security-opt=apparmor=unconfined"], ["--env-file", "/etc/shadow"], ["--cidfile", "/etc/x"],
        ["--label-file", "/etc/shadow"], ["--label", "ij.owner=judge"], ["-p", "8100:8100"], ["--add-host", "x:1.2.3.4"],
        ["--dns", "8.8.8.8"], ["--runtime", "runc"], ["--cgroupns", "host"], ["--pull", "always"], ["-e", "NOVALUE"],
        ["-e", "1BAD=x"], ["--shm-size", "500g"], ["--name", "../judge"], ["--workdir", "relative"], ["--", "x"],
    ]

    def test_cand_refusals_never_call_docker(self):
        self.L.hold()
        for opts in self.REFUSED:
            r = self.L("cand", "run", *opts, "ij-cand/engine", "serve", holder_uid=USERS["judge"][0])
            self.assertEqual(r.returncode, 2, (opts, r.stdout, r.stderr))
            self.assertIn("refused", r.stderr, opts)
            self.assertEqual(self.L.calls(), [], f"docker was called for {opts}")

    def test_images_other_than_the_callers_own_are_refused(self):
        self.L.hold()
        for img in ["ubuntu", "ij-builder/engine", "ij-cand", "rocm/atom-dev@sha256:ab", "ij-atom-ref:pinned",
                    "evil.example/ij-cand/engine", "ij-cand/../x"]:
            r = self.L("cand", "run", "--rm", img, holder_uid=USERS["judge"][0])
            self.assertEqual(r.returncode, 2, img)
        self.assertEqual(self.L.calls(), [])

    def test_cand_run_carries_the_forced_flags(self):
        self.L.hold("tokJ")
        r = self.L("cand", "run", "-d", "--rm", "--name", "engine", "-e", "A=1", "--env=B=x y", "--shm-size", "16g",
                   "-w", "/app", "ij-cand/engine", "python3", "-m", "srv", "--port", "8101",
                   holder_uid=USERS["judge"][0])
        self.assertEqual(r.returncode, 0, r.stderr)
        (a,) = self.L.calls()
        s = " ".join(a)
        for want in ["run --pull never --network host --ipc private --user 1003:1003", "--device /dev/kfd",
                     "--device /dev/dri", "--cap-drop ALL", "--security-opt no-new-privileges",
                     "--label ij.gpu.job=tokJ", "--label ij.owner=cand", "--volume /srv/models:/models:ro",
                     "--name ij-cand-engine", "--env A=1", "--shm-size 16g", "--workdir /app"]:
            self.assertIn(want, s)
        self.assertIn("B=x y", a)
        self.assertEqual(a[a.index("ij-cand/engine"):], ["ij-cand/engine", "python3", "-m", "srv", "--port", "8101"])
        self.assertEqual(sum(x.startswith("/srv/models") or x.startswith("/home") for x in a), 1)  # one host path
        # the weights boundary (NOTES, 2026-11-18): the one host path is the models folder, read-only; nothing
        # else from the machine; and no size limit on the container's own files (/tmp is its writable layer,
        # on the Docker data disk) or its memory, only on /dev/shm (--shm-size, at most 64g)
        self.assertEqual([a[i + 1] for i, x in enumerate(a) if x in ("--volume", "-v", "--mount")], ["/srv/models:/models:ro"])
        self.assertIn("HOME=/tmp", a)
        for absent in ("--tmpfs", "--read-only", "--memory", "--storage-opt", "--mount"):
            self.assertNotIn(absent, a)

    def test_run_needs_the_right_holder(self):
        r = self.L("cand", "run", "ij-cand/engine", holder_uid=USERS["judge"][0])
        self.assertEqual(r.returncode, 2)
        self.assertIn("nobody holds", r.stderr)
        self.L.hold()
        r = self.L("builder", "run", "ij-builder/engine", holder_uid=USERS["judge"][0])
        self.assertIn("held by another account", r.stderr)
        r = self.L("cand", "run", "ij-cand/engine", holder_uid=USERS["builder"][0])
        self.assertIn("held by another account", r.stderr)
        r = self.L("judge", "run", "ij-judge/engine", holder_uid=USERS["judge"][0])
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.L.calls(), [])
        r = self.L("builder", "run", "ij-builder/engine", holder_uid=USERS["builder"][0])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_build_packs_the_context_as_the_caller_and_has_no_network(self):
        d = self.L.home / "builder" / "work" / "eng"
        d.mkdir(parents=True)
        (d / "Dockerfile").write_text("FROM scratch\nCOPY . /app\n")
        (d / "leak").symlink_to("/etc/passwd")
        r = self.L("builder", "build", str(d))
        self.assertEqual(r.returncode, 0, r.stderr)
        (a,) = self.L.calls()
        self.assertEqual(a, ["build", "--network", "none", "--label", "ij.built-by=builder", "-t",
                             "ij-builder/engine:latest", "-"])
        with tarfile.open(self.L.tmp / "context.tar") as t:
            m = {x.name.lstrip("./"): x for x in t.getmembers()}
        self.assertIn("Dockerfile", m)
        self.assertTrue(m["leak"].issym(), "a symlink goes in as a symlink, not as the file it names")

    def test_a_judged_build_gets_the_modes_check_builds_with(self):
        """The judge's copy is unpacked with umask 077 (0600, 0700); the image
        must see 0644 and 0755 as ./check's does, or cand cannot read a COPY'd
        file (submission #1, 2026-11-18: PermissionError on eng/__init__.py)."""
        d = self.L.home / "judge" / "runs" / "1" / "src"
        (d / "eng" / "csrc").mkdir(parents=True)
        (d / "Dockerfile").write_text("FROM scratch\nCOPY eng /opt/eng/eng\n")
        (d / "eng" / "__init__.py").write_text("x = 1\n")
        (d / "eng" / "csrc" / "k.hip").write_text("// kernel\n")
        (d / "serve").write_text("#!/bin/sh\n")
        for p, m in [(d / "Dockerfile", 0o600), (d / "eng" / "__init__.py", 0o600), (d / "eng" / "csrc" / "k.hip", 0o600),
                     (d / "serve", 0o700), (d / "eng" / "csrc", 0o700), (d / "eng", 0o700)]:
            p.chmod(m)
        r = self.L("judge", "build", "--for", "cand", str(d))
        self.assertEqual(r.returncode, 0, r.stderr)
        with tarfile.open(self.L.tmp / "context.tar") as t:
            m = {x.name.lstrip("./"): x for x in t.getmembers()
                 if x.name.lstrip("./") and not Path(x.name).name.startswith("._")}     # macOS tar's AppleDouble files
        self.assertEqual({k: oct(v.mode) for k, v in m.items()},
                         {"Dockerfile": "0o644", "eng": "0o755", "eng/__init__.py": "0o644", "eng/csrc": "0o755",
                          "eng/csrc/k.hip": "0o644", "serve": "0o755"})
        with tarfile.open(self.L.tmp / "context.tar") as t:
            self.assertEqual(t.extractfile("./eng/__init__.py" if "./eng/__init__.py" in t.getnames() else "eng/__init__.py").read(), b"x = 1\n")
        self.assertEqual(oct((d / "eng" / "__init__.py").stat().st_mode & 0o777), "0o600")    # the source keeps its own

    def test_build_refusals(self):
        out = self.L.tmp / "elsewhere"
        out.mkdir()
        (out / "Dockerfile").write_text("FROM scratch\n")
        link = self.L.home / "builder" / "link"
        link.symlink_to(out)
        for caller, args in [("cand", ["build", str(self.L.home / "builder")]), ("builder", ["build", str(out)]),
                             ("builder", ["build", str(link)]), ("builder", ["build", "--for", "cand", str(out)]),
                             ("cand", ["build", "--for", "cand", str(out)]),
                             ("judge", ["build", str(self.L.home / "judge")])]:
            r = self.L(caller, *args)
            self.assertEqual(r.returncode, 2, (caller, args, r.stderr))
        self.assertEqual(self.L.calls(), [])
        sub = self.L.home / "judge" / "runs" / "r1" / "src"
        sub.mkdir(parents=True)
        (sub / "Dockerfile").write_text("FROM scratch\n")
        r = self.L("judge", "build", "--for", "cand", str(sub))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ij-cand/engine:latest", self.L.calls()[0])

    def test_reference_profile_is_fixed(self):
        self.L.hold("tokR")
        r = self.L("cand", "ref", "start", "engine", "8101", "0.5", holder_uid=USERS["judge"][0])
        self.assertEqual(r.returncode, 0, r.stderr)
        rm, a = self.L.calls()
        self.assertEqual(rm, ["rm", "-f", "ij-cand-engine"])
        s = " ".join(a)
        self.assertIn("--name ij-cand-engine --label ij.gpu.job=tokR --label ij.owner=cand", s)
        self.assertIn(f"--volume {self.L.tmp}/cache/ij-ref-cand:/root/.cache", s)
        self.assertIn("ij-atom-ref:pinned python3 -m atom.entrypoints.openai_server --model /models/Qwen3.8-27B", s)
        self.assertIn("--server-port 8101", s)
        self.L.clear()
        for args in [["start", "judge-ref", "8001"], ["start", "ref", "80"], ["start", "ref", "8001", "2"],
                     ["start", "ref", "8001", "0.5", "--enable-plugins"], ["start", "ref", "8001;rm"]]:
            r = self.L("cand", "ref", *args, holder_uid=USERS["judge"][0])
            self.assertEqual(r.returncode, 2, args)
        r = self.L("judge", "ref", "start", "ref", "8001", holder_uid=USERS["judge"][0])
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.L.calls(), [])

    def test_rm_and_logs_only_of_own_containers(self):
        r = self.L("builder", "rm", "ref", owner="judge")
        self.assertEqual([c[0] for c in self.L.calls()], ["inspect"])
        self.assertIn("not the caller's", r.stdout)
        self.L.clear()
        r = self.L("cand", "logs", "engine", "5", owner="judge")
        self.assertEqual(r.returncode, 2)
        self.L.clear()
        self.L("cand", "ref", "stop", "engine", owner="cand")
        self.assertEqual(self.L.calls()[-1], ["rm", "-f", "ij-cand-engine"])

    def test_reap_only_by_the_holder(self):
        jobs = "a1 ij-builder-x tokB builder\nb2 ij-cand-engine tokOld cand\n"
        self.L.hold("tokB")
        r = self.L("builder", "reap", "--keep", "tokX", holder_uid=USERS["builder"][0], jobs=jobs)
        self.assertEqual(r.returncode, 2)
        r = self.L("cand", "reap", "--keep", "tokB", holder_uid=USERS["judge"][0], jobs=jobs)
        self.assertEqual(r.returncode, 2)
        self.L.clear()
        r = self.L("builder", "reap", "--keep", "tokB", holder_uid=USERS["builder"][0], jobs=jobs)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c for c in self.L.calls() if c[0] == "rm"], [["rm", "-f", "b2"]])
        self.L.clear()
        self.L("builder", "reap", "--end", "tokB", holder_uid=USERS["builder"][0], jobs=jobs)
        self.assertEqual([c for c in self.L.calls() if c[0] == "rm"], [["rm", "-f", "a1"]])

    def test_not_root_without_test_mode_refuses(self):
        if os.geteuid() == 0:
            self.skipTest("as root")
        r = subprocess.run([sys.executable, str(LAUNCHER), "ps"], capture_output=True, text=True,
                           env={k: v for k, v in os.environ.items() if not k.startswith("IJ_DOCKER")})
        self.assertEqual(r.returncode, 2)
        self.assertIn("through sudo", r.stderr)


if __name__ == "__main__":
    unittest.main()

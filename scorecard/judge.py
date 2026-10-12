"""The one way scenarios reach the judge: this tree's bench/, research/loop/
and hooks, run as their own processes.

Scenarios call only the methods of Judge and never import bench code. Every
call starts the judge's code in a child process the way bench/run.sh starts
it (python -I, bench/ first on sys.path), so what is tested is what a scored
run executes, and the suite's own interpreter state cannot leak into it.
probes/probe.py checks that the modules it loads come from this tree.

Interface (every method returns plain data; none writes outside `work`):

  serve(behaviour)                 context manager -> base_url of the mock
                                   server behaving as `behaviour`
  measure_tps(base_url, tag)       {tps, latency_p95_ms, n, n_invalid, ...}:
                                   the harness's own closed-loop measurement
  quality_gate(ref_url, cand_url, tag)
                                   {passed, detail}: the candidate judged on a
                                   reference built from ref_url
  promotion_decisions(sessions)    {promoted: [bool, ...], ...}: the promotion
                                   rule on each simulated session
  verdict(launches, quality, board)
                                   {verdict, reasons, on_leaderboard}: the
                                   verdict on four launch summaries, then the
                                   promotion step run.sh would take
  run_with_foreign_incumbent()     {verdict, reasons, launched}: a real run
                                   whose incumbent another ruler measured
  run_after_ruler_edit()           {locked_run_ok, refused, rc, detail}
  scored_launch_flags()            docker run flags of a scored launch
  admit_experiment(exp_dir, exp_id, engine_tree)
                                   {slot_spent, verdict, detail}: queue an
                                   experiment and let the runner take one step
  script(rel)                      a script of this tree that a scenario runs
                                   itself
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from core import SCORECARD, free_port, python, run, server
from fixtures import CALIBRATION, prompt_pool

# run.sh's own bootstrap: bench first on sys.path, then runpy the script.
PY_BOOT = ("import sys, runpy\n"
           "sys.path.insert(0, sys.argv[1])\n"
           "_script = sys.argv[2]\n"
           "sys.argv = [_script] + sys.argv[3:]\n"
           "runpy.run_path(_script, run_name='__main__')")
PROBE = SCORECARD / "probes" / "probe.py"
# What a private copy of the ruler is made of (bench/tests/test_run_sh.py).
RULER_FILES = ("harness.py", "metrics.py", "quality2.py", "stats.py", "tokenizer.py",
               "target.py", "compare.py", "run.sh", "mock_server.py")
TARGET = "cpu-mock"
TPS = {"concurrency": 8, "isl": 256, "osl": 64, "ramp_s": 1, "window_s": 4}
RUN_ID = "scorecard-candidate"
# Stands in for the sealed ruler when the runner is driven: it only records
# that it was reached. Reaching it is what spending a slot means.
BENCH_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_CALLS"
id="${!#}"
mkdir -p "$STUB_RESULTS/$id"
printf '{"id":"%s","run_id":"%s-r1","verdict":"REJECTED","duration_s":1,"reasons":["stub"]}\\n' \\
  "$id" "$id" > "$STUB_RESULTS/$id/results.json"
"""


class Judge:
    def __init__(self, repo: Path, work: Path) -> None:
        self.repo = Path(repo).resolve()
        self.bench = self.repo / "bench"
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers --
    def script(self, rel: str) -> Path:
        p = self.repo / rel
        if not p.exists():
            raise FileNotFoundError(f"{rel} is not in this tree")
        return p

    def scratch(self, tag: str) -> Path:
        # resolved: run.sh compares real paths, and /tmp is a link on macOS
        return Path(os.path.realpath(tempfile.mkdtemp(prefix=f"{tag}-", dir=self.work)))

    def bench_py(self, script: Path | str, *args, timeout: float = 300):
        """Run a script with bench/ first on sys.path, python -I."""
        return run([python(), "-I", "-c", PY_BOOT, str(self.bench), str(script),
                    *map(str, args)], timeout=timeout, cwd=self.bench)

    def compare(self, *args, timeout: float = 120):
        return self.bench_py(self.bench / "compare.py", *args, timeout=timeout)

    def probe(self, cmd: str, args: dict, timeout: float = 600) -> dict:
        d = self.scratch("probe")
        (d / "in.json").write_text(json.dumps(args))
        p = self.bench_py(PROBE, cmd, d / "in.json", d / "out.json", timeout=timeout)
        if p.returncode != 0 or not (d / "out.json").exists():
            raise RuntimeError(f"probe {cmd} failed rc={p.returncode}: {p.stderr[-800:]}")
        return json.loads((d / "out.json").read_text())

    def bench_root(self, dest: Path) -> Path:
        """A private copy of the ruler, the way bench/tests/test_run_sh.py makes one."""
        root = dest / "bench_root"
        root.mkdir(parents=True)
        for name in RULER_FILES:
            shutil.copy(self.bench / name, root / name)
        shutil.copytree(self.bench / "canary", root / "canary")
        shutil.copytree(self.repo / "targets", root / "targets")
        return root

    def run_sh(self, root: Path, args: list[str], extra_env: dict | None = None,
               timeout: float = 180):
        env = dict(os.environ)
        env.update({"BENCH_ROOT": str(root), "BENCH_PY": python(),
                    "BENCH_TARGET": TARGET, "HARD_TIMEOUT_S": "600"})
        env.pop("SUDO_USER", None)
        env.update(extra_env or {})
        return run(["bash", str(root / "run.sh"), *args], timeout=timeout, env=env)

    # ------------------------------------------------------- 1. tokens/s --
    @contextlib.contextmanager
    def serve(self, behaviour: str):
        port = free_port()
        argv = [python(), "-I", "-c", PY_BOOT, str(self.bench),
                str(self.bench / "mock_server.py"), "--port", str(port),
                "--behaviour", behaviour]
        with server(argv, f"http://127.0.0.1:{port}/v1/models",
                    self.work / "logs" / f"mock-{behaviour}-{port}.log"):
            yield f"http://127.0.0.1:{port}/v1"

    def _tps_run(self, base_url: str, out: Path, pool: Path, seed: int) -> dict:
        p = self.bench_py(self.bench / "harness.py", "--base-url", base_url,
                          "--model", "mock", "--mode", "tps", "--tps-load", "closed",
                          "--concurrency", TPS["concurrency"], "--isl", TPS["isl"],
                          "--osl", TPS["osl"], "--ramp-s", TPS["ramp_s"],
                          "--window-s", TPS["window_s"], "--prompt-pool", pool,
                          "--seed", seed, "--tokenizer", "ws", "--out", out, timeout=120)
        if p.returncode != 0 or not (out / "summary.json").exists():
            raise RuntimeError(f"harness --mode tps failed rc={p.returncode}: "
                               f"{p.stderr[-600:]}")
        return json.loads((out / "summary.json").read_text())

    def measure_tps(self, base_url: str, tag: str) -> dict:
        """Closed loop, ignore_eos, tokens counted on the client, after a warm
        run on another seed (as a scored launch does). The level summary is
        the harness's own (metrics.tps_level_summary)."""
        d = self.scratch(f"tps-{tag}")
        pool = prompt_pool(d / "pool.jsonl")
        self._tps_run(base_url, d / "warmup", pool, 1)
        summary = self._tps_run(base_url, d / "measured", pool, 2)
        s = summary["metric"]["levels"][0]
        # The headline is what is scored: goodput of a level with 0 invalid
        # requests, inside the SLO and the sanity band, else 0.
        return {"tps": summary["metric"].get("headline"),
                "latency_p95_ms": s.get("p95_ttft_ms"),
                "n": s.get("n"), "n_invalid": s.get("n_invalid"),
                "goodput_tps_per_chip": s.get("goodput_tps_per_chip"),
                "accounting": "client-side token count; goodput = valid requests only",
                "summary": {k: v for k, v in s.items() if not isinstance(v, (list, dict))}}

    # ------------------------------------------------------- 2. quality --
    def quality_gate(self, ref_url: str, cand_url: str, tag: str) -> dict:
        """quality2: build the reference from ref_url, then gate cand_url on it."""
        q2 = self.bench / "quality2.py"
        target = self.repo / "targets" / f"{TARGET}.yaml"
        d = self.scratch(f"quality-{tag}")
        ref = d / "reference"
        p = self.bench_py(q2, "make-reference", "--base-url", ref_url, "--target", target,
                          "--out", ref)
        if p.returncode != 0:
            raise RuntimeError(f"quality2 make-reference failed: {p.stderr[-600:]}")
        out = d / "gate.json"
        p = self.bench_py(q2, "gate", "--base-url", cand_url, "--target", target,
                          "--reference", ref, "--out", out)
        if not out.exists():
            raise RuntimeError(f"quality2 gate wrote nothing rc={p.returncode}: {p.stderr[-600:]}")
        g = json.loads(out.read_text())
        return {"passed": bool(g.get("passed")),
                "detail": {"reasons": g.get("reasons"), "checks": g.get("checks")}}

    # ---------------------------------------------------- 3. statistics --
    def promotion_decisions(self, sessions: list[dict]) -> dict:
        return self.probe("promotions", {"sessions": sessions})

    # ----------------------------------------------- 2b, 4. the verdict --
    def verdict(self, launches: list[dict], quality: list[dict | None], board: dict) -> dict:
        """compare.py tps-verdict on these launch summaries, then tps-promote
        if (and only if) the verdict is PROMOTED: the two steps of run.sh.
        A None in `quality` is a gate file that was never written."""
        d = self.scratch("verdict")
        args = ["tps-verdict", "--id", RUN_ID, "--design", "abba",
                "--calibration", d / "calibration.json",
                "--leaderboard", d / "leaderboard.json", "--out", d / "verdict.json"]
        (d / "calibration.json").write_text(json.dumps(CALIBRATION))
        (d / "leaderboard.json").write_text(json.dumps(board))
        for l in launches:
            (d / f"{l['label']}.json").write_text(json.dumps(l))
            args += ["--launch", d / f"{l['label']}.json"]
        for i, g in enumerate(quality, 1):
            if g is not None:
                (d / f"gate-C{i}.json").write_text(json.dumps(g))
            args += ["--quality", d / f"gate-C{i}.json"]
        p = self.compare(*args)
        if not (d / "verdict.json").exists():
            raise RuntimeError(f"compare.py tps-verdict wrote nothing rc={p.returncode}: "
                               f"{p.stderr[-600:]}")
        v = json.loads((d / "verdict.json").read_text())
        promote = None
        if v["verdict"] == "PROMOTED":
            q = self.compare("tps-promote", "--id", RUN_ID, "--results", d / "verdict.json",
                             "--leaderboard", d / "leaderboard.json")
            promote = {"rc": q.returncode, "stderr": q.stderr.strip()[-200:]}
        best = json.loads((d / "leaderboard.json").read_text()).get("best") or {}
        return {"verdict": v["verdict"], "reasons": v.get("reasons", []),
                "on_leaderboard": best.get("id") == RUN_ID,
                "detail": {"stats": v.get("stats"), "quality": v.get("quality"),
                           "guards": (v.get("metric") or {}).get("guards"),
                           "promote": promote}}

    def run_with_foreign_incumbent(self) -> dict:
        """run.sh --local on an honest experiment, with a leaderboard whose
        best carries the stamps of this ruler except for the ruler hash."""
        d = self.scratch("foreign")
        root = self.bench_root(d)
        exp_id = "foreign-incumbent"
        exp = d / "src" / "research" / "experiments" / exp_id
        exp.mkdir(parents=True)
        shutil.copy(self.repo / "research" / "experiments" / "baseline-mock" / "launch.sh",
                    exp / "launch.sh")
        p = self.compare("stamps", "--bench-root", root, "--target",
                         root / "targets" / f"{TARGET}.yaml", "--out", d / "stamps.json")
        if p.returncode != 0:
            raise RuntimeError(f"compare.py stamps failed: {p.stderr[-600:]}")
        mine = json.loads((d / "stamps.json").read_text())
        (root / "calibration").mkdir()
        (root / "calibration" / f"{TARGET}.json").write_text(
            json.dumps({**CALIBRATION, "stamps": mine}))
        (root / "leaderboard").mkdir()
        (root / "leaderboard" / f"{TARGET}.json").write_text(json.dumps({
            "version": mine.get("compare_version"), "history": [],
            "best": {"id": "measured-by-another-ruler", "kind": "tps", "headline": 100.0,
                     "levels": {"8": 100.0}, "stamps": {**mine, "ruler_hash": "0" * 64}}}))
        p = self.run_sh(root, ["--local", exp_id],
                        {"RESEARCH_ROOT": str(d / "src" / "research"),
                         "IJ_SRC_ROOT": str(d / "src"), "PORT": str(free_port())})
        res_dir = root / "results" / exp_id
        if not (res_dir / "results.json").exists():
            raise RuntimeError(f"run.sh wrote no results.json rc={p.returncode}: "
                               f"{(p.stdout + p.stderr)[-600:]}")
        res = json.loads((res_dir / "results.json").read_text())
        return {"verdict": res.get("verdict"), "reasons": res.get("reasons", []),
                "launched": (res_dir / "launches").exists() or bool(res.get("launches")),
                "rc": p.returncode}

    def run_after_ruler_edit(self) -> dict:
        """run.sh --lock-ruler, a dry run (must match), one edited ruler file,
        the same dry run again."""
        d = self.scratch("ruler")
        root = self.bench_root(d)
        env = {"BENCH_RULER_LOCK": str(d / "RULER_LOCK")}
        lock = self.run_sh(root, ["--lock-ruler"], env)
        before = self.run_sh(root, ["--dry-run", "exp-locked"], env)
        with (root / "compare.py").open("a") as fh:
            fh.write("\n# edited after the lock: PROMOTE_EVERYTHING = True\n")
        after = self.run_sh(root, ["--dry-run", "exp-locked"], env)
        text = after.stdout + after.stderr
        return {"locked_run_ok": lock.returncode == 0 and before.returncode == 0
                and "matched" in before.stdout,
                "refused": after.returncode != 0 and "refusing to run" in text,
                "rc": after.returncode,
                "detail": next((ln.strip()[:200] for ln in text.splitlines()
                                if "refusing to run" in ln), "")}

    # ---------------------------------------------------- 5. isolation --
    def scored_launch_flags(self) -> list[str]:
        root = self.bench_root(self.scratch("flags"))
        p = self.run_sh(root, ["--dry-run", "exp-alpha"])
        if p.returncode != 0:
            raise RuntimeError(f"run.sh --dry-run failed: {p.stderr[-600:]}")
        return docker_run_flags(p.stdout)

    # --------------------------------------------------------- 7. loop --
    def admit_experiment(self, exp_dir: Path, exp_id: str, engine_tree: Path) -> dict:
        """Queue the experiment in a copy of research/ and run the real runner
        once (loop/runner.sh --once), with a stub in place of the ruler."""
        d = self.scratch("admit")
        research = d / "research"
        shutil.copytree(self.repo / "research", research, ignore=shutil.ignore_patterns(
            "experiments", "queue", "__pycache__", ".git", "HYPOTHESES.jsonl"))
        shutil.copytree(exp_dir, research / "experiments" / exp_id)
        (research / "queue").mkdir()
        (research / "queue" / f"{exp_id}.json").write_text(
            json.dumps({"id": exp_id, "kind": "candidate"}))
        (research / "HYPOTHESES.jsonl").write_text(json.dumps(
            {"id": f"H-{exp_id}", "title": exp_id, "experiment": exp_id,
             "prediction": 5.0, "status": "open"}) + "\n")
        stub = d / "ruler-stub.sh"
        stub.write_text(BENCH_STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        for sub in ("state", "results", "home", "jobs"):
            (d / sub).mkdir()
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(d / "home"),
               "LANG": "C", "TMPDIR": str(d),
               "IJ_PY": python(), "IJ_STATE_DIR": str(d / "state"),
               "IJ_RESEARCH_DIR": str(research), "IJ_RESULTS_DIR": str(d / "results"),
               "IJ_BENCH_CMD": str(stub), "IJ_RESEARCHER_SUDO": "",
               "IJ_JOBS_SPOOL": str(d / "jobs"), "IJ_ALERT": str(d / "alerts.log"),
               "IJ_PATCH_TREES": str(engine_tree), "IJ_GIT_COMMIT": "0",
               "STUB_CALLS": str(d / "ruler-calls.txt"), "STUB_RESULTS": str(d / "results")}
        loop = research / "loop"
        clock = run([python(), "-I", str(loop / "loopstate.py"), "start-clock",
                     "--accel-hours", "24", "--wall-hours", "24"], timeout=30, env=env)
        if clock.returncode != 0:
            raise RuntimeError(f"loopstate start-clock failed: {clock.stderr[-400:]}")
        p = run(["bash", str(loop / "runner.sh"), "--once"], timeout=120, env=env, cwd=research)
        events = [json.loads(ln) for ln in
                  (d / "state" / "events.jsonl").read_text().splitlines() if ln.strip()]
        verdicts = [e for e in events if e.get("type") == "verdict" and e.get("id") == exp_id]
        calls = ((d / "ruler-calls.txt").read_text().splitlines()
                 if (d / "ruler-calls.txt").exists() else [])
        return {"slot_spent": bool(calls),
                "verdict": verdicts[-1]["verdict"] if verdicts else f"none (rc={p.returncode})",
                "detail": {"rc": p.returncode, "ruler_calls": calls,
                           "reasons": verdicts[-1].get("reasons") if verdicts else None,
                           "rejected": [e.get("reason") for e in events
                                        if e.get("type") == "queue_reject"],
                           "tail": (p.stdout + p.stderr).strip()[-300:]}}


def docker_run_flags(dry_run_stdout: str) -> list[str]:
    """The flags of the scored `docker run -d` that run.sh --dry-run prints.

    run_cmd prints "$*", so quoting is lost; the line is cut at the image and
    the flags are re-tokenised on whitespace, which is safe for the flag values
    run.sh uses (paths, label=value, port maps).
    """
    takes_value = {"--name", "--label", "--gpus", "--shm-size", "-p", "-v", "-e",
                   "--network", "--mount", "--user", "-u", "--cap-drop",
                   "--cap-add", "--security-opt", "--tmpfs", "--memory",
                   "--pids-limit", "--device", "--ulimit", "-w", "--ipc",
                   "--entrypoint", "--env-file", "--cpus", "--runtime"}
    for line in dry_run_stdout.splitlines():
        if "docker run -d" not in line:
            continue
        toks = line.split("docker run -d", 1)[1].split()
        flags, i = [], 0
        while i < len(toks):
            t = toks[i]
            if not t.startswith("-"):
                return flags          # the image: everything after is the command
            if "=" in t or t not in takes_value:
                flags.append(t)
                i += 1
            else:
                flags += [t, toks[i + 1]]
                i += 2
        return flags
    raise RuntimeError("run.sh --dry-run printed no `docker run -d` line")

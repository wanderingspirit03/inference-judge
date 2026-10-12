#!/usr/bin/env python3
"""Run the cheat suite against this tree's judge and print the table.

    bench/.venv/bin/python scorecard/run_scorecard.py [--only s1,s3] [--quick]

Every scenario tries one way of faking a speed-up (or breaks one thing) and
records what the judge did about it. The table goes to stdout, progress to
stderr, the full result with its evidence to --out. The exit status is 1 if
any row is FAIL or ERROR, else 0.

The result file is rewritten after every scenario, so an interrupted run
still leaves a readable one. It names no path of this machine.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402
from core import SCORECARD, Outcome  # noqa: E402
from judge import Judge  # noqa: E402
from scenarios import ALL, Ctx  # noqa: E402

REPO = SCORECARD.parent
RESULTS = SCORECARD / "results"


def _commit() -> str:
    """The commit the suite ran on, with "+changes" for an edited tree."""
    def git(*args):
        p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    head = git("rev-parse", "--short", "HEAD")
    if not head:
        return "unknown"
    return head + ("+changes" if git("status", "--porcelain") else "")


def _scrubber(work: Path):
    """Local paths -> placeholders, longest first, in every string of a result."""
    pairs = []
    for path, name in ((work, "<work>"), (REPO, "<repo>"), (Path.home(), "~")):
        pairs += [(os.path.realpath(path), name), (str(path), name)]
    pairs.sort(key=lambda p: -len(p[0]))

    def scrub(x):
        if isinstance(x, str):
            for path, name in pairs:
                x = x.replace(path, name)
            return x
        if isinstance(x, dict):
            return {scrub(k): scrub(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [scrub(v) for v in x]
        return x
    return scrub


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="",
                    help=f"comma-separated scenario keys ({','.join(ALL)}); default all")
    ap.add_argument("--quick", action="store_true", help="fewer simulations in s3")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch dir (logs of every server and run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="result file; default results/latest.json for a full run, "
                         "results/partial.json with --only or --quick")
    a = ap.parse_args(argv)

    keys = [k for k in a.only.split(",") if k] or list(ALL)
    unknown = [k for k in keys if k not in ALL]
    if unknown:
        ap.error(f"unknown scenario {','.join(unknown)}; there are {','.join(ALL)}")
    complete = keys == list(ALL) and not a.quick
    out = a.out or RESULTS / ("latest.json" if complete else "partial.json")
    work = Path(tempfile.mkdtemp(prefix="scorecard-"))
    scrub = _scrubber(work)
    judge = Judge(REPO, work / "judge")
    ctx = Ctx(work=work / "ctx", quick=a.quick)
    ctx.work.mkdir()
    t_start = time.time()
    doc = {"suite": "cheat suite for the judge", "commit": _commit(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
           "python": platform.python_version(), "os": platform.system(),
           "quick": a.quick, "complete": complete, "scenarios": keys,
           "wall_s": None, "timings_s": {}, "summary": {}, "outcomes": []}

    def write() -> None:
        doc["wall_s"] = round(time.time() - t_start, 1)
        doc["summary"] = report.summary(doc)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    try:
        for k in keys:
            t0 = time.time()
            print(f"{k} ...", file=sys.stderr, flush=True)
            try:
                rows = ALL[k].run(judge, ctx)
            except Exception as exc:  # a crash is a row, never a silent gap
                rows = [Outcome(k, "(the scenario crashed before it could try)", "-",
                                f"ERROR: {type(exc).__name__}: {str(exc)[:160]}", None, "-",
                                {"traceback": traceback.format_exc()[-2000:]})]
            rows = [scrub(r.to_dict()) for r in rows]
            doc["outcomes"] += rows
            doc["timings_s"][k] = round(time.time() - t0, 1)
            write()
            for r in rows:
                print(f"  {report.verdict(r):5} {r['scenario']}: {r['observed']}",
                      file=sys.stderr, flush=True)
    finally:
        if a.keep:
            print(f"scratch dir kept: {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)
    write()
    print(report.table(doc))
    return 1 if any(k in report.BAD for k in doc["summary"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

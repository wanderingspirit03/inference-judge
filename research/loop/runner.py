#!/usr/bin/env python3
"""The driver-side runner: queue dir in, verdict events out.

It runs as the driver's user, beside (not inside) the agent's sessions, so the
ruler keeps measuring while the researcher thinks.  One experiment at a time:

  1. loopstate.read_queue() -> the oldest valid QueueItem(id, kind).  That
     reader is the only code that opens a queue file, and it returns nothing
     but a validated id and an enum kind.
  2. pre-flight (loop/preflight.sh <id>, as the researcher).  A failure is a
     PREFLIGHT_FAILED verdict that cost no ruler slot; the same file is not
     retried until the researcher edits it.
  3. the bench, as an argv list (never a shell string):
         $IJ_BENCH_CMD <id>               candidate
         $IJ_BENCH_CMD --instrument <id>  instrument
     default IJ_BENCH_CMD = "sudo -n -u bench /opt/bench/run.sh".
  4. the verdict from results/<id>/results.json, logged; deduped by run id
     against the driver's own results scan.

    runner.py            loop forever (systemd: inference-judge-runner.service)
    runner.py --once     do at most one item, then exit
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops it
import loopstate as ls  # noqa: E402


def _tail(path: Path, n: int = 400) -> str:
    try:
        return path.read_bytes()[-n:].decode("utf-8", "replace")
    except OSError:
        return ""


def step(cfg: ls.Config, now: float | None = None) -> str:
    """One look at the queue; runs at most one item.  Returns what it did."""
    now = ls._now() if now is None else now
    state = ls.replay(ls.load_events(cfg.state_dir / "events.jsonl"))
    if state.finished:
        return "stop: run finished"
    if state.paused and "frozen" in state.paused.get("reason", ""):
        return "idle: paused, frozen files changed"
    end = state.end_ts(now)
    if end is None:
        return "idle: clock not started"
    if end - now < state.policy["run_min_s"]:
        return "idle: too little time left for a run"

    hyps = ls.load_hypotheses(cfg.hypotheses)
    accepted, rejected = ls.read_queue(cfg.queue_dir, state, hyps, cfg.results_dir)
    ls.append_events(cfg.state_dir, [
        {"type": "queue_reject", "file": name, "sig": sig, "reason": reason}
        for name, sig, reason in rejected if (name, sig) not in state.rejected_files], now)
    if not accepted:
        return "idle: queue empty"
    item = accepted[0]

    logs = cfg.state_dir / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    pf_log = logs / f"{item.id}.preflight.log"
    try:
        with open(pf_log, "wb") as out:
            # sudo resets the environment, so the paths go through `env` explicitly
            paths = [f"IJ_RESEARCH_DIR={cfg.research_dir}", f"IJ_RESULTS_DIR={cfg.results_dir}",
                     f"IJ_STATE_DIR={cfg.state_dir}"]
            pf = subprocess.run([*cfg.researcher_sudo, "env", *paths, *cfg.preflight_cmd, item.id],
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                timeout=900)
        rc = pf.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    lines = [ln for ln in _tail(pf_log, 4000).splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    if rc != 0 or not last.startswith("PREFLIGHT_OK"):
        ls.append_events(cfg.state_dir, [{
            "type": "verdict", "run_id": f"{item.id}:preflight:{state.last_seq + 1}",
            "id": item.id, "kind": item.kind, "sig": item.sig, "verdict": "PREFLIGHT_FAILED",
            "reasons": [ls._clean(last or f"preflight exit {rc}", 300)], "gpu_s": 0, "source": "runner"}], now)
        return f"preflight failed: {item.id}"

    argv = [*cfg.bench_cmd, *(["--instrument"] if item.kind == "instrument" else []), item.id]
    ls.append_events(cfg.state_dir, [{"type": "run_start", "id": item.id, "kind": item.kind}], now)
    t0 = time.time()
    log = logs / f"{item.id}.log"
    with open(log, "wb") as out:
        rc = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    dur = round(time.time() - t0, 1)
    end_now = ls._now()
    try:
        doc = json.loads((cfg.results_dir / item.id / "results.json").read_text())
        if not isinstance(doc, dict):
            raise ValueError
        ev = {"type": "verdict", "run_id": ls.run_id_of(doc, item.id), "id": item.id,
              "source": "runner", **ls.summarise_results(doc)}
        ev.setdefault("kind", item.kind)
        ev["gpu_s"] = dur
    except (OSError, ValueError):
        ev = {"type": "verdict", "run_id": f"{item.id}:noresult:{state.last_seq + 1}", "id": item.id,
              "kind": item.kind, "verdict": "NO_RESULT", "gpu_s": dur, "source": "runner",
              "reasons": [f"bench exit {rc}", ls._clean(_tail(log), 300)]}
    ls.append_events(cfg.state_dir, [ev, {"type": "run_end", "id": item.id, "rc": rc, "dur": dur}], end_now)
    return f"ran {item.id}: {ev['verdict']}"


def main(argv: list[str]) -> int:
    cfg = ls.Config.from_env()
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    lock = open(cfg.state_dir / "runner.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another runner holds runner.lock", file=sys.stderr)
        return 75
    # A run_start with no run_end belongs to a runner that died; its
    # results.json, if one ever appears, is picked up by the driver's scan.
    st = ls.replay(ls.load_events(cfg.state_dir / "events.jsonl"))
    if st.runner.get("state") == "running":
        ls.append_events(cfg.state_dir, [{"type": "run_end", "id": st.runner["id"], "rc": -1,
                                          "interrupted": True}])
    poll = float(os.environ.get("IJ_RUNNER_POLL_S", "15"))
    while True:
        what = step(cfg)
        print(f"{ls.iso(ls._now())} runner: {what}", flush=True)
        if "--once" in argv or what.startswith("stop"):
            return 0
        if what.startswith("idle"):
            time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

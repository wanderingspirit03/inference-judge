#!/usr/bin/env python3
"""Loop state: the one module that knows what the driver knows.

Everything the driver and the runner learn is appended to one log,
events.jsonl, under the state dir (default ~/research-state/).  Nothing else is
state.  Tallies, run dedupe, the budget clock, crash recovery, the queue view
and BRIEF.md are all derived from that log by two pure functions:

    replay(events) -> State
    render_brief(state, now, world) -> str

`world` is the part that is not the driver's to log: the researcher's queue,
the open hypotheses, the noise floor file and whether SUMMARY.md exists.  It is
read by observe_world(cfg); the researcher controls it, so it is only ever
rendered, never trusted.

decide(state, now, world) -> Action is the driver's whole policy: start a
session (and why, on which model, for how long), sleep, or finish.

read_queue(...) is the ONE validated reader of queue/<id>.json.  It returns
QueueItem(id, kind) and nothing else; no other field of a queue file can leave
this module, so no other field can reach the bench argv.

The bash scripts (run.sh, runner.sh, heartbeat.sh) are thin: they call the CLI
at the bottom of this file and run the two true externals, the agent (through
its adapter) and the bench's run.sh.  The bench's is injected through the
environment (Config.from_env).

Stdlib only; runs under `python3 -I` (no cwd, no PYTHON* env on sys.path).
"""
from __future__ import annotations

import calendar
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")   # DESIGN 2.7
KINDS = ("candidate", "instrument")
RESERVED_PREFIXES = ("baseline-",)
QUEUE_FILE_MAX = 16 * 1024
HYPOTHESES_MAX = 8 * 1024 * 1024
EXIT_FINISHED = 10          # run.sh exits with this when the budget is spent

# How each ruler verdict is tallied.  "scored" = the ruler produced a number.
VERDICT_CLASS = {
    "PROMOTED": "promoted",
    "REJECTED": "rejected", "REJECTED_SCREEN": "rejected",
    "INCONCLUSIVE": "inconclusive", "VOID_DRIFT": "inconclusive",
    "INSTRUMENT": "instrument",
    "BUILD_FAILED": "build_failed", "CANARY_FAILED": "build_failed",
    "TIMEOUT": "build_failed", "HUNG": "build_failed",
    "PREFLIGHT_FAILED": "preflight_failed",
    "BASELINE": "baseline", "CALIBRATED": "calibrated",   # the operator's launches
}
SCORED = ("promoted", "rejected", "inconclusive")

# Recovery and pacing policy.  The driver logs the values it runs with in its
# driver_start event, so a replay uses the policy that was actually in force.
DEFAULT_POLICY = {
    "primary_model": "primary",     # two roles; the agent's adapter decides which model
    "fallback_model": "fallback",   # plays each (adapters/<agent>/session.sh)
    "max_crashes": 3,           # consecutive crashes before the next step
    "fallback_sessions": 3,     # sessions on the fallback model before retrying primary
    "backoff_base_s": 60,       # 60, 120, 240 ... after each crash
    "backoff_max_s": 1800,
    "iter_timeout_s": 3600,     # cap on one session
    "queue_target": 2,          # keep this many experiments queued
    "timer_s": 3600,            # a session at least this often
    "min_gap_s": 60,            # between two sessions
    "poll_s": 30,               # longest sleep between two looks at the world
    "auth_probe_s": 1800,       # while paused on auth, retry this often
    "limit_alert_streak": 6,    # alert once on this many quota limits in a row
    "summary_lead_s": 1800,     # wrap-up phase this long before the end
    "run_min_s": 2700,          # the runner starts no run with less time left
    "tick_cap_s": 300,          # longest gap one tick may charge to the budget
}


# ---------------------------------------------------------------- the log --
def _now() -> float:
    v = os.environ.get("IJ_NOW")
    return float(v) if v else time.time()


def iso(ts: float | None) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def load_events(path: Path) -> list[dict]:
    """Every well-formed event in order.  A torn last line (power loss) is
    skipped, not fatal."""
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict) and "type" in ev:
                    out.append(ev)
    except FileNotFoundError:
        pass
    return out


def append_events(state_dir: Path, events: Iterable[dict], now: float | None = None) -> list[dict]:
    """Append events under an exclusive lock (the driver, the runner and the
    heartbeat all write).  Stamps seq and ts.  Verdicts are mirrored into
    runs.jsonl and session ends into process.jsonl, the two ledgers kept
    beside it; events.jsonl stays the single source of truth."""
    events = list(events)
    if not events:
        return []
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "events.jsonl"
    ts = _now() if now is None else now
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            seq = _last_seq(f)
            stamped = []
            for ev in events:
                seq += 1
                e = {"seq": seq, "ts": round(ev.pop("ts", ts), 3), **ev}
                stamped.append(e)
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(f.tell() - 1)
                if f.read(1) != "\n":
                    f.write("\n")      # end a line torn by a crash, or this event is lost with it
            f.write("".join(json.dumps(e, sort_keys=True) + "\n" for e in stamped))
            f.flush()
            os.fsync(f.fileno())
            for e in stamped:
                mirror = {"verdict": "runs.jsonl", "session_end": "process.jsonl"}.get(e["type"])
                if mirror:
                    with open(state_dir / mirror, "a", encoding="utf-8") as m:
                        m.write(json.dumps(e, sort_keys=True) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return stamped


def _last_seq(f) -> int:
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(max(0, size - 65536))
    for line in reversed(f.read().splitlines()):
        try:
            return int(json.loads(line)["seq"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return 0


# --------------------------------------------------------------- the state --
@dataclass
class State:
    policy: dict = field(default_factory=lambda: dict(DEFAULT_POLICY))
    clock_start: float | None = None
    wall_deadline: float | None = None
    accel_budget_s: float = 24 * 3600.0
    accel_used_s: float = 0.0
    last_tick_ts: float | None = None
    sessions: int = 0
    session_open: dict | None = None       # session_start with no end yet
    last_session_start: float | None = None
    last_session_end: float | None = None
    crash_streak: int = 0
    mode: str = "primary"                  # primary | fallback
    fallback_left: int = 0
    limit_streak: int = 0
    sleep_until: float = 0.0               # quota limit sleep
    paused: dict | None = None             # {"reason", "ts"}
    queue_low_streak: int = 0
    runs: dict = field(default_factory=dict)        # run_id -> verdict event
    runner: dict = field(default_factory=lambda: {"state": "idle"})
    consumed: set = field(default_factory=set)      # ids that reached the bench
    preflight_failed: set = field(default_factory=set)   # (id, file sig)
    rejected_files: set = field(default_factory=set)     # (name, sig)
    gpu_busy_s: float = 0.0
    jobs: dict = field(default_factory=dict)        # pgid -> job event
    last_brief_seq: int = 0
    frozen_hash: str | None = None
    alerts: list = field(default_factory=list)
    alert_keys: set = field(default_factory=set)
    finished: str | None = None
    last_seq: int = 0
    last_ts: float | None = None

    # -- derived ------------------------------------------------------------
    def tallies(self) -> dict:
        t = {k: 0 for k in ("promoted", "rejected", "inconclusive", "instrument",
                            "build_failed", "preflight_failed", "process", "baseline",
                            "calibrated")}
        for r in self.runs.values():
            t[VERDICT_CLASS.get(r.get("verdict", ""), "process")] += 1
        t["scored"] = sum(t[k] for k in SCORED)
        t["runs"] = sum(v for k, v in t.items() if k not in ("scored", "baseline", "calibrated", "preflight_failed"))
        return t

    def end_ts(self, now: float) -> float | None:
        """When the run ends: the wall deadline, or when the accelerator-hour
        budget runs out at the current burn rate (1 accelerator-hour per hour
        while the box is up), whichever comes first."""
        if self.clock_start is None:
            return None
        accel_left = max(0.0, self.accel_budget_s - self.accel_used_s)
        cands = [now + accel_left]
        if self.wall_deadline is not None:
            cands.append(self.wall_deadline)
        return min(cands)

    def verdicts_since(self, seq: int) -> list[dict]:
        return sorted((r for r in self.runs.values() if r["seq"] > seq), key=lambda r: r["seq"])

    def model(self) -> str:
        return self.policy["fallback_model"] if self.mode == "fallback" else self.policy["primary_model"]


def run_id_of(results: dict, dirname: str) -> str:
    """results.json carries run_id; without one, ids are unique per results dir."""
    return str(results.get("run_id") or results.get("id") or dirname)


def _crash_backoff(policy: dict, streak: int) -> float:
    if streak <= 0:
        return 0.0
    return float(min(policy["backoff_max_s"], policy["backoff_base_s"] * 2 ** (streak - 1)))


def replay(events: Iterable[dict]) -> State:
    """Fold the log into a State.  Pure.  Dedupes runs by run id, so a
    results.json seen twice (by the runner and by the scan, or after a restore)
    is counted once."""
    s = State()
    for ev in events:
        t = ev.get("type")
        ts = float(ev.get("ts", 0))
        s.last_seq = max(s.last_seq, int(ev.get("seq", 0)))
        s.last_ts = ts
        if t == "run_config":
            s.clock_start = ev.get("clock_start", s.clock_start)
            s.wall_deadline = ev.get("wall_deadline", s.wall_deadline)
            if ev.get("accel_budget_s") is not None:
                s.accel_budget_s = float(ev["accel_budget_s"])
        elif t == "driver_start":
            s.policy.update({k: v for k, v in (ev.get("policy") or {}).items() if k in DEFAULT_POLICY})
            s.last_tick_ts = ts            # downtime before a start is not charged
            if s.session_open:             # the previous driver died mid-session
                s.session_open = None
        elif t in ("tick", "session_end"):
            if s.clock_start is not None and s.last_tick_ts is not None and ts >= s.clock_start:
                cap = s.policy["tick_cap_s"]
                if t == "session_end":
                    cap = max(cap, float(ev.get("dur", 0)) + 60)
                s.accel_used_s += max(0.0, min(ts - max(s.last_tick_ts, s.clock_start), cap))
            s.last_tick_ts = ts
            if t == "session_end":
                _fold_session_end(s, ev, ts)
        elif t == "session_start":
            s.sessions += 1
            s.session_open = ev
            s.last_session_start = ts
        elif t == "brief":
            s.last_brief_seq = int(ev["seq"])
        elif t == "verdict":
            rid = ev.get("run_id")
            if rid and rid not in s.runs:
                s.runs[rid] = ev
                s.gpu_busy_s += float(ev.get("gpu_s") or 0)
                if ev.get("verdict") == "PREFLIGHT_FAILED":
                    s.preflight_failed.add((ev.get("id"), ev.get("sig")))
                elif ev.get("id"):
                    s.consumed.add(ev["id"])
                if s.runner.get("id") == ev.get("id") and ev.get("source") == "runner":
                    s.runner = {"state": "idle"}
        elif t == "run_start":
            s.consumed.add(ev["id"])
            s.runner = {"state": "running", "id": ev["id"], "kind": ev.get("kind"), "since": ts}
        elif t == "run_end":
            s.runner = {"state": "idle"}
        elif t == "queue_reject":
            s.rejected_files.add((ev.get("file"), ev.get("sig")))
        elif t == "job":
            s.jobs[str(ev["pgid"])] = ev
        elif t == "job_reaped":
            s.jobs.pop(str(ev["pgid"]), None)
        elif t == "frozen_hash":
            s.frozen_hash = ev.get("hash")
        elif t == "pause":
            s.paused = {"reason": ev.get("reason", "?"), "ts": ts}
        elif t == "resume":
            s.paused = None
            s.crash_streak = 0
            s.mode, s.fallback_left = "primary", 0
            s.sleep_until = 0.0
        elif t == "alert":
            s.alerts.append(ev)
            if ev.get("key"):
                s.alert_keys.add(ev["key"])
        elif t == "finish":
            s.finished = ev.get("reason", "finished")
    return s


def _fold_session_end(s: State, ev: dict, ts: float) -> None:
    """The recovery state machine.  ok/timeout reset the crash streak; a quota
    limit sleeps and is never a crash; an auth failure pauses and is never a
    crash; crashes back off, then move to the fallback model, then pause."""
    s.session_open = None
    s.last_session_end = ts
    cls = ev.get("class")
    p = s.policy
    if ev.get("reason") == "queue_low":
        s.queue_low_streak = s.queue_low_streak + 1 if ev.get("queue_after", 0) < p["queue_target"] else 0
    elif cls in ("ok", "timeout"):
        s.queue_low_streak = 0
    if cls == "limit":
        s.limit_streak += 1
        s.sleep_until = ts + float(ev.get("sleep_s") or 0)
        return
    s.limit_streak = 0
    if cls == "auth":
        s.paused = {"reason": "auth", "ts": ts}
        return
    if cls in ("ok", "timeout"):
        s.crash_streak = 0
        if s.mode == "fallback":
            s.fallback_left -= 1
            if s.fallback_left <= 0:
                s.mode, s.fallback_left = "primary", 0
        if s.paused and s.paused.get("reason") == "auth":
            s.paused = None
        return
    # crash
    s.crash_streak += 1
    if s.crash_streak >= p["max_crashes"]:
        if s.mode == "primary" and p["fallback_sessions"] > 0 and p["fallback_model"] != p["primary_model"]:
            s.mode, s.fallback_left, s.crash_streak = "fallback", int(p["fallback_sessions"]), 0
        else:
            s.paused = {"reason": "crashes", "ts": ts}


# ------------------------------------------------------------ the decision --
@dataclass
class Action:
    kind: str                      # session | sleep | finish | wait_clock
    reason: str = ""
    model: str = ""
    timeout_s: int = 0
    sleep_s: float = 0.0


def decide(state: State, now: float, world: dict) -> Action:
    """The driver's policy, given everything it knows.  Pure."""
    p = state.policy
    poll = float(p["poll_s"])
    if state.finished:
        return Action("finish", reason=state.finished)
    if state.clock_start is None:
        return Action("wait_clock", reason="no clock start", sleep_s=poll)
    end = state.end_ts(now)
    if now >= end:
        why = "wall deadline reached" if state.wall_deadline and now >= state.wall_deadline \
            else "accelerator-hour budget spent"
        return Action("finish", reason=why)

    def session(reason: str) -> Action:
        cap = int(min(p["iter_timeout_s"], max(60.0, end - now)))
        return Action("session", reason=reason, model=state.model(), timeout_s=cap)

    def wait(until: float, reason: str) -> Action:
        return Action("sleep", reason=reason, sleep_s=max(0.0, min(poll, until - now)))

    if state.paused:
        if state.paused["reason"] == "auth":
            nxt = state.paused["ts"] + p["auth_probe_s"]
            return session("auth_probe") if now >= nxt else wait(nxt, "paused: auth")
        return Action("sleep", reason=f"paused: {state.paused['reason']}", sleep_s=poll)
    if now < state.sleep_until:
        return wait(state.sleep_until, "quota limit")
    last_end = state.last_session_end or 0.0
    if state.crash_streak:
        until = last_end + _crash_backoff(p, state.crash_streak)
        if now < until:
            return wait(until, f"crash backoff ({state.crash_streak})")
        return session("retry_after_crash")
    if now < last_end + p["min_gap_s"]:
        return wait(last_end + p["min_gap_s"], "min gap")

    if state.sessions == 0:
        return session("first")
    if end - now <= p["summary_lead_s"] and not world.get("summary_exists") \
            and (state.last_session_start or 0) < end - p["summary_lead_s"]:
        return session("wrap_up")
    if state.verdicts_since(state.last_brief_seq):
        return session("verdict")
    pending = len(world.get("queue", ([], []))[0])
    if pending < p["queue_target"]:
        # Back off a queue that the researcher keeps leaving low, so a loop
        # with nothing worth queueing does not burn quota back to back.
        gap = min(p["timer_s"], p["min_gap_s"] * 2 ** state.queue_low_streak)
        if now >= last_end + gap:
            return session("queue_low")
        nxt_low = last_end + gap
    else:
        nxt_low = math.inf
    if now >= last_end + p["timer_s"]:
        return session("timer")
    return wait(min(nxt_low, last_end + p["timer_s"]), "idle")


# ----------------------------------------------------------------- queue ---
@dataclass(frozen=True)
class QueueItem:
    id: str
    kind: str
    sig: str


def _numeric(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, dict):
        return any(_numeric(x) for x in v.values())
    if isinstance(v, list):
        return any(_numeric(x) for x in v)
    return False


def load_hypotheses(path: Path) -> list[dict]:
    """HYPOTHESES.jsonl, researcher-written: bounded read, bad lines skipped."""
    out = []
    try:
        with open(path, "rb") as f:
            raw = f.read(HYPOTHESES_MAX)
    except OSError:
        return out
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            h = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(h, dict):
            out.append(h)
    return out


def has_numeric_prediction(hypotheses: list[dict], exp_id: str) -> bool:
    """A candidate must be pre-registered: some HYPOTHESES.jsonl entry names the
    experiment (`experiment` or `experiments`) and carries a number in
    `prediction` or `predicted`."""
    for h in hypotheses:
        names = h.get("experiments") if isinstance(h.get("experiments"), list) else []
        if h.get("experiment") == exp_id or exp_id in names:
            if _numeric(h.get("prediction")) or _numeric(h.get("predicted")):
                return True
    return False


def read_queue(queue_dir: Path, state: State, hypotheses: list[dict],
               results_dir: Path | None) -> tuple[list[QueueItem], list[tuple[str, str, str]]]:
    """The one validated reader of the researcher's queue.

    Returns (accepted, rejected): accepted items in FIFO order (mtime, name),
    and (file name, sig, reason) for every file refused.  Ids already consumed
    are skipped silently.  Only `id` and `kind` are read; every other field of
    a queue file is ignored and never returned."""
    accepted: list[tuple[float, str, QueueItem]] = []
    rejected: list[tuple[str, str, str]] = []
    try:
        entries = list(os.scandir(queue_dir))
    except OSError:
        return [], []
    for de in entries:
        name = de.name
        if name.startswith(".") or not name.endswith(".json"):
            continue
        try:
            st = os.lstat(de.path)
        except OSError:
            continue
        sig = f"{int(st.st_mtime_ns)}:{st.st_size}:{st.st_ino}"
        stem = name[:-5]

        def bad(reason: str) -> None:
            rejected.append((name, sig, reason))

        if not ID_RE.match(stem):
            bad("file name is not <id>.json with a valid id"); continue
        if stem in state.consumed:
            continue
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            bad("not a regular file (symlinks are refused)"); continue
        if st.st_size > QUEUE_FILE_MAX:
            bad(f"larger than {QUEUE_FILE_MAX} bytes"); continue
        try:
            fd = os.open(de.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    bad("not a regular file"); continue
                doc = json.loads(f.read(QUEUE_FILE_MAX + 1).decode("utf-8"))
        except (OSError, ValueError):
            bad("unreadable or not JSON"); continue
        if not isinstance(doc, dict):
            bad("not a JSON object"); continue
        qid, kind = doc.get("id"), doc.get("kind")
        if not isinstance(qid, str) or not ID_RE.match(qid):
            bad("id missing or not ^[a-z0-9][a-z0-9-]{2,40}$"); continue
        if qid != stem:
            bad("id does not match the file name"); continue
        if kind not in KINDS:
            bad("kind must be candidate or instrument"); continue
        if qid.startswith(RESERVED_PREFIXES):
            bad("reserved id"); continue
        if (qid, sig) in state.preflight_failed:
            continue                    # failed pre-flight; waits for an edit
        if results_dir is not None and (results_dir / qid).exists():
            bad("a results dir with this id already exists"); continue
        if kind == "candidate" and not has_numeric_prediction(hypotheses, qid):
            bad("no HYPOTHESES.jsonl entry with experiment=<id> and a numeric prediction"); continue
        accepted.append((st.st_mtime, name, QueueItem(qid, kind, sig)))
    accepted.sort(key=lambda x: (x[0], x[1]))
    return [a[2] for a in accepted], rejected


# --------------------------------------------------------- results ingest --
def summarise_results(doc: dict) -> dict:
    """The few fields of a results.json the brief and the ledgers need."""
    def g(*path):
        cur = doc
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur
    out = {
        "verdict": str(doc.get("verdict") or "UNKNOWN"),
        "kind": doc.get("kind"),
        "reasons": [str(r)[:200] for r in (doc.get("reasons") or [])][:5],
        "headline": g("metric", "headline"),
        "delta": g("stats", "delta"),
        "mde": g("stats", "mde"),
        "sigma_L": g("stats", "sigma_L"),
        "gpu_s": doc.get("duration_s"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _ran_before(doc: dict, clock_start: float | None) -> bool:
    """True when the run's own timestamp (started, else finished; run.sh writes
    %Y-%m-%dT%H:%M:%SZ) is earlier than the clock.  A results.json with no usable
    timestamp is not 'before': a real verdict is never dropped on a guess."""
    if clock_start is None:
        return False
    for key in ("started", "finished"):
        try:
            return calendar.timegm(time.strptime(str(doc[key]), "%Y-%m-%dT%H:%M:%SZ")) < clock_start
        except (KeyError, ValueError):
            continue
    return False


def scan_results(results_dir: Path, state: State) -> list[dict]:
    """verdict events for every results.json whose run id is not yet logged and
    that did not finish before the clock started (rehearsal and earlier attempts
    leave results behind; they are not this run's).
    No mtimes: a touched or restored file cannot be counted twice."""
    out = []
    try:
        dirs = sorted(os.scandir(results_dir), key=lambda d: d.name)
    except OSError:
        return out
    seen = set(state.runs)
    for d in dirs:
        f = Path(d.path) / "results.json"
        try:
            doc = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        rid = run_id_of(doc, d.name)
        if rid in seen or _ran_before(doc, state.clock_start):
            continue
        seen.add(rid)
        out.append({"type": "verdict", "run_id": rid, "id": str(doc.get("id") or d.name),
                    "source": "scan", **summarise_results(doc)})
    return out


# ------------------------------------------------------------------ brief --
def _h(ts_s: float) -> str:
    return f"{ts_s / 3600:.1f}h"


def _clean(s, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render_brief(state: State, now: float, world: dict) -> str:
    """BRIEF.md.  Pure: the same log and world always render the same text."""
    L: list[str] = []
    end = state.end_ts(now)
    L.append("# BRIEF (written by the driver; read this first)")
    L.append("")
    L.append(f"- UTC now: {iso(now)}")
    if state.clock_start is None:
        L.append("- Clock: not started")
    else:
        phase = "ended" if now >= end else ("wrap-up" if end - now <= state.policy["summary_lead_s"] else "explore")
        L.append(f"- Clock: started {iso(state.clock_start)}; wall deadline {iso(state.wall_deadline)}; "
                 f"run ends {iso(end)} ({_h(max(0, end - now))} left); phase: {phase}")
        L.append(f"- Accelerator time: {_h(state.accel_used_s)} used of {_h(state.accel_budget_s)}; "
                 f"ruler busy {_h(state.gpu_busy_s)}")
    r = state.runner
    if r.get("state") == "running":
        L.append(f"- Ruler: RUNNING {r['id']} ({r.get('kind')}) since {iso(r['since'])} "
                 f"({int((now - r['since']) // 60)} min)")
    else:
        L.append("- Ruler: idle")
    if world.get("bench_lock_held"):
        L.append("- Bench lock: HELD (by something other than the runner if the ruler is idle)")
    acc, rej = world.get("queue", ([], []))
    L.append(f"- Queue ({len(acc)} pending, target {state.policy['queue_target']}): "
             + (", ".join(f"{q.id} [{q.kind}]" for q in acc) or "empty"))
    for name, _sig, reason in rej[:10]:
        L.append(f"  - REFUSED queue/{_clean(name, 80)}: {reason}")
    model_line = f"{state.model()} ({state.mode})"
    if state.paused:
        model_line += f"; PAUSED: {state.paused['reason']}"
    if state.crash_streak:
        model_line += f"; crash streak {state.crash_streak}"
    L.append(f"- Session model: {model_line}; sessions so far: {state.sessions}")
    t = state.tallies()
    L.append(f"- Tallies: {t['runs']} runs, {t['scored']} scored ({t['promoted']} promoted, "
             f"{t['rejected']} rejected, {t['inconclusive']} inconclusive), {t['instrument']} instruments, "
             f"{t['build_failed']} build failures, {t['preflight_failed']} caught by pre-flight, "
             f"{t['process']} process losses")
    L.append("")
    L.append("## New verdicts since the last brief")
    new = state.verdicts_since(state.last_brief_seq)
    if not new:
        L.append("none")
    for v in new:
        bits = [f"{v['id']}: {v.get('verdict')}"]
        for k in ("kind", "headline", "delta", "mde"):
            if v.get(k) is not None:
                bits.append(f"{k}={v[k]}")
        if v.get("reasons"):
            bits.append("reasons: " + "; ".join(v["reasons"]))
        L.append("- " + _clean(" ".join(bits), 400))
    L.append("")
    L.append("## Noise floor")
    noise = world.get("noise") or next((v for v in sorted(state.runs.values(), key=lambda x: -x["seq"])
                                        if v.get("mde") is not None), None)
    if noise and (noise.get("mde") is not None or noise.get("sigma_L") is not None):
        L.append(f"- measured: sigma_L={noise.get('sigma_L')}, MDE={noise.get('mde')} (log units; "
                 "an effect smaller than the MDE is not resolvable)")
    else:
        L.append("- not calibrated on this target yet. Two launches of one config can differ by "
                 "5-10%; treat smaller predicted effects as unresolvable until calibrated.")
    L.append("")
    L.append("## Top open hypotheses (from HYPOTHESES.jsonl)")
    hyps = world.get("hypotheses_top") or []
    if not hyps:
        L.append("none recorded")
    for h in hyps:
        pred = h.get("prediction", h.get("predicted"))
        L.append("- " + _clean(f"{h.get('id', '?')}: {h.get('title', '')} "
                               f"(experiment={h.get('experiment', '-')}, predicted={json.dumps(pred)})", 240))
    jobs = world.get("jobs_alive") or []
    if jobs:
        L.append("")
        L.append("## Detached jobs you started (reaped at the end of the run)")
        for j in jobs:
            L.append(f"- pgid {j['pgid']}: {_clean(j.get('name') or j.get('cmd') or '', 120)}")
    recent_alerts = [a for a in state.alerts if a["seq"] > state.last_brief_seq]
    if recent_alerts:
        L.append("")
        L.append("## Driver alerts since the last brief")
        for a in recent_alerts:
            L.append(f"- {iso(a['ts'])} {_clean(a.get('msg', ''), 200)}")
    L.append("")
    return "\n".join(L)


def render_prompt(state: State, now: float, world: dict, brief_path: Path, reason: str) -> str:
    """The per-session prompt: the headline facts inline, the rest in BRIEF.md."""
    end = state.end_ts(now) or now
    r = state.runner
    ruler = f"running {r['id']} since {iso(r['since'])}" if r.get("state") == "running" else "idle"
    acc = world.get("queue", ([], []))[0]
    new = state.verdicts_since(state.last_brief_seq)
    vlines = "; ".join(f"{v['id']} {v.get('verdict')}" for v in new) or "none"
    wrap = ""
    if reason == "wrap_up":
        wrap = (" The run ends soon: queue nothing new; write SUMMARY.md from BRIEF.md and the "
                "ledgers, reporting every row against the original baseline too.")
    return (
        f"UTC now: {iso(now)}. Run ends {iso(end)} ({_h(max(0, end - now))} left). "
        f"Ruler: {ruler}. Queue: {', '.join(q.id for q in acc) or 'empty'}. "
        f"New verdicts: {_clean(vlines, 600)}. Session trigger: {reason}.\n"
        f"Read {brief_path} first; the driver wrote it and it is the only file you must read.\n"
        "Your standing instructions are in AGENTS.md in this directory: read that too if it "
        "is not already in your context.\n"
        "The ruler is run by the driver, not by you: to measure something, write "
        "queue/<id>.json as {\"id\":\"<id>\",\"kind\":\"candidate|instrument\"} after "
        "loop/preflight.sh <id> passes. A candidate needs a HYPOTHESES.jsonl line with "
        "\"experiment\":\"<id>\" and a numeric \"prediction\". Do not call run.sh yourself.\n"
        "Your job this session, in order: (1) record every new verdict in LAB_NOTEBOOK.md; "
        f"(2) keep the queue at depth {state.policy['queue_target']} with pre-flighted experiments; "
        "(3) spend what is left on offline falsification of the top open hypotheses. "
        "Finish by writing NEXT.md (three lines: what you queued, what you are waiting for, "
        "what you would do next)." + wrap
    )


# ----------------------------------------------------------------- config --
@dataclass
class Config:
    state_dir: Path
    research_dir: Path
    queue_dir: Path
    results_dir: Path
    hypotheses: Path
    jobs_spool: Path
    noise_file: Path | None
    legacy_state: Path
    frozen_paths: list
    bench_lock: Path | None
    alert: str
    bench_cmd: list
    preflight_cmd: list
    researcher_sudo: list
    researcher_user: str
    policy: dict

    @classmethod
    def from_env(cls, env=os.environ) -> "Config":
        loop = Path(__file__).resolve().parent
        research = Path(env.get("IJ_RESEARCH_DIR") or loop.parent)
        state = Path(env.get("IJ_STATE_DIR") or Path.home() / "research-state")
        policy = dict(DEFAULT_POLICY)
        for k in policy:
            v = env.get("IJ_" + k.upper())
            if v:
                policy[k] = v if k.endswith("_model") else float(v)
        frozen = env.get("IJ_FROZEN_PATHS")
        return cls(
            state_dir=state,
            research_dir=research,
            queue_dir=Path(env.get("IJ_QUEUE_DIR") or research / "queue"),
            results_dir=Path(env.get("IJ_RESULTS_DIR") or "/opt/bench/results"),
            hypotheses=Path(env.get("IJ_HYPOTHESES") or research / "HYPOTHESES.jsonl"),
            jobs_spool=Path(env.get("IJ_JOBS_SPOOL") or research.parent / ".inference-judge-jobs"),
            noise_file=Path(env["IJ_NOISE_FILE"]) if env.get("IJ_NOISE_FILE") else None,
            legacy_state=Path(env.get("IJ_LEGACY_STATE") or loop / "state.json"),
            frozen_paths=[Path(p) for p in (frozen.split(":") if frozen is not None
                                            else [str(loop), str(loop.parent / "guard"),
                                                  str(loop.parent / "adapters"),
                                                  str(research / ".claude"), str(research / ".codex")]) if p],
            bench_lock=Path(env["IJ_BENCH_LOCK"]) if env.get("IJ_BENCH_LOCK") else None,
            alert=env.get("IJ_ALERT", ""),
            bench_cmd=shlex.split(env.get("IJ_BENCH_CMD") or "sudo -n -u bench /opt/bench/run.sh"),
            preflight_cmd=shlex.split(env.get("IJ_PREFLIGHT_CMD") or str(loop / "preflight.sh")),
            researcher_sudo=shlex.split(env.get("IJ_RESEARCHER_SUDO") or ""),
            researcher_user=env.get("IJ_RESEARCHER_USER", ""),
            policy=policy,
        )


# ------------------------------------------------------------- the world ---
def top_open_hypotheses(hyps: list[dict], state: State, n: int = 5) -> list[dict]:
    done = {r.get("id") for r in state.runs.values()}
    open_ = [h for h in hyps if str(h.get("status", "open")).lower() == "open"
             and h.get("experiment") not in done]
    ranked = sorted(enumerate(open_), key=lambda x: (
        x[1].get("rank") if isinstance(x[1].get("rank"), (int, float)) else math.inf, x[0]))
    return [h for _, h in ranked[:n]]


def observe_world(cfg: Config, state: State) -> dict:
    hyps = load_hypotheses(cfg.hypotheses)
    noise = None
    if cfg.noise_file:
        try:
            noise = json.loads(cfg.noise_file.read_text())
        except (OSError, ValueError):
            noise = None
    return {
        "queue": read_queue(cfg.queue_dir, state, hyps, cfg.results_dir),
        "hypotheses": hyps,
        "hypotheses_top": top_open_hypotheses(hyps, state),
        "noise": noise if isinstance(noise, dict) else None,
        "summary_exists": (cfg.research_dir / "SUMMARY.md").exists(),
        "bench_lock_held": bool(cfg.bench_lock and cfg.bench_lock.exists()),
        "jobs_alive": [j for j in state.jobs.values() if _pid_alive(int(j["pid"]))],
    }


def frozen_hash(paths: list) -> str:
    """sha256 over every file under the frozen trees (loop, guard, adapters, the agent's settings)."""
    h = hashlib.sha256()
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*")
            if p.is_file() and not ({"__pycache__", ".pytest_cache"} & set(p.parts)))
        for p in files:
            h.update(str(p).encode() + b"\0")
            try:
                h.update(hashlib.sha256(p.read_bytes()).digest())
            except OSError:
                h.update(b"<unreadable>")
    return h.hexdigest()


# --------------------------------------------------------------- alerts ----
def send_alert(cfg: Config, msg: str, level: str = "alert", key: str | None = None,
               now: float | None = None) -> None:
    """Log the alert, append it to ALERTS.md in the state dir, and push it to
    IJ_ALERT: an http(s) URL gets an ntfy-style POST, anything else is a
    file to append to.  A failing sink never stops the driver."""
    ev = {"type": "alert", "level": level, "msg": msg}
    if key:
        ev["key"] = key
    append_events(cfg.state_dir, [ev], now)
    line = f"{iso(_now() if now is None else now)} [{level}] {msg}\n"
    try:
        with open(cfg.state_dir / "ALERTS.md", "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    sink = cfg.alert.strip()
    if not sink:
        return
    try:
        if sink.startswith(("http://", "https://")):
            req = urllib.request.Request(sink, data=msg.encode(), method="POST",
                                         headers={"Title": f"inference-judge {level}", "Priority": "high"})
            urllib.request.urlopen(req, timeout=10).read()
        else:
            with open(sink[5:] if sink.startswith("file:") else sink, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:  # noqa: BLE001 - an alert sink must never kill the loop
        print(f"alert sink failed: {e}", file=sys.stderr)


# ------------------------------------------------------------------ jobs ---
def _pid_alive(pid: int) -> bool:
    r = subprocess.run(["ps", "-o", "pid=", "-p", str(pid)], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def _ps(pid: int) -> dict | None:
    r = subprocess.run(["ps", "-o", "uid=,pgid=", "-p", str(pid)], capture_output=True, text=True)
    parts = r.stdout.split()
    if r.returncode != 0 or len(parts) < 2:
        return None
    return {"uid": int(parts[0]), "pgid": int(parts[1])}


def _researcher_uid(cfg: Config) -> int:
    if cfg.researcher_user:
        import pwd
        return pwd.getpwnam(cfg.researcher_user).pw_uid
    return os.getuid()


def discover_jobs(cfg: Config, state: State) -> list[dict]:
    """job events for detached researcher jobs not yet in the ledger: those
    loop/bg.sh registered in the spool, and (in the two-user layout) any
    tty-less process of the researcher user alive between sessions."""
    found: dict[str, dict] = {}
    uid = _researcher_uid(cfg)
    try:
        spool = sorted(cfg.jobs_spool.glob("*.json"))
    except OSError:
        spool = []
    for f in spool:
        try:
            j = json.loads(f.read_text()[:4096])
            pid = int(j["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        info = _ps(pid)
        if not info or info["uid"] != uid or info["pgid"] != pid:
            continue                        # gone, not the researcher's, or not a group leader
        found[str(pid)] = {"type": "job", "pid": pid, "pgid": pid, "via": "spool",
                           "name": _clean(j.get("name", ""), 80), "cmd": _clean(j.get("cmd", ""), 200)}
    if cfg.researcher_user and uid != os.getuid():
        r = subprocess.run(["ps", "-U", cfg.researcher_user, "-o", "pid=,pgid=,tty=,comm="],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            pid, pgid, tty, comm = int(parts[0]), int(parts[1]), parts[2], parts[3]
            if tty not in ("?", "??") or comm in ("systemd", "(sd-pam)", "sshd"):
                continue
            found.setdefault(str(pgid), {"type": "job", "pid": pgid, "pgid": pgid, "via": "ps",
                                         "name": comm, "cmd": comm})
    return [j for k, j in found.items() if k not in state.jobs]


def reap_jobs(cfg: Config, state: State, grace_s: float = 5.0) -> list[dict]:
    """SIGTERM, then SIGKILL, every tracked job group still alive and still the
    researcher's.  Through the researcher sudo prefix when the driver is a
    different user."""
    out = []
    if os.environ.get("IJ_DRY_RUN"):
        return out                      # run.sh --dry-run: never signal real processes
    uid = _researcher_uid(cfg)
    for key, j in list(state.jobs.items()):
        pgid = int(j["pgid"])
        info = _ps(pgid)
        if info and info["uid"] == uid:
            for sig in ("TERM", "KILL"):
                subprocess.run([*cfg.researcher_sudo, "kill", f"-{sig}", "--", f"-{pgid}"],
                               capture_output=True)
                t0 = time.time()
                while time.time() - t0 < grace_s and _pid_alive(pgid):
                    time.sleep(0.1)
                if not _pid_alive(pgid):
                    break
        out.append({"type": "job_reaped", "pgid": pgid, "alive_before": bool(info)})
    return out


# -------------------------------------------------------------------- CLI --
def _state(cfg: Config) -> State:
    state = replay(load_events(cfg.state_dir / "events.jsonl"))
    if os.environ.get("IJ_DRY_RUN"):
        # run.sh --dry-run previews a driver started now: what this environment
        # sets wins over the policy the last driver start logged.
        state.policy.update({k: cfg.policy[k] for k in DEFAULT_POLICY if os.environ.get("IJ_" + k.upper())})
    return state


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _seed_clock(cfg: Config, state: State, now: float) -> None:
    """Take the clock from bootstrap's legacy loop/state.json once."""
    if state.clock_start is not None:
        return
    try:
        legacy = json.loads(cfg.legacy_state.read_text())
    except (OSError, ValueError):
        return
    start = float(legacy.get("clock_start_epoch") or 0)
    if start <= 0:
        return
    hours = float(legacy.get("budget_hours") or 24)
    append_events(cfg.state_dir, [{"type": "run_config", "clock_start": start,
                                   "wall_deadline": start + hours * 3600,
                                   "accel_budget_s": hours * 3600, "source": str(cfg.legacy_state)}], now)


def cmd_start(cfg: Config, now: float, args: list[str]) -> int:
    append_events(cfg.state_dir, [{"type": "driver_start", "pid": os.getppid(), "policy": cfg.policy}], now)
    return 0


def cmd_start_clock(cfg: Config, now: float, args: list[str]) -> int:
    """--accel-hours H  --wall-hours H | --deadline EPOCH"""
    opts = dict(zip(args[::2], args[1::2]))
    accel_h = float(opts.get("--accel-hours", 24))
    if "--deadline" in opts:
        deadline = float(opts["--deadline"])
    else:
        deadline = now + float(opts.get("--wall-hours", accel_h)) * 3600
    append_events(cfg.state_dir, [{"type": "run_config", "clock_start": now, "wall_deadline": deadline,
                                   "accel_budget_s": accel_h * 3600, "source": "start-clock"}], now)
    print(f"clock started {iso(now)}; accelerator budget {accel_h}h; wall deadline {iso(deadline)}")
    return 0


def _observe_and_record(cfg: Config, now: float) -> State:
    """Everything the driver learns between two decisions, logged."""
    state = _state(cfg)
    _seed_clock(cfg, state, now)
    evs: list[dict] = []
    if state.clock_start is not None:
        evs.append({"type": "tick"})
    if state.clock_start is not None:       # before the clock there is no run for a result to belong to
        evs += scan_results(cfg.results_dir, state)
    evs += discover_jobs(cfg, state)
    resume = cfg.state_dir / "RESUME"
    if state.paused and resume.exists():
        evs.append({"type": "resume", "by": "RESUME file"})
        resume.unlink(missing_ok=True)
        evs.append({"type": "frozen_hash", "hash": frozen_hash(cfg.frozen_paths), "why": "re-baselined on resume"})
    append_events(cfg.state_dir, evs, now)
    state = _state(cfg)
    if cfg.frozen_paths and not state.paused:
        h = frozen_hash(cfg.frozen_paths)
        if state.frozen_hash is None:
            append_events(cfg.state_dir, [{"type": "frozen_hash", "hash": h}], now)
        elif h != state.frozen_hash:
            send_alert(cfg, "the frozen files (loop/, guard/, adapters/ and the agent's settings) changed "
                       "under the running loop; sessions are paused. Inspect, then touch RESUME in the "
                       "state dir.",
                       key=f"frozen:{h}", now=now)
            append_events(cfg.state_dir, [{"type": "pause", "reason": "frozen files changed"}], now)
        state = _state(cfg)
    return state


def cmd_next(cfg: Config, now: float, args: list[str]) -> int:
    """Observe, decide, and print one action line for run.sh:
         session <model> <timeout_s> <prompt_file> <reason>
         sleep <seconds> <reason...>
         finish <reason...>"""
    state = _observe_and_record(cfg, now)
    world = observe_world(cfg, state)
    act = decide(state, now, world)
    if act.kind == "session":
        brief_path = cfg.state_dir / "BRIEF.md"
        _write_atomic(brief_path, render_brief(state, now, world))
        prompt = render_prompt(state, now, world, brief_path, act.reason)
        prompt_file = cfg.state_dir / "prompt.txt"
        _write_atomic(prompt_file, prompt)
        append_events(cfg.state_dir, [{"type": "brief"},
                                      {"type": "session_start", "model": act.model, "reason": act.reason,
                                       "timeout_s": act.timeout_s,
                                       "queue_before": len(world["queue"][0])}], now)
        print(f"session {act.model} {act.timeout_s} {prompt_file} {act.reason}")
    elif act.kind == "finish":
        _finish(cfg, state, now, act.reason)
        print(f"finish {act.reason}")
    else:
        print(f"sleep {act.sleep_s:.1f} {act.reason}")
    return 0


def _finish(cfg: Config, state: State, now: float, reason: str) -> None:
    if state.finished:
        return
    append_events(cfg.state_dir, reap_jobs(cfg, state), now)
    world = observe_world(cfg, _state(cfg))
    st = _state(cfg)
    _write_atomic(cfg.state_dir / "BRIEF.md", render_brief(st, now, world))
    if not world["summary_exists"]:
        t = st.tallies()
        rows = "\n".join(f"| {v['id']} | {v.get('verdict')} | {v.get('headline', v.get('delta', '-'))} |"
                         for v in sorted(st.runs.values(), key=lambda r: r["seq"]))
        _write_atomic(cfg.state_dir / "SUMMARY.driver.md",
                      f"# Driver summary stub ({iso(now)})\n\nThe researcher wrote no SUMMARY.md. "
                      f"Stopped: {reason}.\n\nTallies: {json.dumps(t)}\n\n| id | verdict | effect |\n"
                      f"|---|---|---|\n{rows}\n")
    append_events(cfg.state_dir, [{"type": "finish", "reason": reason}], now)
    send_alert(cfg, f"run finished: {reason}", level="info", now=now)


def cmd_session_end(cfg: Config, now: float, args: list[str]) -> int:
    """--class ok|timeout|limit|auth|crash --rc N --dur S [--sleep S] [--usage-file F]"""
    opts = dict(zip(args[::2], args[1::2]))
    before = _state(cfg)
    world = observe_world(cfg, before)
    ev = {"type": "session_end", "class": opts["--class"], "rc": int(opts.get("--rc", 0)),
          "dur": float(opts.get("--dur", 0)), "model": before.model(),
          "reason": (before.session_open or {}).get("reason"),
          "queue_after": len(world["queue"][0])}
    if opts.get("--sleep"):
        ev["sleep_s"] = float(opts["--sleep"])
    uf = opts.get("--usage-file")
    if uf:
        try:
            doc = json.loads(Path(uf).read_text())
            ev["usage"] = {k: doc.get(k) for k in ("total_cost_usd", "num_turns", "duration_ms")
                           if doc.get(k) is not None}
            if isinstance(doc.get("usage"), dict):
                ev["usage"].update({k: v for k, v in doc["usage"].items() if isinstance(v, (int, float))})
        except (OSError, ValueError, AttributeError):
            pass
    append_events(cfg.state_dir, [ev], now)
    after = _state(cfg)
    if after.mode == "fallback" and before.mode == "primary":
        send_alert(cfg, f"{after.policy['max_crashes']} consecutive crashed sessions on the "
                   f"{after.policy['primary_model']} model; running the next {after.fallback_left} on the "
                   f"{after.policy['fallback_model']} model. See iterations/ in the state dir.",
                   level="warn", now=now)
    if after.paused and not before.paused:
        why = {"auth": "the agent's authentication failed (expired login or key). Sessions are paused and "
                       "retried every %ds; the ruler queue keeps running. Log in again as the "
                       "researcher user." % after.policy["auth_probe_s"],
               "crashes": "sessions keep crashing on the fallback model too. Sessions are paused; the "
                          "ruler queue keeps running. Fix, then touch RESUME in the state dir."}
        send_alert(cfg, why.get(after.paused["reason"], after.paused["reason"]), now=now)
    if after.limit_streak >= after.policy["limit_alert_streak"] and \
            f"limit:{after.sessions}" not in after.alert_keys and before.limit_streak < after.policy["limit_alert_streak"]:
        send_alert(cfg, f"{after.limit_streak} quota limits in a row (a weekly limit?); the driver keeps "
                   "sleeping and retrying.", level="warn", key=f"limit:{after.sessions}", now=now)
    print(json.dumps({"crash_streak": after.crash_streak, "mode": after.mode,
                      "paused": after.paused, "limit_streak": after.limit_streak}))
    return 0


def cmd_get(cfg: Config, now: float, args: list[str]) -> int:
    """Print one State field (or a derived one) for the bash side."""
    s = _state(cfg)
    k = args[0]
    v = {"model": s.model(), "end": s.end_ts(now), "tallies": s.tallies()}.get(k, getattr(s, k, None))
    print(json.dumps(v, default=list) if not isinstance(v, (str, int, float)) else v)
    return 0


def cmd_resume(cfg: Config, now: float, args: list[str]) -> int:
    append_events(cfg.state_dir, [{"type": "resume", "by": "cli"},
                                  {"type": "frozen_hash", "hash": frozen_hash(cfg.frozen_paths),
                                   "why": "re-baselined on resume"}], now)
    print("resumed")
    return 0


def cmd_brief(cfg: Config, now: float, args: list[str]) -> int:
    s = _state(cfg)
    print(render_brief(s, now, observe_world(cfg, s)))
    return 0


def cmd_heartbeat(cfg: Config, now: float, args: list[str]) -> int:
    """heartbeat.json for the outside world, a budget tick, and an alert when
    the driver has logged nothing for far longer than a session can last."""
    s = _state(cfg)
    if s.clock_start is not None and not s.finished:
        append_events(cfg.state_dir, [{"type": "tick", "via": "heartbeat"}], now)
        s = _state(cfg)
    pid = None
    try:
        pid = int((cfg.state_dir / "driver.pid").read_text().strip())
    except (OSError, ValueError):
        pass
    try:
        du = os.statvfs(cfg.state_dir)
        disk_pct = round(100 * (1 - du.f_bavail / du.f_blocks), 1)
    except OSError:
        disk_pct = None
    driver_events = [e for e in load_events(cfg.state_dir / "events.jsonl")
                     if e["type"] in ("session_start", "session_end", "driver_start", "brief")
                     or (e["type"] == "tick" and "via" not in e)]
    last_driver = driver_events[-1]["ts"] if driver_events else None
    last = max(s.runs.values(), key=lambda r: r["seq"]) if s.runs else None
    hb = {"ts": now, "utc": iso(now), "driver_pid": pid,
          "driver_alive": bool(pid and _pid_alive(pid)), "last_driver_event": iso(last_driver),
          "runner": s.runner, "paused": s.paused, "finished": s.finished, "mode": s.mode,
          "accel_used_h": round(s.accel_used_s / 3600, 3), "end": iso(s.end_ts(now)),
          "disk_pct": disk_pct, "last_verdict": last and {"id": last["id"], "verdict": last.get("verdict")}}
    _write_atomic(cfg.state_dir / "heartbeat.json", json.dumps(hb, indent=1))
    wedge = s.policy["iter_timeout_s"] + 900
    if last_driver and not s.paused and not s.finished and now - last_driver > wedge \
            and f"wedged:{int(last_driver)}" not in s.alert_keys:
        send_alert(cfg, f"the driver has logged nothing for {int((now - last_driver) // 60)} min "
                   f"(pid {pid}, alive={hb['driver_alive']}).", key=f"wedged:{int(last_driver)}", now=now)
    return 0


COMMANDS = {
    "start": cmd_start, "start-clock": cmd_start_clock, "next": cmd_next,
    "session-end": cmd_session_end, "get": cmd_get, "resume": cmd_resume,
    "brief": cmd_brief, "heartbeat": cmd_heartbeat,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print("usage: loopstate.py {" + "|".join(COMMANDS) + "} [args]", file=sys.stderr)
        return 2
    cfg = Config.from_env()
    return COMMANDS[argv[0]](cfg, _now(), argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

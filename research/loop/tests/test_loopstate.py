"""loopstate: replay(events) -> State, decide(state, now, world) -> Action,
render_brief(state, now, world) -> str.  Pure functions over an event log;
the tests build logs by hand."""
from __future__ import annotations

import json
import os
import time

import loopstate as ls
from loopstate import decide, render_brief, render_prompt, replay

H = 3600.0


def log(*evs):
    """Number and timestamp hand-written events like append_events would."""
    out = []
    for i, (ts, ev) in enumerate(evs, 1):
        out.append({"seq": i, "ts": ts, **ev})
    return out


def clock(t0, accel_h=24, wall_h=24):
    return (t0, {"type": "run_config", "clock_start": t0, "wall_deadline": t0 + wall_h * H,
                 "accel_budget_s": accel_h * H})


def start(t, **policy):
    return (t, {"type": "driver_start", "policy": policy})


def sess(t, reason="first"):
    return (t, {"type": "session_start", "reason": reason, "model": "primary"})


def end(t, cls="ok", **kw):
    return (t, {"type": "session_end", "class": cls, **kw})


def verdict(t, rid, exp_id, v="REJECTED", **kw):
    return (t, {"type": "verdict", "run_id": rid, "id": exp_id, "verdict": v, **kw})


EMPTY = {"queue": ([], [])}


# ------------------------------------------------------------- dedupe ------
def test_a_run_seen_twice_counts_once(T0):
    st = replay(log(clock(T0), verdict(T0 + 10, "r1", "exp-a", source="runner"),
                    verdict(T0 + 20, "r1", "exp-a", source="scan"),
                    verdict(T0 + 30, "r2", "exp-b", "PROMOTED")))
    t = st.tallies()
    assert (t["runs"], t["rejected"], t["promoted"], t["scored"]) == (2, 1, 1, 2)


def test_touching_results_json_does_not_double_count(tmp_path, T0):
    """An earlier version counted a results.json again whenever its mtime moved past the last
    one recorded (restore from archive, touch, rsync).  The scan keys on run id."""
    d = tmp_path / "exp-a"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"id": "exp-a", "verdict": "REJECTED"}))
    first = ls.scan_results(tmp_path, ls.State())
    st = replay([{"seq": 1, "ts": T0, **first[0]}])
    os.utime(d / "results.json", (T0 + 999, T0 + 999))
    assert ls.scan_results(tmp_path, st) == []
    assert st.tallies()["runs"] == 1


def _results(tmp_path, name, **doc):
    d = tmp_path / name
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"id": name, "verdict": "REJECTED", **doc}))


def test_a_run_from_before_the_clock_is_not_tallied(tmp_path, T0):
    """rehearse.sh (or a failed earlier attempt) leaves candidate
    results behind; the scan logged them and they counted against the new run."""
    stamp = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    _results(tmp_path, "exp-before", started=stamp(T0 - 7200), finished=stamp(T0 - 7000))
    _results(tmp_path, "exp-after", started=stamp(T0 + 60), finished=stamp(T0 + 600))
    _results(tmp_path, "exp-straddles", started=stamp(T0 - 30), finished=stamp(T0 + 300))
    _results(tmp_path, "exp-untimed")                       # stub results carry no timestamps
    st = replay(log(clock(T0)))
    got = {e["id"] for e in ls.scan_results(tmp_path, st)}
    assert got == {"exp-after", "exp-untimed"}


def test_the_scan_waits_for_the_clock(tmp_path, T0):
    _results(tmp_path, "exp-a")
    assert ls.scan_results(tmp_path, ls.State()) != []          # no clock: nothing to be before (unit level)
    cfg = ls.Config.from_env({"IJ_STATE_DIR": str(tmp_path / "state"),
                              "IJ_RESEARCH_DIR": str(tmp_path / "research"),
                              "IJ_RESULTS_DIR": str(tmp_path),
                              "IJ_LEGACY_STATE": str(tmp_path / "none.json")})
    cfg.state_dir.mkdir()
    ls._observe_and_record(cfg, T0)
    assert ls._state(cfg).tallies()["runs"] == 0


def test_preflight_failures_and_process_losses_are_not_runs(T0):
    st = replay(log(clock(T0), verdict(T0, "p1", "exp-a", "PREFLIGHT_FAILED", sig="s1"),
                    verdict(T0, "x1", "exp-b", "NO_RESULT"), verdict(T0, "b1", "exp-c", "BUILD_FAILED")))
    t = st.tallies()
    assert (t["preflight_failed"], t["process"], t["build_failed"], t["scored"], t["runs"]) == (1, 1, 1, 0, 2)
    assert "exp-a" not in st.consumed and {"exp-b", "exp-c"} <= st.consumed


# ------------------------------------------------------------- budget ------
def test_wall_deadline_and_accelerator_budget(T0):
    st = replay(log(clock(T0, accel_h=10, wall_h=48), start(T0)))
    assert st.end_ts(T0) == T0 + 10 * H                 # accel budget binds first
    st = replay(log(clock(T0, accel_h=10, wall_h=5), start(T0)))
    assert st.end_ts(T0) == T0 + 5 * H                  # wall deadline binds first


def test_accelerator_time_is_charged_only_while_up(T0):
    evs = [clock(T0, wall_h=48), start(T0)]
    evs += [(T0 + 60 * i, {"type": "tick"}) for i in range(1, 11)]    # 10 min up
    evs += [(T0 + 5 * H, {"type": "tick"})]                          # box was off: gap capped
    evs += [start(T0 + 6 * H), (T0 + 6 * H + 60, {"type": "tick"})]   # restart: not charged
    st = replay(log(*evs))
    assert st.accel_used_s == 600 + 300 + 60
    assert abs(st.end_ts(T0 + 6 * H + 60) - (T0 + 6 * H + 60 + 24 * H - 960)) < 1e-6


def test_a_long_session_is_charged_in_full(T0):
    st = replay(log(clock(T0), start(T0), sess(T0), end(T0 + 3000, dur=3000)))
    assert st.accel_used_s == 3000


def test_finish_at_wall_deadline_and_when_budget_spent(T0):
    st = replay(log(clock(T0, wall_h=1), start(T0), sess(T0), end(T0 + 10)))
    a = decide(st, T0 + H, EMPTY)
    assert (a.kind, a.reason) == ("finish", "wall deadline reached")
    evs = [clock(T0, accel_h=0.1, wall_h=24), start(T0)] + [(T0 + 60 * i, {"type": "tick"}) for i in range(1, 7)]
    a = decide(replay(log(*evs)), T0 + 360, EMPTY)
    assert (a.kind, a.reason) == ("finish", "accelerator-hour budget spent")


def test_session_is_capped_by_the_end_of_the_run(T0):
    st = replay(log(clock(T0, wall_h=1), start(T0)))
    a = decide(st, T0 + H - 600, EMPTY)
    assert a.kind == "session" and a.timeout_s == 600


def test_no_clock_means_wait(T0):
    assert decide(replay(log(start(T0))), T0, EMPTY).kind == "wait_clock"


# ----------------------------------------------------------- recovery ------
def crashes(t, n, step=10):
    return [end(t + step * i, "crash", rc=1) for i in range(1, n + 1)]


def test_crashes_back_off_then_fall_back_then_pause(T0):
    base = [clock(T0), start(T0, backoff_base_s=60), sess(T0)]
    st = replay(log(*base, *crashes(T0, 1)))
    assert st.crash_streak == 1 and st.mode == "primary"
    a = decide(st, T0 + 11, EMPTY)
    assert a.kind == "sleep" and "crash backoff" in a.reason
    assert decide(st, T0 + 10 + 60, EMPTY).reason == "retry_after_crash"
    st = replay(log(*base, *crashes(T0, 2)))
    assert decide(st, T0 + 20 + 119, EMPTY).kind == "sleep"            # 2nd back-off is 120 s
    assert decide(st, T0 + 20 + 120, EMPTY).kind == "session"

    st = replay(log(*base, *crashes(T0, 3)))                           # 3rd: fallback model
    assert (st.mode, st.fallback_left, st.crash_streak, st.paused) == ("fallback", 3, 0, None)
    assert decide(st, T0 + 100, EMPTY).model == "fallback"   # a role; the adapter picks the model

    st = replay(log(*base, *crashes(T0, 3), *crashes(T0 + 100, 3)))  # 3 more on the fallback: pause
    assert st.paused["reason"] == "crashes"
    a = decide(st, T0 + 10 * H, EMPTY)
    assert a.kind == "sleep" and "paused" in a.reason


def test_fallback_returns_to_primary_after_n_good_sessions(T0):
    evs = [clock(T0), start(T0), *crashes(T0, 3), end(T0 + 100), end(T0 + 200)]
    st = replay(log(*evs))
    assert (st.mode, st.fallback_left) == ("fallback", 1)
    st = replay(log(*evs, end(T0 + 300)))
    assert (st.mode, st.model()) == ("primary", "primary")


def test_resume_clears_a_pause(T0):
    evs = [clock(T0), start(T0), *crashes(T0, 6), (T0 + 500, {"type": "resume"})]
    st = replay(log(*evs))
    assert st.paused is None and st.mode == "primary" and st.crash_streak == 0


def test_quota_limit_is_never_a_crash(T0):
    evs = [clock(T0), start(T0)] + [end(T0 + 10 * i, "limit", sleep_s=600) for i in range(1, 10)]
    st = replay(log(*evs))
    assert st.crash_streak == 0 and st.paused is None and st.limit_streak == 9
    a = decide(st, T0 + 100, EMPTY)
    assert a.kind == "sleep" and a.reason == "quota limit"
    assert decide(st, T0 + 90 + 600, EMPTY).kind == "session"


def test_auth_failure_pauses_and_probes(T0):
    evs = [clock(T0), start(T0, auth_probe_s=1800), *crashes(T0, 2), end(T0 + 50, "auth", rc=1)]
    st = replay(log(*evs))
    assert st.paused["reason"] == "auth" and st.crash_streak == 2      # not counted as a crash
    assert decide(st, T0 + 60, EMPTY).reason == "paused: auth"
    a = decide(st, T0 + 50 + 1800, EMPTY)
    assert (a.kind, a.reason) == ("session", "auth_probe")
    st = replay(log(*evs, end(T0 + 2000, "ok")))
    assert st.paused is None


def test_timeout_is_not_a_crash(T0):
    st = replay(log(clock(T0), start(T0), *crashes(T0, 2), end(T0 + 100, "timeout", rc=124)))
    assert st.crash_streak == 0


# ------------------------------------------------------------ triggers -----
def test_triggers(T0):
    base = [clock(T0), start(T0, timer_s=3600, min_gap_s=60, queue_target=2), sess(T0), end(T0 + 100)]
    st = replay(log(*base))
    q2 = {"queue": ([ls.QueueItem("a-1", "candidate", "s"), ls.QueueItem("b-1", "candidate", "s")], [])}
    assert decide(st, T0 + 130, q2).reason == "min gap"
    assert decide(st, T0 + 200, q2).reason == "idle"
    assert decide(st, T0 + 200, EMPTY).reason == "queue_low"
    assert decide(st, T0 + 100 + 3600, q2).reason == "timer"
    st = replay(log(*base, (T0 + 150, {"type": "brief"}), verdict(T0 + 160, "r1", "exp-a")))
    assert decide(st, T0 + 200, q2).reason == "verdict"
    st = replay(log(*base, (T0 + 150, {"type": "brief"}), verdict(T0 + 140, "r1", "exp-a")))
    # verdict logged before the brief: already shown, not a trigger
    st.runs["r1"]["seq"] = 0
    assert decide(st, T0 + 200, q2).reason == "idle"


def test_queue_low_backs_off_when_the_researcher_leaves_it_low(T0):
    evs = [clock(T0), start(T0, timer_s=3600, min_gap_s=60, queue_target=2)]
    for i in range(4):
        evs += [sess(T0 + 1000 * i, "queue_low"), end(T0 + 1000 * i + 10, reason="queue_low", queue_after=0)]
    st = replay(log(*evs))
    last_end = T0 + 3010
    assert st.queue_low_streak == 4
    assert decide(st, last_end + 60 * 16 - 1, EMPTY).kind == "sleep"
    assert decide(st, last_end + 60 * 16, EMPTY).reason == "queue_low"


def test_wrap_up_once_before_the_end(T0):
    evs = [clock(T0, wall_h=10), start(T0, summary_lead_s=1800), sess(T0), end(T0 + 10)]
    st = replay(log(*evs))
    now = T0 + 10 * H - 1700
    assert decide(st, now, EMPTY).reason == "wrap_up"
    assert decide(st, now, {**EMPTY, "summary_exists": True}).reason != "wrap_up"
    st2 = replay(log(*evs, sess(now, "wrap_up"), end(now + 100)))
    assert decide(st2, now + 200, EMPTY).reason != "wrap_up"


# --------------------------------------------------------------- brief -----
def test_brief_has_what_the_researcher_needs(T0):
    evs = [clock(T0, accel_h=20, wall_h=48), start(T0), sess(T0), end(T0 + 100),
           verdict(T0 + 150, "r0", "exp-old"), (T0 + 200, {"type": "brief"}),
           verdict(T0 + 300, "r1", "exp-new", "PROMOTED", delta=-0.08, mde=0.05, sigma_L=0.02),
           (T0 + 400, {"type": "run_start", "id": "exp-run", "kind": "candidate"}),
           (T0 + 450, {"type": "job", "pid": 4242, "pgid": 4242, "name": "sweep"}),
           (T0 + 460, {"type": "alert", "msg": "something odd", "level": "warn"})]
    st = replay(log(*evs))
    world = {"queue": ([ls.QueueItem("exp-q1", "candidate", "s")], [("bad.json", "s", "kind must be ...")]),
             "hypotheses_top": [{"id": "H7", "title": "fuse sampler", "experiment": "exp-q1", "prediction": -6}],
             "jobs_alive": [{"pgid": 4242, "name": "sweep"}], "bench_lock_held": True}
    now = T0 + 1000
    b = render_brief(st, now, world)
    assert "UTC now: " + ls.iso(now) in b
    assert st.accel_used_s == 100                          # the session was charged
    assert "run ends " + ls.iso(now + 20 * H - 100) in b and "phase: explore" in b
    assert "Ruler: RUNNING exp-run" in b and "Bench lock: HELD" in b
    assert "exp-q1 [candidate]" in b and "REFUSED queue/bad.json" in b
    new = b.split("## New verdicts since the last brief")[1].split("##")[0]
    assert "exp-new: PROMOTED" in new and "exp-old" not in new
    assert "MDE=0.05" in b and "sigma_L=0.02" in b
    assert "H7: fuse sampler" in b and "predicted=-6" in b
    assert "pgid 4242: sweep" in b and "something odd" in b
    assert render_brief(st, now, world) == b               # pure


def test_brief_noise_floor_fallback_and_file(T0):
    st = replay(log(clock(T0), start(T0)))
    assert "not calibrated" in render_brief(st, T0, EMPTY)
    b = render_brief(st, T0, {**EMPTY, "noise": {"mde": 0.031, "sigma_L": 0.011}})
    assert "MDE=0.031" in b and "not calibrated" not in b


def test_prompt_points_at_the_brief(T0, tmp_path):
    st = replay(log(clock(T0), start(T0)))
    p = render_prompt(st, T0 + 5, EMPTY, tmp_path / "BRIEF.md", "first")
    assert f"Read {tmp_path / 'BRIEF.md'} first" in p
    assert "UTC now: " + ls.iso(T0 + 5) in p and "Do not call run.sh yourself" in p
    assert "AGENTS.md" in p                 # for an agent that does not load that file by itself
    assert "SUMMARY.md" in render_prompt(st, T0 + 5, EMPTY, tmp_path / "BRIEF.md", "wrap_up")


def test_torn_last_line_is_skipped(tmp_path):
    f = tmp_path / "events.jsonl"
    f.write_text('{"seq":1,"ts":1,"type":"tick"}\n{"seq":2,"ts":2,"type":"ver')
    assert [e["seq"] for e in ls.load_events(f)] == [1]
    ls.append_events(tmp_path, [{"type": "tick"}], now=3)
    assert [e["seq"] for e in ls.load_events(f)][-1] == 2


def test_operator_runs_are_not_counted_as_experiment_runs(T0):
    """Baseline and calibration are the operator's launches.  Counting them as
    runs made the brief overstate what the researcher had spent."""
    st = replay(log(clock(T0), verdict(T0, "b1", "baseline-mock", "BASELINE"),
                    verdict(T0, "c1", "calibrate-1", "CALIBRATED"),
                    verdict(T0, "r1", "exp-a", "REJECTED")))
    t = st.tallies()
    assert t["runs"] == 1 and t["scored"] == 1
    assert t["baseline"] == 1 and t["calibrated"] == 1

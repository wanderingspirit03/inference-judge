"""The runner: queue dir in, verdict events out (DESIGN 2.7).

Everything here goes through loop/runner.sh --once, the same entry point
systemd runs, with the bench replaced by a stub that records its argv.  The
security property under test: nothing from a queue file but a validated id
and an enum kind can reach the bench argv.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from conftest import LOOP, Lab


@pytest.fixture
def clocked(lab: Lab) -> Lab:
    lab.start_clock(24)
    return lab


def verdicts(lab: Lab) -> list[dict]:
    return [e for e in lab.events() if e["type"] == "verdict"]


def rejects(lab: Lab) -> dict[str, str]:
    return {e["file"]: e["reason"] for e in lab.events() if e["type"] == "queue_reject"}


# ------------------------------------------------------------ happy paths --
def test_candidate_reaches_bench_as_exactly_its_id(clocked: Lab):
    clocked.ready("exp-alpha")
    r = clocked.runner_once()
    assert r.returncode == 0, r.stderr
    assert clocked.bench_calls() == [["exp-alpha"]]
    v = verdicts(clocked)
    assert [(x["id"], x["verdict"], x["run_id"]) for x in v] == [("exp-alpha", "REJECTED", "exp-alpha-r1")]
    assert (clocked.state / "runs.jsonl").read_text().count("exp-alpha-r1") == 1
    assert (clocked.state / "runs" / "exp-alpha.log").exists()


def test_instrument_gets_the_instrument_flag_and_needs_no_prediction(clocked: Lab):
    clocked.ready("noise-floor-1", kind="instrument")
    clocked.runner_once()
    assert clocked.bench_calls() == [["--instrument", "noise-floor-1"]]
    assert verdicts(clocked)[0]["verdict"] == "INSTRUMENT"


def test_serial_fifo_and_never_twice(clocked: Lab):
    clocked.ready("exp-one")
    time.sleep(0.02)
    clocked.ready("exp-two")
    for _ in range(3):
        clocked.runner_once()
    assert clocked.bench_calls() == [["exp-one"], ["exp-two"]]
    assert clocked.state_now().tallies()["runs"] == 2


def test_runner_and_driver_scan_count_a_run_once(clocked: Lab):
    clocked.ready("exp-dedupe")
    clocked.runner_once()
    clocked.lstate("next")       # scans the results dir too
    clocked.lstate("next")
    st = clocked.state_now()
    assert st.tallies()["runs"] == 1
    assert len([e for e in clocked.events() if e["type"] == "verdict"]) == 1


def test_no_result_is_a_process_verdict_not_a_hang(clocked: Lab):
    clocked.ready("exp-noresult")
    clocked.runner_once(FAKE_BENCH_NO_RESULT=1)
    v = verdicts(clocked)
    assert v[0]["verdict"] == "NO_RESULT" and v[0]["id"] == "exp-noresult"
    assert clocked.state_now().runner == {"state": "idle"}


# ------------------------------------------------------- malicious queue --
def test_extra_fields_with_shell_metacharacters_never_reach_argv(clocked: Lab, tmp_path):
    pwned = tmp_path / "pwned"
    clocked.experiment("exp-evil")
    clocked.predict("exp-evil")
    clocked.enqueue("exp-evil", cmd=f"; touch {pwned}", args=["--allow-baseline", f"$(touch {pwned})"],
                    flags="`touch " + str(pwned) + "`", instrument=True, env={"LD_PRELOAD": "/x.so"})
    clocked.runner_once()
    assert clocked.bench_calls() == [["exp-evil"]]
    assert not pwned.exists()


@pytest.mark.parametrize("name,doc,why", [
    ("evil.json", {"id": "../../etc/passwd", "kind": "candidate"}, "id missing"),
    ("exp-a;touch x.json", {"id": "exp-a;touch x", "kind": "candidate"}, "file name"),
    ("exp-b.json", {"id": "exp-b;touch x", "kind": "candidate"}, "id missing"),
    ("exp-c.json", {"id": "exp-c", "kind": "candidate --instrument"}, "kind must be"),
    ("exp-d.json", {"id": "exp-x", "kind": "candidate"}, "does not match the file name"),
    ("exp-e.json", {"id": "exp-e"}, "kind must be"),
    ("exp-f.json", ["exp-f", "candidate"], "not a JSON object"),
    ("baseline-v2.json", {"id": "baseline-v2", "kind": "instrument"}, "reserved"),
    ("Exp-G.json", {"id": "Exp-G", "kind": "candidate"}, "file name"),
    ("ab.json", {"id": "ab", "kind": "candidate"}, "file name"),
    ("-rf.json", {"id": "-rf", "kind": "candidate"}, "file name"),
])
def test_bad_queue_files_are_refused_and_logged(clocked: Lab, name, doc, why):
    (clocked.queue / name).write_text(json.dumps(doc))
    clocked.runner_once()
    assert clocked.bench_calls() == []
    assert why in rejects(clocked).get(name, ""), rejects(clocked)


def test_traversal_in_file_name_cannot_escape(clocked: Lab, tmp_path):
    # a file can not have "/" in its name; a "..": prefixed name must still be refused
    (clocked.queue / "..exp-a.json").write_text(json.dumps({"id": "exp-a", "kind": "candidate"}))
    (clocked.queue / "...json").write_text(json.dumps({"id": "exp-a", "kind": "candidate"}))
    clocked.runner_once()
    assert clocked.bench_calls() == []


def test_symlinked_queue_file_is_refused(clocked: Lab, tmp_path):
    clocked.experiment("exp-link")
    clocked.predict("exp-link")
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"id": "exp-link", "kind": "candidate"}))
    os.symlink(target, clocked.queue / "exp-link.json")
    clocked.runner_once()
    assert clocked.bench_calls() == []
    assert "symlink" in rejects(clocked)["exp-link.json"]


def test_fifo_and_huge_queue_files_are_refused_without_hanging(clocked: Lab):
    os.mkfifo(clocked.queue / "exp-fifo.json")
    clocked.experiment("exp-huge")
    clocked.predict("exp-huge")
    clocked.enqueue("exp-huge", pad="x" * 20000)
    r = clocked.runner_once()       # would block forever on the FIFO if opened
    assert r.returncode == 0
    assert clocked.bench_calls() == []
    rj = rejects(clocked)
    assert "regular" in rj["exp-fifo.json"] and "larger" in rj["exp-huge.json"]


@pytest.mark.parametrize("prediction", [None, "faster", True, [], {"note": "big"}, float("nan")])
def test_candidate_without_numeric_prediction_is_refused(clocked: Lab, prediction):
    clocked.experiment("exp-nopred")
    clocked.hypotheses.write_text(json.dumps({"id": "H1", "experiment": "exp-nopred",
                                              "prediction": prediction}).replace("NaN", '"nan"') + "\n")
    clocked.enqueue("exp-nopred")
    clocked.runner_once()
    assert clocked.bench_calls() == []
    assert "numeric prediction" in rejects(clocked)["exp-nopred.json"]


def test_prediction_may_be_structured(clocked: Lab):
    clocked.experiment("exp-struct")
    clocked.predict("exp-struct", prediction={"tps_delta_pct": 8, "why": "fewer syncs"})
    clocked.enqueue("exp-struct")
    clocked.runner_once()
    assert clocked.bench_calls() == [["exp-struct"]]


def test_used_id_is_refused(clocked: Lab):
    (clocked.results / "exp-used").mkdir()
    clocked.ready("exp-used")
    clocked.runner_once()
    assert clocked.bench_calls() == []
    assert "already exists" in rejects(clocked)["exp-used.json"]


# --------------------------------------------------------------- preflight --
def test_preflight_failure_costs_no_slot_and_waits_for_an_edit(clocked: Lab):
    clocked.experiment("exp-broken", launch="#!/usr/bin/env bash\nif then fi\n")
    clocked.predict("exp-broken")
    q = clocked.enqueue("exp-broken")
    clocked.runner_once()
    assert clocked.bench_calls() == []
    v = verdicts(clocked)
    assert [x["verdict"] for x in v] == ["PREFLIGHT_FAILED"] and v[0]["gpu_s"] == 0
    assert "launch.sh does not parse" in v[0]["reasons"][0]
    clocked.runner_once()                     # same file: not retried
    assert len(verdicts(clocked)) == 1
    clocked.experiment("exp-broken")          # fixed
    time.sleep(0.01)
    q.write_text(q.read_text() + " ")         # the researcher re-queues
    clocked.runner_once()
    assert clocked.bench_calls() == [["exp-broken"]]
    t = clocked.state_now().tallies()
    assert t["preflight_failed"] == 1 and t["runs"] == 1


def test_preflight_runs_through_the_researcher_prefix(clocked: Lab, tmp_path):
    trace = tmp_path / "sudo.trace"
    sudo = clocked.bin / "fake-sudo"
    sudo.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> {trace}\nexec "$@"\n')
    sudo.chmod(0o755)
    clocked.ready("exp-sudo")
    clocked.runner_once(IJ_RESEARCHER_SUDO=str(sudo))
    assert "preflight.sh exp-sudo" in trace.read_text()
    assert clocked.bench_calls() == [["exp-sudo"]]


def test_preflight_gets_the_paths_through_a_sudo_that_resets_the_environment(clocked: Lab):
    # real sudo drops IJ_*; preflight run from /opt/inference-judge/loop would then look
    # for /opt/inference-judge/experiments/<id> (found in the two-user container test)
    sudo = clocked.bin / "fake-sudo-reset"
    sudo.write_text('#!/usr/bin/env bash\nexec env -i PATH="$PATH" HOME="$HOME" "$@"\n')
    sudo.chmod(0o755)
    elsewhere = clocked.tmp / "opt-inference-judge" / "loop"
    elsewhere.mkdir(parents=True)
    (elsewhere / "preflight.sh").write_bytes((LOOP / "preflight.sh").read_bytes())
    (elsewhere / "preflight.sh").chmod(0o755)
    clocked.ready("exp-reset")
    clocked.runner_once(IJ_RESEARCHER_SUDO=str(sudo), IJ_PREFLIGHT_CMD=str(elsewhere / "preflight.sh"))
    assert clocked.bench_calls() == [["exp-reset"]], clocked.events()[-1]


# ------------------------------------------------------------- when idle --
def test_runner_idles_without_a_clock(lab: Lab):
    lab.ready("exp-noclock")
    r = lab.runner_once()
    assert "clock not started" in r.stdout and lab.bench_calls() == []


def test_runner_starts_nothing_too_close_to_the_end(lab: Lab):
    lab.start_clock(24, deadline=time.time() + 600)
    lab.ready("exp-late")
    r = lab.runner_once()
    assert "too little time" in r.stdout and lab.bench_calls() == []


def test_a_second_runner_refuses_to_start(clocked: Lab):
    import fcntl
    with open(clocked.state / "runner.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = clocked.runner_once()
    assert r.returncode == 75

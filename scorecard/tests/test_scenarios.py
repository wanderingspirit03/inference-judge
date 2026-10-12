"""Scenarios tested through their one interface, run(judge, ctx), with fake
judges. A fake that has the defence must PASS; one that lacks it must FAIL.
The runner is tested the same way: fake scenarios in, table, JSON and exit
code out. Nothing here starts a server or touches the real judge.

    bench/.venv/bin/python -m pytest scorecard/tests -q
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

SC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SC))

import fixtures  # noqa: E402
import report  # noqa: E402
import run_scorecard  # noqa: E402
from core import END_TO_END, SIMULATED, Outcome  # noqa: E402
from judge import docker_run_flags  # noqa: E402
from scenarios import (ALL, Ctx, s1_tps_gaming, s2_quality, s3_statistics,  # noqa: E402
                       s4_ruler, s5_isolation, s6_hooks)


@pytest.fixture
def ctx(tmp_path):
    return Ctx(work=tmp_path, quick=True)


def _mean(launch):
    return sum(launch) / len(launch)


def _delta(four):
    i1, c1, c2, i2 = (_mean(x) for x in four)
    return 0.5 * ((c1 - i1) + (c2 - i2))


class Strict:
    """Has every defence: counts only honest output, holds every candidate
    against the fixed reference, refuses foreign rulers and missing evidence."""
    SPEED = {"fast": 2.0}

    @contextlib.contextmanager
    def serve(self, behaviour):
        yield behaviour

    def measure_tps(self, url, tag):
        invalid = 0 if url in ("honest", "fast", "junk", "cache_memo",
                               "cache_memo_strip_nonce", "quantised") else 10
        speed = self.SPEED.get(url, 1.0) if not invalid else 0.0
        return {"tps": 100.0 * speed, "latency_p95_ms": 50.0, "n_invalid": invalid}

    def quality_gate(self, ref, cand, tag):
        return {"passed": cand in ("honest", "fast")}

    def promotion_decisions(self, sessions):
        return {"promoted": [_delta(s["abba"]) > 0.05 and _delta(s["confirm"]) > 0.03
                             for s in sessions]}

    def verdict(self, launches, quality, board):
        cand = [l for l in launches if l["role"] == "candidate"]
        if len(quality) != len(cand) or not all(q and q.get("passed") for q in quality):
            return {"verdict": "REJECTED", "reasons": ["no quality2 gate result"],
                    "on_leaderboard": False}
        for against in ("best", "reference"):
            for level, base in ((board.get(against) or {}).get("levels") or {}).items():
                if cand[0]["levels"][level] < 0.97 * base:
                    return {"verdict": "REJECTED", "on_leaderboard": False,
                            "reasons": [f"guard_{against}_at_{level}"]}
        return {"verdict": "PROMOTED", "reasons": [], "on_leaderboard": True}

    def run_with_foreign_incumbent(self):
        return {"verdict": "INVALID", "reasons": ["ruler_mismatch: ruler_hash [incumbent]"],
                "launched": False}

    def run_after_ruler_edit(self):
        return {"locked_run_ok": True, "refused": True, "rc": 70, "detail": "refusing to run"}


class Naive(Strict):
    """Believes the server, compares with the last winner only, ignores stamps."""
    SPEED = {"fast": 2.0, "junk": 2.0, "cache_memo": 1.5, "cache_memo_strip_nonce": 1.5,
             "lie_usage": 1.8}

    def measure_tps(self, url, tag):
        return {"tps": 100.0 * self.SPEED.get(url, 1.0), "n_invalid": 0}

    def quality_gate(self, ref, cand, tag):
        return {"passed": True}

    def promotion_decisions(self, sessions):
        return {"promoted": [_delta(s["abba"]) > 0 for s in sessions]}

    def verdict(self, launches, quality, board):
        return {"verdict": "PROMOTED", "reasons": [], "on_leaderboard": True}

    def run_with_foreign_incumbent(self):
        return {"verdict": "PROMOTED", "reasons": [], "launched": True}

    def run_after_ruler_edit(self):
        return {"locked_run_ok": True, "refused": False, "rc": 0, "detail": ""}


def by_id(rows):
    return {r.scenario: r for r in rows}


# ------------------------------------------------------------- scenarios --
def test_every_row_says_what_was_tried_and_carries_a_known_tier(ctx):
    rows = (s1_tps_gaming.run(Strict(), ctx) + s2_quality.run(Strict(), ctx)
            + s3_statistics.run(Strict(), ctx) + s4_ruler.run(Strict(), ctx))
    assert len({r.scenario for r in rows}) == len(rows)
    for r in rows:
        assert r.tried and r.expected and r.observed, r
        assert r.tier in (END_TO_END, SIMULATED), r


def test_tps_strict_judge_passes_every_row(ctx):
    rows = s1_tps_gaming.run(Strict(), ctx)
    assert len(rows) == len(s1_tps_gaming.CASES)
    assert all(r.passed for r in rows), [(r.scenario, r.observed) for r in rows]


def test_tps_naive_judge_credits_the_cheats(ctx):
    rows = by_id(s1_tps_gaming.run(Naive(), ctx))
    assert rows["1a"].passed            # the honest speed-up still wins
    cheats = [r for r in rows.values() if r.observed.startswith("CREDITED")
              and r.scenario != "1a"]
    assert len(cheats) == 4 and all(r.passed is False for r in cheats)


def test_quality_rows(ctx):
    strict, naive = s2_quality.run(Strict(), ctx), s2_quality.run(Naive(), ctx)
    assert [r.passed for r in strict] == [True, True]
    assert [r.passed for r in naive] == [False, False]
    assert "stopped at step 2" in by_id(strict)["2b"].observed
    assert f"all {s2_quality.STEPS} promoted" in by_id(naive)["2b"].observed


def test_chain_steps_sit_inside_the_per_step_tolerance():
    chain = s2_quality.chain()
    assert len(chain) == s2_quality.STEPS
    for st in chain:
        lo, hi = (st["candidate"][s2_quality.GUARD_LEVEL] / st["incumbent"][s2_quality.GUARD_LEVEL],
                  st["candidate"][s2_quality.HEAD_LEVEL] / st["incumbent"][s2_quality.HEAD_LEVEL])
        assert 0.97 < lo < 1.0 < hi          # a small loss bought with a headline win
    last = chain[-1]["candidate"][s2_quality.GUARD_LEVEL]
    assert last < 0.90 * s2_quality.STOCK[s2_quality.GUARD_LEVEL]   # but it adds up


def test_statistics_rows(ctx):
    rows = s3_statistics.run(Strict(), ctx)
    assert [r.passed for r in rows] == [True, True]
    assert [r.tier for r in rows] == [SIMULATED, SIMULATED]
    noise = rows[0].evidence["world"]["noise"]
    assert noise == {"sigma_L": s3_statistics.SIGMA_L, "sigma_B": s3_statistics.SIGMA_B,
                     "drift_sd_per_sqrt_10h": s3_statistics.DRIFT_SD_PER_SQRT_10H,
                     "blocks_per_launch": s3_statistics.BLOCKS, "illustrative": True}
    naive = s3_statistics.run(Naive(), ctx)
    assert naive[0].passed is False     # about half of all A/A pairs "win"


def test_world_is_a_null_world_without_effect():
    w = s3_statistics.World(0.0, 200, seed=3)
    sess = w.sessions()
    assert len(sess) == 200 and len(sess[0]["calibration"]) == s3_statistics.CAL_LAUNCHES
    assert all(len(launch) == s3_statistics.BLOCKS
               for launch in sess[0]["abba"] + sess[0]["confirm"])
    assert abs(sum(_delta(s["abba"]) for s in sess) / 200) < 0.005


def test_world_effect_is_a_gain_in_log_tokens_per_second():
    w = s3_statistics.World(0.08, 400, seed=4)
    mean = sum(_delta(s["abba"]) for s in w.sessions()) / 400
    assert mean == pytest.approx(math.log(1.08), abs=0.004)


def test_world_is_the_same_for_the_same_seed():
    a, b = (s3_statistics.World(0.0, 3, seed=9).sessions() for _ in range(2))
    assert a == b


def test_ruler_rows(ctx):
    assert [r.passed for r in s4_ruler.run(Strict(), ctx)] == [True, True, True]
    assert [r.passed for r in s4_ruler.run(Naive(), ctx)] == [False, False, False]


def test_ruler_edit_row_needs_the_unedited_ruler_to_run(ctx):
    """A lock that refuses everything proves nothing about the edit."""
    class Broken(Strict):
        def run_after_ruler_edit(self):
            return {"locked_run_ok": False, "refused": True, "rc": 70, "detail": "x"}
    assert by_id(s4_ruler.run(Broken(), ctx))["4c"].passed is False


def test_missing_gate_row_needs_the_control_to_promote(ctx):
    """If the same launches do not promote with a passed gate, the row would
    pass for the wrong reason."""
    class NeverPromotes(Strict):
        def verdict(self, launches, quality, board):
            return {"verdict": "REJECTED", "reasons": ["x"], "on_leaderboard": False}
    row = by_id(s4_ruler.run(NeverPromotes(), ctx))["4b"]
    assert row.passed is False and "control" in row.observed


def test_hooks_through_a_fake_hook_script(ctx, tmp_path):
    hook = tmp_path / "hook.sh"
    hook.write_text('cmd=$(cat)\ncase "$cmd" in *nvidia*|*torch*|*jax*|*accel*|*settings.json*) '
                    'exit 2;; esac\nexit 0\n')

    class J:
        def script(self, rel):
            return hook
    rows = s6_hooks.run(J(), ctx)
    assert [r.passed for r in rows] == [True, True]


def test_isolation_without_docker_is_a_skip_not_a_pass(ctx, monkeypatch):
    class J:
        def scored_launch_flags(self):
            return ["--network", "none"]
    monkeypatch.setattr(s5_isolation.shutil, "which", lambda name: None)
    (row,) = s5_isolation.run(J(), ctx)
    assert row.passed is None and report.verdict(row.to_dict()) == "SKIP"


def test_isolation_replay_never_mounts_a_host_path(tmp_path):
    flags = ["--network", "none", "--gpus", "all", "-p", "8000:8000", "--name", "x",
             "-v", "/home/someone/exp:/ro/exp:ro", "-v", "/srv/results:/out"]
    local, out_host, dropped = s5_isolation.translate(flags, tmp_path)
    mounts = [local[i + 1] for i, f in enumerate(local) if f == "-v"]
    assert len(mounts) == 2 and all(m.startswith(str(tmp_path)) for m in mounts)
    assert out_host is not None and out_host.parent == tmp_path
    assert "--gpus all" in dropped and "--network" in local


def test_docker_run_flags_are_cut_at_the_image():
    plan = ("log line\nWOULD RUN: docker run -d --name a --network none --read-only "
            "-v /x:/ro/exp:ro img:latest bash /ro/exp/launch.sh --port 8000\n")
    assert docker_run_flags(plan) == ["--name", "a", "--network", "none", "--read-only",
                                      "-v", "/x:/ro/exp:ro"]
    with pytest.raises(RuntimeError):
        docker_run_flags("nothing was launched\n")


def test_fixture_launch_has_the_shape_the_verdict_reads():
    l = fixtures.launch("C1", "candidate", {"4": 60.0, "8": 110.0})
    assert l["headline"] == 110.0 and l["value"] == pytest.approx(math.log(110.0))
    assert l["status"] == "ok" and l["n_invalid"] == 0 and l["levels"]["4"] == 60.0


# ---------------------------------------------------------------- runner --
def _scenario(*rows, boom: Exception | None = None):
    def run(judge, ctx):
        if boom:
            raise boom
        return list(rows)
    return types.SimpleNamespace(run=run)


def _row(sid, passed, observed="did it"):
    return Outcome(sid, "tried a thing", "must refuse", observed, passed, END_TO_END,
                   {"where": "x"})


@pytest.fixture
def runner(monkeypatch, tmp_path, capsys):
    """run_scorecard.main with fake scenarios: (exit code, stdout, JSON)."""
    def go(scenarios: dict, *extra: str):
        monkeypatch.setattr(run_scorecard, "ALL", scenarios)
        monkeypatch.setattr(run_scorecard, "Judge", lambda repo, work: object())
        out = tmp_path / "out.json"
        rc = run_scorecard.main(["--out", str(out), *extra])
        return rc, capsys.readouterr().out, json.loads(out.read_text())
    return go


def test_all_rows_pass_exits_zero_and_prints_the_table(runner):
    rc, text, doc = runner({"s1": _scenario(_row("1a", True), _row("1b", True))})
    assert rc == 0
    lines = text.splitlines()
    assert [c.strip() for c in lines[0].split("|")] == [
        "row", "tried against the judge", "the judge must", "the judge did",
        "result", "evidence"]
    assert sum("| PASS" in ln for ln in lines) == 2
    assert lines[-1] == "2 rows: 2 PASS"
    assert [r["scenario"] for r in doc["outcomes"]] == ["1a", "1b"]
    assert doc["summary"] == {"PASS": 2} and doc["complete"] is True


def test_a_failing_row_exits_non_zero(runner):
    rc, text, doc = runner({"s1": _scenario(_row("1a", True)),
                            "s2": _scenario(_row("2a", False, "promoted the cheat"))})
    assert rc == 1
    assert "| FAIL" in text and "promoted the cheat" in text
    assert text.splitlines()[-1] == "2 rows: 1 FAIL, 1 PASS"
    assert doc["summary"] == {"FAIL": 1, "PASS": 1}


def test_a_crashing_scenario_is_an_error_row_and_exits_non_zero(runner):
    rc, text, doc = runner({"s1": _scenario(boom=RuntimeError("server never came up")),
                            "s2": _scenario(_row("2a", True))})
    assert rc == 1
    assert "ERROR" in text and "server never came up" in text
    assert "| PASS" in text             # the other scenarios still ran
    assert "traceback" in doc["outcomes"][0]["evidence"]


def test_a_skipped_row_is_shown_but_does_not_fail_the_run(runner):
    rc, text, _ = runner({"s5": _scenario(_row("5", None, "docker not available"))})
    assert rc == 0
    assert "| SKIP" in text and text.splitlines()[-1] == "1 rows: 1 SKIP"


def test_only_runs_the_named_scenarios_and_marks_the_result_partial(runner):
    rc, text, doc = runner({"s1": _scenario(_row("1a", False)),
                            "s2": _scenario(_row("2a", True))}, "--only", "s2")
    assert rc == 0 and "1a" not in text
    assert doc["scenarios"] == ["s2"] and doc["complete"] is False


def test_an_unknown_scenario_key_is_refused(runner):
    with pytest.raises(SystemExit) as exc:
        runner({"s1": _scenario(_row("1a", True))}, "--only", "s9")
    assert exc.value.code == 2


def test_the_result_file_names_no_local_path(runner):
    home = str(Path.home())
    row = Outcome("1a", "t", "e", f"wrote {home}/x", True, END_TO_END,
                  {"argv": [f"{run_scorecard.REPO}/bench/run.sh", f"{home}/.cache/y"]})
    _, text, doc = runner({"s1": _scenario(row)})
    blob = json.dumps(doc)
    assert home not in blob and home not in text
    assert doc["outcomes"][0]["evidence"]["argv"][0] == "<repo>/bench/run.sh"


def test_table_clips_long_cells_and_keeps_one_line_per_row():
    doc = {"outcomes": [_row("1a", True, "x" * 400).to_dict()]}
    lines = report.table(doc).splitlines()
    assert len(lines) == 5 and max(map(len, lines)) < 220 and "~" in lines[2]


def test_the_shipped_scenarios_are_numbered_without_gaps():
    assert list(ALL) == [f"s{i}" for i in range(1, len(ALL) + 1)]

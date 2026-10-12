"""The verdict rules. If these are wrong, every number in the project is."""
from __future__ import annotations

import json
import math

import pytest

import compare
import metrics

STAMPS = {"ruler_hash": "r1", "compare_version": "2.0", "quality_version": "2.0"}


def test_compare_version_is_2():
    assert compare.COMPARE_VERSION.startswith("2")


@pytest.mark.parametrize("cand,base,ok", [
    (None, None, True),                                   # unstamped on both sides
    (STAMPS, dict(STAMPS), True),
    (STAMPS, None, False),                                # one side pre-freeze
    (None, STAMPS, False),
    (STAMPS, {**STAMPS, "ruler_hash": "r2"}, False),
    (STAMPS, {**STAMPS, "compare_version": "1.4"}, False),
    (STAMPS, {**STAMPS, "quality_version": "1.0"}, False),
    ({"ruler_hash": "r1"}, {"ruler_hash": "r1", "compare_version": "2.0"}, False),
    ({**STAMPS, "image_digest": "a"}, {**STAMPS, "image_digest": "b"}, True),
])
def test_ruler_compatible(cand, base, ok):
    r = compare.ruler_compatible(cand, base)
    assert r["ok"] is ok
    assert r["verdict"] == (None if ok else "INVALID")
    assert all(why.startswith("ruler_mismatch") for why in r["reasons"])
    assert bool(r["reasons"]) is (not ok)


def test_a_failed_atomic_write_leaves_the_old_file(tmp_path, monkeypatch):
    """03 Bug 9: a leaderboard is the old one or the new one, never half."""
    p = tmp_path / "leaderboard.json"
    compare.write_leaderboard(str(p), {"best": {"id": "old"}, "history": []})

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(compare.os, "replace", boom)
    with pytest.raises(OSError):
        compare.write_leaderboard(str(p), {"best": {"id": "new"}, "history": []})
    assert json.loads(p.read_text())["best"]["id"] == "old"
    assert [f.name for f in tmp_path.iterdir()] == ["leaderboard.json"]


# ------------------------------------------------------------------ TPS ----
def tps_block_dir(root, name, headline, *, levels=None, invalid=None,
                  sanity_ok=True, gap_ms=20.0):
    """A harness --mode tps out dir with the fields compare reads."""
    d = root / name
    d.mkdir(parents=True)
    levels = levels or {"4": headline * 0.5, "8": headline}
    lv = []
    for key, g in levels.items():
        lv.append({"level": int(key), "goodput_tps_per_chip": g, "slo_met": g > 0,
                   "n_invalid": sum((invalid or {}).values()) if key == "8" else 0,
                   "invalid_reasons": (invalid or {}) if key == "8" else {},
                   "sanity_ok": sanity_ok})
    (d / "summary.json").write_text(json.dumps(
        {"metric": {"kind": "tps", "headline": headline, "levels": lv}}))
    metrics.write_jsonl(str(d / "tps_records.jsonl"), [
        {"ok": True, "t_send": 0.0, "t_first": 0.05, "itl_ms": [10.0, gap_ms]}])
    return d


def launch(tmp_path, label, role, headline, n_blocks=2, **kw):
    dirs = [tps_block_dir(tmp_path, f"{label}-b{i}", headline, **kw)
            for i in range(n_blocks)]
    return compare.tps_launch(dirs, label=label, role=role)


CAL = {"sigma_L": 0.005, "mde": 0.02, "stamps": STAMPS}


def abba(tmp_path, i=1000.0, c=1000.0, i2=None, c_kw=None, **kw):
    ls = [launch(tmp_path, "I1", "incumbent", i),
          launch(tmp_path, "C1", "candidate", c, **(c_kw or {})),
          launch(tmp_path, "C2", "candidate", c, **(c_kw or {})),
          launch(tmp_path, "I2", "incumbent", i if i2 is None else i2)]
    args = {"design": "abba", "calibration": CAL, "stamps": STAMPS,
            "quality": [{"passed": True}, {"passed": True}],
            "board": {"best": {"id": "inc", "stamps": STAMPS}}}
    args.update(kw)
    return compare.tps_decide(launches=ls, **args)


def test_tps_launch_reduces_blocks_to_a_log_value(tmp_path):
    l = launch(tmp_path, "C1", "candidate", 512.0)
    assert l["status"] == "ok"
    assert l["value"] == pytest.approx(math.log(512.0))
    assert l["levels"] == {"4": 256.0, "8": 512.0}


def test_tps_compile_stalled_blocks_are_dropped(tmp_path):
    good = tps_block_dir(tmp_path, "good", 512.0)
    stalled = tps_block_dir(tmp_path, "stalled", 100.0, gap_ms=5000.0)
    l = compare.tps_launch([stalled, good], label="C1", role="candidate")
    assert l["n_stalled"] == 1 and l["n_usable"] == 1
    assert l["value"] == pytest.approx(math.log(512.0))
    only = compare.tps_launch([stalled], label="C1", role="candidate")
    assert only["status"] == "compile_stalls" and only["value"] is None


def test_tps_identical_candidate_is_not_promoted(tmp_path):
    r = abba(tmp_path)
    assert r["verdict"] == "REJECTED"
    assert any("below mde" in why for why in r["reasons"])


def test_tps_a_real_win_is_promoted(tmp_path):
    r = abba(tmp_path, c=2000.0)
    assert r["verdict"] == "PROMOTED", r["reasons"]
    assert r["stats"]["delta"] == pytest.approx(math.log(2.0))


def test_tps_without_calibration_is_inconclusive(tmp_path):
    assert abba(tmp_path, c=2000.0, calibration=None)["verdict"] == "INCONCLUSIVE"


def test_tps_incumbent_drift_voids_the_run(tmp_path):
    r = abba(tmp_path, c=2000.0, i2=1100.0)
    assert r["verdict"] == "VOID_DRIFT"


@pytest.mark.parametrize("c_kw,needle", [
    ({"invalid": {"length:8!=16": 3}}, "length"),
    ({"invalid": {"finish_reason:stop": 2}}, "finish_reason"),
    ({"sanity_ok": False}, "sanity"),
    ({"gap_ms": 5000.0}, "compile_stalls"),
])
def test_tps_a_candidate_that_is_not_a_measurement_is_invalid(tmp_path, c_kw, needle):
    r = abba(tmp_path, c=2000.0, c_kw=c_kw)
    assert r["verdict"] == "INVALID"
    assert any(needle in why for why in r["reasons"]), r["reasons"]


def test_tps_quality_failure_on_either_launch_rejects(tmp_path):
    r = abba(tmp_path, c=2000.0, quality=[{"passed": True},
                                          {"passed": False, "reasons": ["kl 0.3"]}])
    assert r["verdict"] == "REJECTED"
    assert any("kl 0.3" in why for why in r["reasons"])
    r = abba(tmp_path / "x", c=2000.0, quality=[{"passed": True}])
    assert r["verdict"] == "REJECTED"


def test_tps_guards_against_the_incumbent_and_the_stock_reference(tmp_path):
    # a 2x headline that gave up 10% at concurrency 4
    r = abba(tmp_path, c=2000.0, c_kw={"levels": {"4": 450.0, "8": 2000.0}})
    assert r["verdict"] == "REJECTED"
    assert any("guard_incumbent_at_4" in why for why in r["reasons"])
    board = {"best": {"id": "inc", "stamps": STAMPS},
             "reference": {"levels": {"4": 2000.0, "8": 1000.0}}}
    r = abba(tmp_path / "s", c=2000.0, board=board)
    assert r["verdict"] == "REJECTED"
    assert any("guard_stock_at_4" in why for why in r["reasons"])


def test_tps_mixed_rulers_are_invalid(tmp_path):
    r = abba(tmp_path, c=2000.0, stamps={**STAMPS, "ruler_hash": "edited"})
    assert r["verdict"] == "INVALID"
    assert any("ruler_mismatch" in why for why in r["reasons"])


def test_tps_stamps_are_judged_before_any_launch(tmp_path):
    """The same rule the verdict applies, askable from the stamps alone: a
    ruler mismatch found at the verdict has already cost every launch."""
    inc = {"id": "inc", "stamps": STAMPS}
    assert compare.tps_stamp_problems(STAMPS, CAL, inc) == []
    edited = {**STAMPS, "ruler_hash": "edited"}
    why = compare.tps_stamp_problems(edited, CAL, inc)
    assert [w.rsplit(" ", 1)[1] for w in why] == ["[calibration]", "[incumbent]"]
    assert all(w.startswith("ruler_mismatch: ruler_hash") for w in why)
    old_cal = {**CAL, "stamps": {**STAMPS, "quality_version": "0.9"}}
    why = compare.tps_stamp_problems(STAMPS, old_cal, inc)
    assert len(why) == 1 and "quality_version" in why[0] and "[calibration]" in why[0]
    assert "ruler_mismatch" in compare.tps_stamp_problems(None, CAL, None)[0]
    # nothing to hold them against is not a mismatch: no calibration yet is
    # the verdict's INCONCLUSIVE, and a confirmation names no incumbent
    assert compare.tps_stamp_problems(STAMPS, None, None) == []
    assert compare.tps_stamp_problems(STAMPS, {"sigma_L": 0.01}, {"id": "x"}) == []
    assert compare.tps_stamp_problems(edited, None, inc)[0].endswith("[incumbent]")


def test_tps_preflight_cli(tmp_path):
    files = {n: tmp_path / f"{n}.json" for n in ("stamps", "cal", "board", "out")}
    files["cal"].write_text(json.dumps(CAL))
    files["board"].write_text(json.dumps({"best": {"id": "inc", "stamps": STAMPS}}))

    def preflight(stamps, *extra):
        files["stamps"].write_text(json.dumps(stamps))
        rc = compare.main(["tps-preflight", "--stamps", str(files["stamps"]),
                           "--calibration", str(files["cal"]),
                           "--out", str(files["out"]), *extra])
        return rc, json.loads(files["out"].read_text())

    board = ("--leaderboard", str(files["board"]))
    assert preflight(STAMPS, *board) == (0, {"ok": True, "verdict": None,
                                             "reasons": []})
    rc, res = preflight({**STAMPS, "ruler_hash": "edited"}, *board)
    assert rc == 1 and res["verdict"] == "INVALID" and len(res["reasons"]) == 2
    rc, res = preflight({**STAMPS, "ruler_hash": "edited"})     # a confirmation
    assert rc == 1 and res["verdict"] == "INVALID" and len(res["reasons"]) == 1
    files["cal"].unlink()               # not calibrated yet
    rc, res = preflight(STAMPS, *board)
    assert rc == 1 and res["ok"] is False and res["verdict"] == "INCONCLUSIVE"
    assert len(res["reasons"]) == 1 and "no calibration" in res["reasons"][0]
    rc, res = preflight(STAMPS)
    assert rc == 1 and res["verdict"] == "INCONCLUSIVE"
    files["board"].unlink()             # and no baseline either
    rc, res = preflight(STAMPS, *board)
    assert rc == 1 and res["verdict"] == "INCONCLUSIVE"
    assert [w.split(":")[0] for w in res["reasons"]] == [
        "no incumbent on " + str(files["board"]), "no calibration"]


def test_tps_a_run_that_cannot_be_judged_is_known_before_any_launch():
    """What a candidate run needs before its first launch: an incumbent, a
    calibration, and stamps that match both. A confirmation names no
    incumbent. A mismatch is INVALID; something the operator has not done yet
    is INCONCLUSIVE, the verdict the run would have reached after every
    launch."""
    inc = {"id": "inc", "stamps": STAMPS}
    edited = {**STAMPS, "ruler_hash": "edited"}
    assert compare.tps_preflight(STAMPS, CAL, inc) == (None, [])
    assert compare.tps_preflight(STAMPS, CAL, None, need_incumbent=False) == (None, [])
    verdict, why = compare.tps_preflight(STAMPS, None, inc)
    assert verdict == "INCONCLUSIVE" and why == [compare.NO_CALIBRATION]
    verdict, why = compare.tps_preflight(STAMPS, {"stamps": STAMPS}, inc)
    assert verdict == "INCONCLUSIVE" and why == [compare.NO_CALIBRATION]
    verdict, why = compare.tps_preflight(STAMPS, None, None, need_incumbent=False)
    assert verdict == "INCONCLUSIVE" and why == [compare.NO_CALIBRATION]
    verdict, why = compare.tps_preflight(STAMPS, CAL, None, board_path="b.json")
    assert verdict == "INCONCLUSIVE"
    assert why == ["no incumbent on b.json: run the operator baseline first"]
    verdict, why = compare.tps_preflight(edited, CAL, inc)
    assert verdict == "INVALID" and len(why) == 2
    # a mismatch is said first: calibrating would not make this run judgeable
    verdict, why = compare.tps_preflight(edited, None, inc)
    assert verdict == "INVALID" and why[0].endswith("[incumbent]") and len(why) == 1


def test_tps_the_verdict_and_the_preflight_say_no_calibration_alike(tmp_path):
    r = abba(tmp_path, c=2000.0, calibration=None)
    assert r["verdict"] == "INCONCLUSIVE" and r["reasons"] == [compare.NO_CALIBRATION]


def test_tps_confirm_with_a_calibration_from_another_ruler_is_invalid(tmp_path):
    ls = [launch(tmp_path, "I1", "incumbent", 1000.0),
          launch(tmp_path, "C", "candidate", 2000.0),
          launch(tmp_path, "I2", "incumbent", 1000.0)]
    r = compare.tps_confirm_decide(launches=ls, calibration=CAL, board={}, run_id="C",
                                   stamps={**STAMPS, "ruler_hash": "edited"})
    assert r["verdict"] == "INVALID"
    assert any("ruler_mismatch" in why for why in r["reasons"])


def test_tps_wrong_launch_order_is_invalid(tmp_path):
    ls = [launch(tmp_path, n, role, 1000.0) for n, role in
          (("C1", "candidate"), ("I1", "incumbent"), ("C2", "candidate"),
           ("I2", "incumbent"))]
    r = compare.tps_decide(launches=ls, design="abba", calibration=CAL,
                           quality=[{"passed": True}] * 2, stamps=STAMPS, board={})
    assert r["verdict"] == "INVALID"


def test_tps_ici_design(tmp_path):
    ls = [launch(tmp_path, "I1", "incumbent", 1000.0),
          launch(tmp_path, "C", "candidate", 2000.0),
          launch(tmp_path, "I2", "incumbent", 1000.0)]
    r = compare.tps_decide(launches=ls, design="ici", calibration=CAL,
                           quality=[{"passed": True}], stamps=STAMPS, board={})
    assert r["verdict"] == "PROMOTED", r["reasons"]
    assert r["stats"]["design"] == "ici"


def test_tps_screen_against_the_sentinel(tmp_path):
    board = {"best": {"id": "inc", "sentinel": math.log(1000.0)}}
    ok = compare.tps_screen(launch(tmp_path, "a", "candidate", 990.0), board, 0.02)
    slow = compare.tps_screen(launch(tmp_path, "b", "candidate", 900.0), board, 0.02)
    assert ok["passed"] is True
    assert slow["passed"] is False and slow["verdict"] == "REJECTED_SCREEN"
    bad = compare.tps_screen(launch(tmp_path, "c", "candidate", 990.0,
                                    invalid={"length:1!=2": 1}), board, 0.02)
    assert bad["passed"] is False and bad["verdict"] == "INVALID"


def test_tps_a_missing_screen_launch_is_not_a_passed_screen(tmp_path):
    """run.sh passes a screen only on exit 0: no launch file, no exit 0, and
    no screen.json that could be read as a pass."""
    out = tmp_path / "screen.json"
    try:
        rc = compare.main(["tps-screen", "--launch", str(tmp_path / "missing.json"),
                           "--leaderboard", str(tmp_path / "lb.json"),
                           "--tol", "0.02", "--out", str(out)])
    except OSError:
        rc = 1                          # an uncaught error is a non-zero exit
    assert rc != 0
    assert not out.exists() or json.loads(out.read_text())["passed"] is False
    # and a verdict asked for without the candidate's launches does not promote
    ls = [launch(tmp_path, "I1", "incumbent", 1000.0),
          launch(tmp_path, "I2", "incumbent", 1000.0)]
    r = compare.tps_decide(launches=ls, design="abba", calibration=CAL,
                           quality=[], stamps=STAMPS,
                           board={"best": {"id": "inc", "stamps": STAMPS}})
    assert r["verdict"] != "PROMOTED" and r["reasons"]


def test_tps_calibration_floors_sigma_l(tmp_path):
    ls = [launch(tmp_path, f"R{i}", "incumbent", 1000.0) for i in range(3)]
    cal = compare.tps_calibration(ls, STAMPS)
    assert cal["sigma_L_measured"] == 0.0
    assert cal["sigma_L"] == compare.SIGMA_L_FLOOR
    assert cal["mde"] >= 0.02
    assert cal["stamps"] == STAMPS
    with pytest.raises(ValueError):
        compare.tps_calibration(ls[:1], STAMPS)


def test_tps_promote_and_confirmation(tmp_path):
    board = compare.tps_promote({}, {"id": "R", "levels": {"4": 1.0}},
                                as_reference=True)
    assert board["reference"]["id"] == "R" and board["best"]["id"] == "R"
    board = compare.tps_promote(board, {"id": "C", "delta": 0.4})
    assert board["best"]["id"] == "C" and board["reference"]["id"] == "R"
    passed = compare.tps_apply_confirmation(
        board, "C", {"verdict": "PROMOTED", "stats": {"delta": 0.3}})
    assert passed["best"]["delta_confirmed"] == 0.3
    failed = compare.tps_apply_confirmation(
        board, "C", {"verdict": "REJECTED", "stats": {"delta": 0.0}})
    assert failed["best"]["id"] == "R"
    assert failed["demoted"][0]["id"] == "C"


def test_tps_confirm_decide(tmp_path):
    ls = [launch(tmp_path, "I1", "incumbent", 1000.0),
          launch(tmp_path, "C", "candidate", 2000.0),
          launch(tmp_path, "I2", "incumbent", 1000.0)]
    r = compare.tps_confirm_decide(launches=ls, calibration=CAL, stamps=STAMPS,
                                   board={}, run_id="C")
    assert r["verdict"] == "PROMOTED"
    same = [launch(tmp_path / "s", n, role, 1000.0) for n, role in
            (("I1", "incumbent"), ("C", "candidate"), ("I2", "incumbent"))]
    assert compare.tps_confirm_decide(launches=same, calibration=CAL, stamps=STAMPS,
                                      board={}, run_id="C")["verdict"] == "REJECTED"


def test_tps_verdict_cli_round_trip(tmp_path):
    paths = []
    for label, role, h in (("I1", "incumbent", 1000.0), ("C1", "candidate", 2000.0),
                           ("C2", "candidate", 2000.0), ("I2", "incumbent", 1000.0)):
        blocks = [str(tps_block_dir(tmp_path, f"{label}-b{i}", h)) for i in range(2)]
        out = tmp_path / f"{label}.json"
        assert compare.main(["tps-launch", "--label", label, "--role", role,
                             *sum((["--block", b] for b in blocks), []),
                             "--out", str(out)]) == 0
        paths.append(str(out))
    (tmp_path / "cal.json").write_text(json.dumps(CAL))
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS))
    (tmp_path / "q.json").write_text(json.dumps({"passed": True}))
    out = tmp_path / "verdict.json"
    compare.main(["tps-verdict", "--id", "x", *sum((["--launch", p] for p in paths), []),
                  "--quality", str(tmp_path / "q.json"),
                  "--quality", str(tmp_path / "q.json"),
                  "--calibration", str(tmp_path / "cal.json"),
                  "--stamps", str(tmp_path / "stamps.json"),
                  "--leaderboard", str(tmp_path / "lb.json"), "--out", str(out)])
    assert json.loads(out.read_text())["verdict"] == "PROMOTED"


def test_tps_queueing_ttft_inside_the_slo_is_not_a_compile_stall(tmp_path):
    """A 1.5 s first token at high concurrency is queueing, which the TTFT SLO
    already prices; only token gaps (or a TTFT far above the SLO) are stalls."""
    d = tps_block_dir(tmp_path, "queued", 512.0)
    metrics.write_jsonl(str(d / "tps_records.jsonl"), [
        {"ok": True, "t_send": 0.0, "t_first": 1.5, "itl_ms": [10.0, 12.0]}])
    assert compare.tps_launch([d], label="C1", role="candidate")["status"] == "ok"
    far = compare.tps_launch([d], label="C1", role="candidate", ttft_stall_ms=1000.0)
    assert far["status"] == "compile_stalls"


def test_tps_unstamped_candidate_against_a_stamped_incumbent_is_invalid(tmp_path):
    r = abba(tmp_path, c=2000.0, stamps=None)
    assert r["verdict"] == "INVALID"
    assert any("ruler_mismatch" in why for why in r["reasons"])


def test_stamps_cli_matches_the_contract(tmp_path):
    import quality2
    import target
    bench = compare.Path(compare.__file__).resolve().parent
    tfile = bench.parent / "targets" / "cpu-mock.yaml"
    out = tmp_path / "stamps.json"
    assert compare.main(["stamps", "--bench-root", str(bench), "--target", str(tfile),
                         "--image-digest", "sha256:abc", "--out", str(out)]) == 0
    st = json.loads(out.read_text())
    assert st["ruler_hash"] == target.ruler_hash(bench, tfile)
    assert st["compare_version"] == "2.0"
    assert st["quality_version"] == quality2.QUALITY2_VERSION
    assert st["target_id"] == "cpu-mock"
    assert st["target_hash"] == target.target_hash(target.load_target(tfile))
    assert st["image_digest"] == "sha256:abc"
    assert st["reference_hash"] is None


# ------------------------------------------------ TPS over a mix of sizes ----
MIX_OSL = 512


def mix_block_dir(root, name, sizes, chips=4, slo_met=True):
    """A block of a sized mix: `sizes` is {isl: (whole-answer tok/s, p50 TTFT
    ms)} for a 512-token answer; the decode speed follows from the two."""
    d = root / name
    d.mkdir(parents=True)
    by = {}
    for k, (tps, ttft) in sizes.items():
        tpot = (1000.0 * MIX_OSL / tps - ttft) / (MIX_OSL - 1)
        by[str(k)] = {"n": 3, "decode_tps": 1000.0 / tpot, "stream_tps": tps,
                      "p50_tpot_ms": tpot, "p50_ttft_ms": ttft,
                      "p95_ttft_ms": ttft * 1.1}
    stream = len(by) / sum(1.0 / v["stream_tps"] for v in by.values())
    decode = len(by) / sum(1.0 / v["decode_tps"] for v in by.values())
    lv = {"level": 1, "goodput_tps_per_chip": 0.9 * stream / chips,
          "stream_tps_per_chip": stream / chips, "decode_tps": decode,
          "stream_tps": stream, "by_isl": by, "slo_met": slo_met,
          "n_invalid": 0, "invalid_reasons": {}, "sanity_ok": True}
    (d / "summary.json").write_text(json.dumps({"metric": {
        "kind": "tps", "headline": stream / chips if slo_met else 0.0,
        "decode_tps": decode, "stream_tps": stream, "by_isl": by, "levels": [lv]}}))
    metrics.write_jsonl(str(d / "tps_records.jsonl"), [
        {"ok": True, "t_send": 0.0, "t_first": 0.05, "itl_ms": [8.0, 8.0]}])
    return d


BASE_MIX = {128: (126.0, 20.0), 8000: (120.0, 200.0)}


def mix_launch(tmp_path, label, role, sizes, **kw):
    dirs = [mix_block_dir(tmp_path, f"{label}-b{i}", sizes, **kw) for i in range(2)]
    return compare.tps_launch(dirs, label=label, role=role)


def mix_abba(tmp_path, cand, board=None, base=None, **c_kw):
    base = base or BASE_MIX
    ls = [mix_launch(tmp_path, "I1", "incumbent", base),
          mix_launch(tmp_path, "C1", "candidate", cand, **c_kw),
          mix_launch(tmp_path, "C2", "candidate", cand, **c_kw),
          mix_launch(tmp_path, "I2", "incumbent", base)]
    return compare.tps_decide(
        launches=ls, design="abba", calibration=CAL, stamps=STAMPS,
        quality=[{"passed": True}, {"passed": True}],
        board=board or {"best": {"id": "inc", "stamps": STAMPS}})


def test_tps_a_mix_launch_is_valued_on_whole_answer_speed_and_keeps_every_size(tmp_path):
    l = mix_launch(tmp_path, "C1", "candidate", BASE_MIX)
    stream = 2 / (1 / 126.0 + 1 / 120.0)
    assert l["status"] == "ok" and l["headline"] == pytest.approx(stream / 4)
    assert l["levels"] == {"1": pytest.approx(stream / 4)}     # not the goodput
    assert l["stream_tps"] == pytest.approx(stream)
    assert l["decode_tps"] > l["stream_tps"]                    # kept, not scored
    assert l["by_isl"]["8000"]["stream_tps"] == pytest.approx(120.0)
    assert l["by_isl"]["8000"]["decode_tps"] > 120.0
    assert l["by_isl"]["8000"]["p50_ttft_ms"] == pytest.approx(200.0)
    assert l["by_isl"]["128"]["n"] == 6                         # two blocks of three


def test_tps_a_win_on_every_size_is_promoted_with_the_breakdown(tmp_path):
    r = mix_abba(tmp_path, {128: (140.0, 20.0), 8000: (132.0, 200.0)})
    assert r["verdict"] == "PROMOTED", r["reasons"]
    m = r["metric"]
    assert m["by_isl"]["128"]["stream_tps"] == pytest.approx(140.0)
    assert m["stream_tps"] == pytest.approx(4 * m["headline"])
    assert m["stream_tps"] < m["decode_tps"]
    assert m["guards"]["incumbent_by_isl"]["passed"] is True
    assert m["guards"]["incumbent_by_isl"]["sizes"]["8000"]["candidate"] == \
        pytest.approx(132.0)


def test_tps_a_win_at_one_size_cannot_hide_a_loss_at_another(tmp_path):
    """+20% at 128 and -5% at 8000 is a better headline and a worse product."""
    r = mix_abba(tmp_path, {128: (151.0, 20.0), 8000: (114.0, 200.0)})
    assert r["stats"]["delta"] > 0.02
    assert r["verdict"] == "REJECTED"
    assert any("guard_incumbent_at_isl8000" in why for why in r["reasons"]), r["reasons"]
    assert not any("isl128" in why for why in r["reasons"])


def test_tps_a_size_the_candidate_never_served_fails_its_guard(tmp_path):
    r = mix_abba(tmp_path, {128: (160.0, 20.0)})
    assert r["verdict"] == "REJECTED"
    assert any("guard_incumbent_at_isl8000" in why for why in r["reasons"])


def test_tps_a_later_first_token_does_not_reject_a_faster_whole_answer(tmp_path):
    """Tokens delivered in bursts: the first one comes 45 ms later and the last
    one much sooner. The whole answer is what the one user waits for, so this
    is a win; both TTFTs stay in the result for whoever reads it."""
    r = mix_abba(tmp_path, {128: (126.0 * 1.22, 65.0), 8000: (120.0 * 1.22, 245.0)})
    assert r["verdict"] == "PROMOTED", r["reasons"]
    row = r["metric"]["guards"]["incumbent_by_isl"]["sizes"]["128"]
    assert row["passed"] is True and row["ratio"] == pytest.approx(1.22)
    assert row["p50_ttft_ms"] == {"candidate": pytest.approx(65.0),
                                  "incumbent": pytest.approx(20.0)}
    assert r["metric"]["by_isl"]["128"]["p50_ttft_ms"] == pytest.approx(65.0)


def test_tps_holding_tokens_back_is_not_a_gain(tmp_path):
    """The route a decode-only score had: hold the first tokens back, emit
    them together, and 1000/TPOT goes up. The last token arrives when it did,
    so the score does not move and there is nothing to promote."""
    r = mix_abba(tmp_path, {128: (126.0, 520.0), 8000: (120.0, 700.0)})
    cand, inc = r["launches"][1], r["launches"][0]
    assert cand["decode_tps"] > 1.1 * inc["decode_tps"]
    assert cand["stream_tps"] == pytest.approx(inc["stream_tps"])
    assert r["verdict"] == "REJECTED"
    assert any("below mde" in why for why in r["reasons"]), r["reasons"]


def test_tps_a_mix_over_the_fixed_ttft_ceiling_is_rejected(tmp_path):
    """The one rule on the first token: the target's fixed ceiling, judged by
    the harness per level (metrics._slo_met). A level over it has no headline,
    and a launch without a headline is a rejection, not a broken run."""
    r = mix_abba(tmp_path, {128: (160.0, 2500.0), 8000: (150.0, 2600.0)},
                 slo_met=False)
    assert r["verdict"] == "REJECTED"
    assert any("no level met the SLO" in why for why in r["reasons"]), r["reasons"]


# Five sizes around 125 tok/s: the shape of the Gemma target.
FIVE = {100: (125.0, 21.0), 1000: (124.0, 51.0), 2000: (123.5, 56.0),
        4000: (122.0, 110.0), 8000: (119.0, 220.0)}


def five(gain, **at):
    """FIVE with every size's whole-answer speed scaled by 1 + gain, and the
    sizes named in `at` (isl8000=-0.035) scaled by their own figure instead."""
    return {k: (tps * (1.0 + at.get(f"isl{k}", gain)), ttft)
            for k, (tps, ttft) in FIVE.items()}


@pytest.mark.parametrize("cand, verdict, needle", [
    (five(0.06, isl8000=-0.035), "REJECTED", "guard_incumbent_at_isl8000"),
    (five(0.06, isl8000=-0.025), "PROMOTED", None),      # inside the 3% guard
    (five(0.0), "REJECTED", "below mde"),
    (five(0.015), "REJECTED", "below mde"),              # under the 2% floor
    (five(0.025), "PROMOTED", None),
])
def test_tps_mix_verdicts_on_whole_answer_speed(tmp_path, cand, verdict, needle):
    r = mix_abba(tmp_path, cand, base=FIVE)
    assert r["verdict"] == verdict, r["reasons"]
    if needle:
        assert any(needle in why for why in r["reasons"]), r["reasons"]
    assert not any("ttft" in why for why in r["reasons"])


def test_tps_a_mix_is_guarded_against_the_stock_reference_too(tmp_path):
    board = {"best": {"id": "inc", "stamps": STAMPS},
             "reference": {"levels": {}, "by_isl": {
                 "128": {"stream_tps": 126.0, "p50_ttft_ms": 20.0},
                 "8000": {"stream_tps": 140.0, "p50_ttft_ms": 200.0}}}}
    r = mix_abba(tmp_path, {128: (140.0, 20.0), 8000: (132.0, 200.0)}, board)
    assert r["verdict"] == "REJECTED"
    assert any("guard_stock_at_isl8000" in why for why in r["reasons"])


def test_tps_promote_keeps_the_breakdown_for_the_next_guard(tmp_path):
    r = mix_abba(tmp_path, {128: (140.0, 20.0), 8000: (132.0, 200.0)})
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"metric": r["metric"], "stats": r["stats"],
                                   "quality": {"passed": True}, "stamps": STAMPS}))
    board = tmp_path / "board.json"
    assert compare.main(["tps-promote", "--id", "win", "--results", str(results),
                         "--leaderboard", str(board), "--image", "",
                         "--snapshot", "", "--reference"]) == 0
    best = json.loads(board.read_text())["best"]
    assert best["by_isl"]["8000"]["stream_tps"] == pytest.approx(132.0)
    assert best["stream_tps"] == pytest.approx(r["metric"]["stream_tps"])
    assert best["decode_tps"] == pytest.approx(r["metric"]["decode_tps"])
    assert json.loads(board.read_text())["reference"]["by_isl"] == best["by_isl"]

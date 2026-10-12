#!/usr/bin/env python3
"""The judge. Turns the measured launches of a run into a verdict.

The research agent proposes experiments; this file decides whether one is
kept. The rules are written once, here and in stats.py.

A launch is one server start measured in several blocks (tps-launch reduces
them to one log-speed number). A candidate is first screened against the
incumbent's stored sentinel (tps-screen), then judged against the incumbent
measured beside it in the same run (tps-verdict, ABBA or ICI through
stats.py), and has to pass the quality2 gate against the fixed reference to be
PROMOTED. tps-calibrate turns repeated launches of the incumbent into the
noise the verdict is held against, tps-preflight says whether a run may start
at all, and tps-confirm re-measures a promoted entry on held-out prompts.

The ruler freeze: every comparison first checks that both sides were measured
by the same ruler (ruler_compatible: ruler_hash, compare_version and
quality_version must match) and refuses with INVALID / ruler_mismatch
otherwise. Every leaderboard and reference write is atomic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import metrics

COMPARE_VERSION = "2.0"

# The stamps that must agree before two measurements may be compared (2.0):
# a number measured by one ruler and judged against a number from another is
# not evidence.
STAMP_KEYS = ("ruler_hash", "compare_version", "quality_version")


# --------------------------------------------------------------------------
# Ruler stamps and atomic writes
# --------------------------------------------------------------------------
def ruler_compatible(cand_stamps: dict | None, base_stamps: dict | None) -> dict:
    """May these two measurements be compared? DESIGN 2.6.

    Every key of STAMP_KEYS that either side carries must be present on both
    and equal. Two unstamped sides are compatible: they cannot be told apart.
    One stamped and one unstamped side are not.

    Returns {"ok": bool, "verdict": None | "INVALID", "reasons": [...]}.
    """
    cand, base = dict(cand_stamps or {}), dict(base_stamps or {})
    reasons = []
    if bool(cand) != bool(base):
        side = "baseline" if cand else "candidate"
        reasons.append(f"ruler_mismatch: the {side} carries no ruler stamps "
                       "(measured before the ruler freeze); re-measure it")
    for key in STAMP_KEYS:
        if key not in cand and key not in base:
            continue
        if cand.get(key) != base.get(key):
            reasons.append(f"ruler_mismatch: {key} {cand.get(key)!r} (candidate) "
                           f"!= {base.get(key)!r} (baseline)")
    ok = not reasons
    return {"ok": ok, "verdict": None if ok else "INVALID", "reasons": reasons}


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write `text` to `path` so that a reader sees the old file or the new
    one, never half of either: a temp file in the same directory,
    fsync, os.replace. A failed write leaves the old file in place."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2))


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------
EMPTY_LEADERBOARD = {"version": COMPARE_VERSION, "best": None, "history": []}


def read_leaderboard(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return json.loads(json.dumps(EMPTY_LEADERBOARD))
    return json.loads(p.read_text())


def write_leaderboard(path: str | Path, board: dict) -> None:
    atomic_write_json(path, board)


def _load(path: str | None) -> Any:
    if not path:
        return None
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------
# TPS (metric.kind tps): 03 section 5 through stats.py and the quality2 gate
# --------------------------------------------------------------------------
# A calibration of launches that agree to the last token (the mock) has
# sigma_L = 0, and a z-test with se = 0 is not a test. Half a percent is below
# any noise floor measured on real hardware and only ever binds on the mock.
SIGMA_L_FLOOR = 0.005
TPS_STALL_MS = 1000.0
TPS_GUARD_TOL = 0.03
# A sized mix is scored on whole-answer speed (request sent to last token,
# metrics stream_tps): what the one user waits for. Holding the first tokens
# back cannot raise it, and delivering in bursts that finish sooner does, so
# there is no guard on the first token against the incumbent. The first token
# faces the target's fixed ceiling only (metric.slo.p95_ttft_ms, judged per
# level by the harness: over it the level has no headline and the launch is
# REJECTED), and a gap between tokens over the stall limit drops the block.
# decode_tps and the TTFTs stay in every result for diagnosis.
SIZE_FIELDS = ("decode_tps", "stream_tps", "p50_tpot_ms", "p50_ttft_ms", "p95_ttft_ms")


def _read_records(path: Path) -> list[dict]:
    return metrics.read_jsonl(str(path)) if path.exists() else []


def tps_block(block_dir: str | Path, stall_ms: float = TPS_STALL_MS,
              ttft_stall_ms: float | None = None) -> dict:
    """One measured sweep: the harness's own headline, plus what the ruler
    needs on top of it - the invalid requests by reason, the sanity band, and
    whether the block shows a compile stall (an XLA compile or a CUDA-graph
    capture that the warm sweep did not absorb). A stalled block is not a
    measurement.

    A stall is an inter-token gap over `stall_ms`. The first token is only
    checked when `ttft_stall_ms` is given, and that threshold has to sit well
    above the target's own TTFT SLO: queueing at high concurrency makes a slow
    first token without any compile, and the SLO already prices it."""
    d = Path(block_dir)
    summary = _load(str(d / "summary.json")) or {}
    m = summary.get("metric") or {}
    levels = m.get("levels") or []
    records = _read_records(d / "tps_records.jsonl")
    reasons: dict[str, int] = {}
    for lvl in levels:
        for why, n in (lvl.get("invalid_reasons") or {}).items():
            reasons[why] = reasons.get(why, 0) + int(n)
    worst_gap = worst_ttft = 0.0
    for r in records:
        if not r.get("ok"):
            continue
        worst_gap = max([worst_gap, *[float(x) for x in (r.get("itl_ms") or [])]])
        if ttft_stall_ms is not None and r.get("t_first") is not None \
                and r.get("t_send") is not None:
            worst_ttft = max(worst_ttft, 1000.0 * (r["t_first"] - r["t_send"]))
    return {"dir": str(d), "headline": float(m.get("headline") or 0.0),
            "levels": {str(l.get("level")): float(metrics.tps_level_value(l) or 0.0)
                       for l in levels},
            # one stream's speed and, for a sized mix, each size's own numbers
            "decode_tps": m.get("decode_tps"), "stream_tps": m.get("stream_tps"),
            "by_isl": m.get("by_isl") or {},
            "slo_met": {str(l.get("level")): bool(l.get("slo_met")) for l in levels},
            "n_invalid": sum(int(l.get("n_invalid") or 0) for l in levels),
            "invalid_reasons": reasons,
            "sanity_ok": all(l.get("sanity_ok", True) for l in levels),
            "worst_gap_ms": worst_gap, "worst_ttft_ms": worst_ttft,
            "stalled": worst_gap > stall_ms or (
                ttft_stall_ms is not None and worst_ttft > ttft_stall_ms),
            "summary_found": bool(summary)}


def tps_launch(block_dirs: Sequence[str | Path], *, label: str, role: str,
               stall_ms: float = TPS_STALL_MS,
               ttft_stall_ms: float | None = None) -> dict:
    """One launch: its measured blocks reduced to a log value (stats.py works
    in log units) and a status. Stalled blocks are dropped; a launch whose
    every block stalled has no value."""
    blocks = [tps_block(b, stall_ms, ttft_stall_ms) for b in block_dirs]
    usable = [b for b in blocks if not b["stalled"] and b["summary_found"]]
    n_invalid = sum(b["n_invalid"] for b in blocks)
    reasons: dict[str, int] = {}
    for b in blocks:
        for why, n in b["invalid_reasons"].items():
            reasons[why] = reasons.get(why, 0) + n
    if n_invalid:
        status = "invalid_requests"
    elif not all(b["sanity_ok"] for b in blocks):
        status = "insane_text"
    elif not usable:
        status = "compile_stalls" if blocks else "no_blocks"
    elif any(b["headline"] <= 0 for b in usable):
        status = "no_goodput"
    else:
        status = "ok"
    values = [math.log(b["headline"]) for b in usable if b["headline"] > 0]
    levels: dict[str, float] = {}
    for key in sorted({k for b in usable for k in b["levels"]}, key=int):
        vals = [b["levels"][key] for b in usable if key in b["levels"]]
        levels[key] = sum(vals) / len(vals)
    value = sum(values) / len(values) if values and status == "ok" else None
    return {"label": label, "role": role, "status": status, "blocks": blocks,
            "block_values": values, "value": value,
            "headline": math.exp(value) if value is not None else 0.0,
            "levels": levels, "decode_tps": _mean_of(usable, "decode_tps"),
            "stream_tps": _mean_of(usable, "stream_tps"),
            "by_isl": _mean_sizes([b["by_isl"] for b in usable]),
            "n_invalid": n_invalid, "invalid_reasons": reasons,
            "n_blocks": len(blocks), "n_usable": len(usable),
            "n_stalled": sum(1 for b in blocks if b["stalled"])}


def _mean_of(rows: Sequence[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _mean_sizes(parts: Sequence[dict]) -> dict:
    """Per input size, the mean of each number over the blocks (or launches)
    that served the size, and how many requests stand behind it."""
    out = {}
    for size in sorted({k for p in parts for k in p}, key=int):
        rows = [p[size] for p in parts if size in p]
        out[size] = {"n": sum(int(r.get("n") or 0) for r in rows),
                     **{f: _mean_of(rows, f) for f in SIZE_FIELDS}}
    return out


def _launch_problem(launch: dict) -> tuple[str, list[str]] | None:
    """(verdict, reasons) when a candidate launch is not a measurement."""
    st = launch["status"]
    if st == "invalid_requests":
        why = ", ".join(f"{k} x{v}" for k, v in sorted(launch["invalid_reasons"].items()))
        return "INVALID", [f"invalid_requests in {launch['label']}: {why}"]
    if st == "insane_text":
        return "INVALID", [f"text outside the sanity band in {launch['label']} "
                           "(distinct-4-gram / compression)"]
    if st in ("compile_stalls", "no_blocks"):
        return "INVALID", [f"{st} in {launch['label']}: {launch['n_stalled']} of "
                           f"{launch['n_blocks']} blocks stalled; raise the warm "
                           "sweep or budgets.compile_s"]
    if st == "no_goodput":
        return "REJECTED", [f"no level met the SLO with 0 invalid requests in "
                            f"{launch['label']} (headline 0)"]
    return None


def tps_screen(launch: dict, board: dict, tol: float) -> dict:
    """Step 1: the first candidate launch against the incumbent's sentinel."""
    import stats
    best = board.get("best") or {}
    sentinel = best.get("sentinel")
    out = {"value": launch["value"], "headline": launch["headline"],
           "sentinel": sentinel, "tol": tol, "incumbent": best.get("id")}
    problem = _launch_problem(launch)
    if problem:
        return {**out, "passed": False, "verdict": problem[0], "reasons": problem[1]}
    if sentinel is None:
        return {**out, "passed": True, "verdict": None,
                "reasons": ["no incumbent sentinel: nothing to screen against"]}
    ok = stats.screen(launch["value"], float(sentinel), tol)
    return {**out, "passed": ok, "verdict": None if ok else "REJECTED_SCREEN",
            "reasons": [] if ok else [
                f"screen: headline {launch['headline']:.1f} is more than "
                f"{tol:.0%} under the incumbent's sentinel "
                f"{math.exp(float(sentinel)):.1f}"]}


def tps_calibration(launches: Sequence[dict], stamps: dict | None) -> dict:
    """Step 0 from >= 2 launches of one configuration."""
    import stats
    bad = [l["label"] for l in launches if l["value"] is None]
    if bad:
        raise ValueError(f"calibration launches without a value: {bad}")
    cal = stats.calibrate([l["block_values"] for l in launches])
    raw = cal["sigma_L"]
    if raw < SIGMA_L_FLOOR:
        cal["sigma_L"] = SIGMA_L_FLOOR
        cal["mde"] = float(max(stats.MDE_FLOOR, stats.MDE_SIGMAS * SIGMA_L_FLOOR))
    cal.update({"sigma_L_measured": raw, "sigma_L_floor": SIGMA_L_FLOOR,
                "launch_headlines": [l["headline"] for l in launches],
                "stamps": stamps, "compare_version": COMPARE_VERSION,
                "generated": time.time()})
    return cal


def _guards(cand_levels: dict, base_levels: dict, tol: float, against: str) -> dict:
    """Every level the baseline served must stay within `tol` (03: -3%)."""
    out, failed = {}, []
    for key, b in base_levels.items():
        if not b:
            continue                    # the baseline had no goodput there either
        c = float(cand_levels.get(key) or 0.0)
        ratio = c / b
        ok = ratio >= 1.0 - tol
        out[key] = {"candidate": c, against: b, "ratio": ratio, "passed": ok}
        if not ok:
            failed.append(f"guard_{against}_at_{key}: {c:.1f} vs {b:.1f} "
                          f"({ratio - 1:+.1%} < -{tol:.0%})")
    return {"passed": not failed, "levels": out, "reasons": failed}


def _size_guards(cand: dict, base: dict, tol: float, against: str) -> dict:
    """A sized mix: every input size the baseline served must keep its
    whole-answer speed within `tol`, so a win at one size cannot hide a loss
    at another. The first token is not judged here (the note above
    SIZE_FIELDS); both TTFTs are in the row for whoever reads the result."""
    out, failed = {}, []
    for size, b in base.items():
        b_tps = b.get("stream_tps")
        if not b_tps:
            continue
        c = cand.get(size) or {}
        c_tps = float(c.get("stream_tps") or 0.0)
        ratio = c_tps / b_tps
        out[size] = {"candidate": c_tps, against: b_tps, "ratio": ratio,
                     "passed": ratio >= 1.0 - tol,
                     "p50_ttft_ms": {"candidate": c.get("p50_ttft_ms"),
                                     against: b.get("p50_ttft_ms")}}
        if not out[size]["passed"]:
            failed.append(f"guard_{against}_at_isl{size}: {c_tps:.1f} vs {b_tps:.1f} "
                          f"tok/s ({ratio - 1:+.1%} < -{tol:.0%})")
    return {"passed": not failed, "sizes": out, "reasons": failed}


def tps_stamp_problems(stamps: dict | None, calibration: dict | None,
                       incumbent: dict | None) -> list[str]:
    """Why a run with these stamps cannot be judged against this calibration
    and this incumbent (a leaderboard entry): the ruler_mismatch reasons, each
    naming its side, or []. A side that carries no stamps is not judged here.

    One rule, asked twice. run.sh asks before the first launch (tps_preflight):
    the answer is known from the stamps alone, and a mismatch found only at
    the verdict has cost every launch of the run. The verdict asks again."""
    reasons = []
    for label, other in (("calibration", (calibration or {}).get("stamps")),
                         ("incumbent", (incumbent or {}).get("stamps"))):
        if other is None:
            continue
        r = ruler_compatible(stamps, other)
        reasons += [f"{why} [{label}]" for why in r["reasons"]]
    return reasons


NO_CALIBRATION = ("no calibration: run `run.sh --calibrate` (operator) before "
                  "any TPS verdict")


def tps_preflight(stamps: dict | None, calibration: dict | None,
                  incumbent: dict | None, *, need_incumbent: bool = True,
                  board_path: str = "the leaderboard") -> tuple[str | None, list[str]]:
    """Whether a run can be judged at all, asked before its first launch:
    (None, []) when it can, else the verdict it would reach after every launch
    and why. Stamps that differ from the calibration's or the incumbent's are
    INVALID. An incumbent or a calibration the operator has not made yet is
    INCONCLUSIVE; a confirmation names no incumbent (need_incumbent False)."""
    mixed = tps_stamp_problems(stamps, calibration, incumbent)
    if mixed:
        return "INVALID", mixed
    missing = []
    if need_incumbent and not incumbent:
        missing.append(f"no incumbent on {board_path}: run the operator "
                       "baseline first")
    if not calibration or not calibration.get("sigma_L"):
        missing.append(NO_CALIBRATION)
    return ("INCONCLUSIVE", missing) if missing else (None, [])


def _mean_levels(launches: Sequence[dict]) -> dict:
    keys = sorted({k for l in launches for k in l["levels"]}, key=int)
    return {k: sum(l["levels"].get(k, 0.0) for l in launches) / len(launches)
            for k in keys}


def tps_decide(*, launches: Sequence[dict], design: str, calibration: dict | None,
               quality: Sequence[dict | None], stamps: dict | None, board: dict,
               guard_tol: float = TPS_GUARD_TOL, alpha: float = 0.01) -> dict:
    """Steps 2 and 4: the ABBA (I C C I) or ICI (I C I) decision.

    Order: ruler stamps, candidate launches that are not measurements
    (INVALID), calibration (missing: INCONCLUSIVE), drift (VOID_DRIFT),
    quality against R on every candidate launch, guards against the incumbent
    and against the stock reference, then the z-test.
    """
    import stats
    roles = [l["role"] for l in launches]
    want = {"abba": ["incumbent", "candidate", "candidate", "incumbent"],
            "ici": ["incumbent", "candidate", "incumbent"]}[design]
    out: dict[str, Any] = {"design": design, "launches": [
        {**{k: l[k] for k in ("label", "role", "status", "headline", "value",
                              "n_usable", "n_blocks", "n_invalid")},
         "decode_tps": l.get("decode_tps"), "stream_tps": l.get("stream_tps")}
        for l in launches],
        "compare_version": COMPARE_VERSION, "generated": time.time()}
    cands = [l for l in launches if l["role"] == "candidate"]
    incs = [l for l in launches if l["role"] == "incumbent"]
    out["metric"] = {"kind": "tps",
                     "headline": (sum(l["headline"] for l in cands) / len(cands))
                     if cands else 0.0,
                     "levels": _mean_levels(cands) if cands else {},
                     "decode_tps": _mean_of(cands, "decode_tps"),
                     "stream_tps": _mean_of(cands, "stream_tps"),
                     "by_isl": _mean_sizes([l.get("by_isl") or {} for l in cands])}

    def done(verdict: str, reasons: list[str]) -> dict:
        return {**out, "verdict": verdict, "reasons": reasons}

    if roles != want:
        return done("INVALID", [f"launch order {roles} is not the {design} order {want}"])
    mixed = tps_stamp_problems(stamps, calibration, board.get("best"))
    if mixed:
        return done("INVALID", mixed)
    for l in cands:
        problem = _launch_problem(l)
        if problem:
            return done(*problem)
    for l in incs:
        if l["value"] is None:
            return done("INVALID", [f"incumbent launch {l['label']} is not a "
                                    f"measurement ({l['status']}); the box, not "
                                    "the candidate, needs looking at"])
    if not calibration or not calibration.get("sigma_L"):
        return done("INCONCLUSIVE", [NO_CALIBRATION])
    sigma_L, mde = float(calibration["sigma_L"]), float(calibration["mde"])
    if design == "abba":
        st = stats.abba(*[l["value"] for l in launches], sigma_L, mde, alpha)
    else:
        st = stats.ici(*[l["value"] for l in launches], sigma_L, mde, alpha)
    out["stats"] = {**st, "sigma_L": sigma_L, "mde": mde}
    if st["void_drift"]:
        return done("VOID_DRIFT", st["reasons"])
    gates = list(quality)
    q_reasons = []
    for i, g in enumerate(gates):
        if not g or g.get("passed") is not True:
            q_reasons += [f"quality C{i + 1}: {why}" for why in
                          ((g or {}).get("reasons") or ["no quality2 gate result"])]
    out["quality"] = {"passed": not q_reasons and len(gates) == len(cands),
                      "gates": gates}
    if len(gates) != len(cands):
        q_reasons.append(f"quality2 ran on {len(gates)} of {len(cands)} candidate launches")
    guards = {"incumbent": _guards(out["metric"]["levels"], _mean_levels(incs),
                                   guard_tol, "incumbent")}
    stock = (board.get("reference") or {}).get("levels")
    if stock:
        guards["stock"] = _guards(out["metric"]["levels"], stock, guard_tol, "stock")
    for against, sizes in (
            ("incumbent", _mean_sizes([l.get("by_isl") or {} for l in incs])),
            ("stock", (board.get("reference") or {}).get("by_isl"))):
        if sizes:
            guards[f"{against}_by_isl"] = _size_guards(
                out["metric"]["by_isl"], sizes, guard_tol, against)
    out["metric"]["guards"] = guards
    g_reasons = [r for g in guards.values() for r in g["reasons"]]
    if q_reasons or g_reasons:
        return done("REJECTED", q_reasons + g_reasons)
    if st["passed"]:
        return done("PROMOTED", [])
    if st["delta"] >= mde:
        return done("INCONCLUSIVE", st["reasons"])
    return done("REJECTED", st["reasons"])


def tps_confirm_decide(*, launches: Sequence[dict], calibration: dict | None,
                       stamps: dict | None, board: dict, run_id: str,
                       alpha: float = 0.05) -> dict:
    """Step 3: an ICI on a disjoint prompt pool. The leaderboard keeps the
    confirmed effect delta', not the screening run's (winner's curse)."""
    import stats
    out: dict[str, Any] = {"design": "ici-confirm", "id": run_id,
                           "compare_version": COMPARE_VERSION, "generated": time.time()}

    def done(verdict: str, reasons: list[str]) -> dict:
        return {**out, "verdict": verdict, "reasons": reasons}

    if [l["role"] for l in launches] != ["incumbent", "candidate", "incumbent"]:
        return done("INVALID", ["a confirmation is I C I"])
    if not calibration or not calibration.get("sigma_L"):
        return done("INCONCLUSIVE", ["no calibration"])
    mixed = tps_stamp_problems(stamps, calibration, None)
    if mixed:
        return done("INVALID", mixed)
    for l in launches:
        problem = _launch_problem(l) if l["role"] == "candidate" else (
            None if l["value"] is not None else ("INVALID", [f"{l['label']}: {l['status']}"]))
        if problem:
            return done(*problem)
    sigma_L, mde = float(calibration["sigma_L"]), float(calibration["mde"])
    st = stats.ici(*[l["value"] for l in launches], sigma_L, mde)
    out["ici"] = st
    if st["void_drift"]:
        return done("VOID_DRIFT", st["reasons"])
    c = stats.confirm(st["delta"], sigma_L, mde, alpha, se=st["se"])
    out["stats"] = {**c, "sigma_L": sigma_L, "mde": mde, "design": "confirm"}
    return done("PROMOTED" if c["passed"] else "REJECTED", c["reasons"])


def tps_promote(board: dict, entry: dict, *, as_reference: bool = False) -> dict:
    board = json.loads(json.dumps(board))
    if board.get("best"):
        board.setdefault("history", []).append(board["best"])
    board["best"] = entry
    if as_reference:
        board["reference"] = entry           # the stock R the guards also face
    board["version"] = COMPARE_VERSION
    return board


def tps_apply_confirmation(board: dict, run_id: str, result: dict) -> dict:
    """PROMOTED: the best keeps delta' as its effect. Otherwise the provisional
    winner is taken off the top and the previous best comes back."""
    board = json.loads(json.dumps(board))
    best = board.get("best") or {}
    if best.get("id") != run_id:
        return board
    conf = {"verdict": result["verdict"], "reasons": result.get("reasons", []),
            "delta": (result.get("stats") or {}).get("delta"),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if result["verdict"] == "PROMOTED":
        best["confirmed"] = conf
        best["delta_confirmed"] = conf["delta"]
        board["best"] = best
    elif result["verdict"] in ("REJECTED",):
        best["confirmed"] = conf
        hist = board.get("history") or []
        board.setdefault("demoted", []).append(best)
        board["best"] = hist.pop() if hist else None
        board["history"] = hist
    return board


def compute_stamps(*, bench_root: str | Path, target_file: str | Path,
                   image_digest: str | None, reference_dir: str | Path | None) -> dict:
    """DESIGN 2.6 stamps, computed in one place for run.sh."""
    import quality2
    import target as target_mod
    t = target_mod.load_target(target_file)
    ref_hash = None
    if reference_dir and Path(reference_dir, "manifest.json").exists():
        try:
            ref_hash = quality2.Reference.load(reference_dir).hash
        except (ValueError, OSError) as exc:
            ref_hash = f"unreadable: {exc}"
    return {"ruler_hash": target_mod.ruler_hash(bench_root, target_file),
            "compare_version": COMPARE_VERSION,
            "quality_version": quality2.QUALITY2_VERSION,
            "target_id": t["id"], "target_hash": target_mod.target_hash(t),
            "image_digest": image_digest or None,
            "model_revision": t["model"]["revision"],
            "reference_hash": ref_hash}


def _launches(paths: Sequence[str]) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def cmd_tps_launch(args: argparse.Namespace) -> int:
    out = tps_launch(args.block, label=args.label, role=args.role,
                     stall_ms=args.stall_ms, ttft_stall_ms=args.ttft_stall_ms)
    atomic_write_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("label", "status", "headline", "n_usable",
                                           "n_blocks", "invalid_reasons")}))
    return 0


def cmd_tps_screen(args: argparse.Namespace) -> int:
    res = tps_screen(_launches([args.launch])[0], read_leaderboard(args.leaderboard),
                     args.tol)
    atomic_write_json(args.out, res)
    print(json.dumps(res))
    return 0 if res["passed"] else 1


def cmd_tps_preflight(args: argparse.Namespace) -> int:
    board = read_leaderboard(args.leaderboard) if args.leaderboard else {}
    verdict, why = tps_preflight(_load(args.stamps), _load(args.calibration),
                                 board.get("best"),
                                 need_incumbent=bool(args.leaderboard),
                                 board_path=str(args.leaderboard))
    res = {"ok": verdict is None, "verdict": verdict, "reasons": why}
    atomic_write_json(args.out, res)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


def cmd_tps_calibrate(args: argparse.Namespace) -> int:
    try:
        cal = tps_calibration(_launches(args.launch), _load(args.stamps))
    except ValueError as exc:
        print(f"calibration refused: {exc}", file=sys.stderr)
        return 1
    atomic_write_json(args.out, cal)
    print(json.dumps({k: cal[k] for k in ("sigma_L", "sigma_L_measured", "sigma_B",
                                           "mde", "n_launches")}))
    return 0


def cmd_tps_verdict(args: argparse.Namespace) -> int:
    res = tps_decide(launches=_launches(args.launch), design=args.design,
                     calibration=_load(args.calibration),
                     quality=[_load(q) for q in args.quality],
                     stamps=_load(args.stamps),
                     board=read_leaderboard(args.leaderboard),
                     guard_tol=args.guard_tol)
    res["id"] = args.id
    atomic_write_json(args.out, res)
    print(json.dumps({"id": args.id, "verdict": res["verdict"],
                      "reasons": res["reasons"],
                      "delta": (res.get("stats") or {}).get("delta")}))
    return 0


def cmd_tps_promote(args: argparse.Namespace) -> int:
    results = _load(args.results) or {}
    q = results.get("quality") or {}
    if not args.reference and q.get("passed") is not True:
        print("refusing to promote: the quality2 gate did not pass on every "
              "candidate launch. The leaderboard is unchanged.", file=sys.stderr)
        return 1
    st = results.get("stats") or {}
    entry = {"id": args.id, "kind": "tps",
             "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "headline": (results.get("metric") or {}).get("headline"),
             "levels": (results.get("metric") or {}).get("levels"),
             **{k: (results.get("metric") or {}).get(k)
                for k in ("decode_tps", "stream_tps", "by_isl")},
             "sentinel": args.sentinel if args.sentinel is not None else (
                 math.log(results["metric"]["headline"])
                 if (results.get("metric") or {}).get("headline") else None),
             "delta": st.get("delta"), "mde": st.get("mde"),
             "delta_confirmed": None, "confirmed": None,
             "stamps": results.get("stamps"), "image": args.image,
             "snapshot": args.snapshot, "results_json": args.results}
    board = tps_promote(read_leaderboard(args.leaderboard), entry,
                        as_reference=args.reference)
    write_leaderboard(args.leaderboard, board)
    print(json.dumps({"best": board["best"]["id"], "history": len(board["history"])}))
    return 0


def cmd_tps_confirm(args: argparse.Namespace) -> int:
    board = read_leaderboard(args.leaderboard)
    res = tps_confirm_decide(launches=_launches(args.launch),
                             calibration=_load(args.calibration),
                             stamps=_load(args.stamps), board=board, run_id=args.id)
    if (board.get("best") or {}).get("id") != args.id:
        res["note"] = f"{args.id} is not the leaderboard best; nothing was changed"
    else:
        write_leaderboard(args.leaderboard, tps_apply_confirmation(board, args.id, res))
    atomic_write_json(args.out, res)
    print(json.dumps({"id": args.id, "verdict": res["verdict"],
                      "reasons": res["reasons"]}))
    return 0


def cmd_stamps(args: argparse.Namespace) -> int:
    st = compute_stamps(bench_root=args.bench_root, target_file=args.target,
                        image_digest=args.image_digest,
                        reference_dir=args.reference_dir)
    if args.out:
        atomic_write_json(args.out, st)
    print(json.dumps(st))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    print(json.dumps(read_leaderboard(args.leaderboard), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tps-launch", help="reduce one launch's measured blocks")
    t.add_argument("--label", required=True)
    t.add_argument("--role", choices=["incumbent", "candidate"], required=True)
    t.add_argument("--block", action="append", default=[], help="a harness tps out dir")
    t.add_argument("--stall-ms", type=float, default=TPS_STALL_MS)
    t.add_argument("--ttft-stall-ms", type=float, default=None,
                   help="also a stall: a first token slower than this "
                        "(set it well above the TTFT SLO)")
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tps_launch)

    t = sub.add_parser("tps-screen", help="step 1: candidate vs the incumbent sentinel")
    t.add_argument("--launch", required=True)
    t.add_argument("--leaderboard", required=True)
    t.add_argument("--tol", type=float, default=0.02)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tps_screen)

    t = sub.add_parser("tps-preflight",
                       help="before any launch: can this run be judged at all")
    t.add_argument("--stamps", default=None)
    t.add_argument("--calibration", default=None)
    t.add_argument("--leaderboard", default=None,
                   help="the incumbent is its best; omit for a confirmation")
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tps_preflight)

    t = sub.add_parser("tps-calibrate", help="step 0: sigma_L, sigma_B, MDE")
    t.add_argument("--launch", action="append", default=[])
    t.add_argument("--stamps", default=None)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tps_calibrate)

    t = sub.add_parser("tps-verdict", help="steps 2 and 4: ABBA / ICI verdict")
    t.add_argument("--id", required=True)
    t.add_argument("--design", choices=["abba", "ici"], default="abba")
    t.add_argument("--launch", action="append", default=[], help="in time order")
    t.add_argument("--quality", action="append", default=[],
                   help="quality2 gate.json of each candidate launch")
    t.add_argument("--calibration", default=None)
    t.add_argument("--stamps", default=None)
    t.add_argument("--leaderboard", required=True)
    t.add_argument("--guard-tol", type=float, default=TPS_GUARD_TOL)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tps_verdict)

    t = sub.add_parser("tps-promote", help="make a TPS result the leaderboard best")
    t.add_argument("--id", required=True)
    t.add_argument("--results", required=True)
    t.add_argument("--leaderboard", required=True)
    t.add_argument("--image", default=None)
    t.add_argument("--snapshot", default=None)
    t.add_argument("--sentinel", type=float, default=None)
    t.add_argument("--reference", action="store_true",
                   help="the bootstrap baseline: also the stock reference")
    t.set_defaults(func=cmd_tps_promote)

    t = sub.add_parser("tps-confirm", help="step 3: held-out ICI confirmation")
    t.add_argument("--id", required=True)
    t.add_argument("--launch", action="append", default=[])
    t.add_argument("--calibration", default=None)
    t.add_argument("--stamps", default=None)
    t.add_argument("--leaderboard", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_tps_confirm)

    t = sub.add_parser("stamps", help="DESIGN 2.6 stamps of this ruler")
    t.add_argument("--bench-root", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--image-digest", default=None)
    t.add_argument("--reference-dir", default=None)
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_stamps)

    sh = sub.add_parser("show", help="print the leaderboard")
    sh.add_argument("--leaderboard", default="leaderboard.json")
    sh.set_defaults(func=cmd_show)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

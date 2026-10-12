"""2. Trading quality or a guarded load level for speed.

(a) A `quantised` server (1.3x faster, its greedy text drifting from the
    honest server's after ~20 tokens) must fail the quality gate, judged
    against a reference built from the honest server.
(b) A chain of STEPS candidates. Each is ~10% faster at the headline load
    level and 2.9% slower at a lower one than the candidate before it: just
    inside the 3% a candidate may lose against the incumbent. Judged only
    against the moving incumbent, every step would be promoted and the low
    level would end 11% down. The judge also holds every candidate against
    the first, stock measurement, so the chain must stop at step 2.

(b) feeds made-up launch summaries to the real verdict (compare.py
tps-verdict, then tps-promote as run.sh would): nothing is measured.
"""
from __future__ import annotations

from core import END_TO_END, SIMULATED, Outcome
from fixtures import abba

GUARD_LEVEL, HEAD_LEVEL = "4", "8"          # concurrency levels
STOCK = {GUARD_LEVEL: 60.0, HEAD_LEVEL: 100.0}   # tokens/s of the first baseline
STEPS = 4
HEAD_GAIN, GUARD_LOSS = 1.10, 0.971
PASSED_GATE = {"passed": True, "reasons": []}


def chain() -> list[dict]:
    """Step k's candidate against step k-1's (the incumbent it would replace)."""
    steps, inc = [], dict(STOCK)
    for k in range(1, STEPS + 1):
        cand = {GUARD_LEVEL: inc[GUARD_LEVEL] * GUARD_LOSS,
                HEAD_LEVEL: inc[HEAD_LEVEL] * HEAD_GAIN}
        steps.append({"step": k, "candidate": cand, "incumbent": inc,
                      "guard_level_vs_stock": cand[GUARD_LEVEL] / STOCK[GUARD_LEVEL] - 1})
        inc = cand
    return steps


def _board(incumbent: dict, stock: dict | None) -> dict:
    entry = lambda i, lv: {"id": i, "kind": "tps", "levels": lv,  # noqa: E731
                           "headline": max(lv.values())}
    return {"best": entry("previous-step", incumbent), "history": [],
            **({"reference": entry("stock", stock)} if stock else {})}


def quantised(judge) -> Outcome:
    with judge.serve("honest") as ref, judge.serve("quantised") as cand:
        g = judge.quality_gate(ref, cand, "quantised")
    return Outcome("2a", "a 1.3x faster server whose output drifts from the reference",
                   "fail it at the quality gate",
                   "quality gate " + ("PASSED" if g["passed"] else "failed"),
                   not g["passed"], END_TO_END, g)


def chained(judge) -> Outcome:
    seen, stopped = [], None
    for st in chain():
        v = judge.verdict(abba(st["incumbent"], st["candidate"]), [PASSED_GATE] * 2,
                          _board(st["incumbent"], STOCK))
        seen.append({**st, **v})
        if not (v["verdict"] == "PROMOTED" and v["on_leaderboard"]):
            stopped = st
            break
    # The same launches without the stock entry: they must promote, or the
    # chain was stopped by something other than the stock guard.
    control = None
    if stopped:
        control = judge.verdict(abba(stopped["incumbent"], stopped["candidate"]),
                                [PASSED_GATE] * 2, _board(stopped["incumbent"], None))
    promoted = seen[:-1] if stopped else seen
    kept = promoted[-1]["guard_level_vs_stock"] if promoted else 0.0
    if stopped:
        why = (seen[-1]["reasons"] or [seen[-1]["verdict"]])[0].split(":")[0]
        obs = (f"stopped at step {stopped['step']} ({seen[-1]['verdict']} {why}), "
               f"low level {kept:+.1%}")
        if control["verdict"] != "PROMOTED":
            obs += "; but so was the control without a stock entry"
    else:
        obs = f"all {STEPS} promoted; low level ended {kept:+.1%} against the baseline"
    ok = bool(stopped and stopped["step"] == 2 and control["verdict"] == "PROMOTED")
    return Outcome("2b", f"{STEPS} candidates in a row, each +10% at one load, -2.9% at another",
                   "stop at step 2 (over 3% below the baseline)",
                   obs, ok, SIMULATED, {"steps": seen, "control_without_stock": control})


def run(judge, ctx) -> list:
    return [quantised(judge), chained(judge)]

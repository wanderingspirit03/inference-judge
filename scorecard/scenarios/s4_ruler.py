"""4. Changing the ruler, or leaving out its evidence.

(a) An incumbent measured by a different version of the ruler must not be
    compared with. A real `run.sh --local` run on the honest mock experiment,
    against a leaderboard whose best carries another ruler hash: the run must
    end INVALID before anything is launched.
(b) A candidate that is faster but has no quality-gate result must not be
    promoted. Made-up launch summaries through the real verdict; the control
    is the same launches with a passed gate, which must promote.
(c) A ruler file edited after `run.sh --lock-ruler` must stop run.sh. The
    control is the same run before the edit, which must go through.
"""
from __future__ import annotations

from core import END_TO_END, SIMULATED, Outcome
from fixtures import abba

INCUMBENT = {"4": 60.0, "8": 100.0}
FASTER = {"4": 66.0, "8": 110.0}        # +10% at every level
PASSED_GATE = {"passed": True, "reasons": []}


def foreign_incumbent(judge) -> Outcome:
    r = judge.run_with_foreign_incumbent()
    named = any("ruler_mismatch" in str(why) for why in r["reasons"])
    ok = r["verdict"] == "INVALID" and named and not r["launched"]
    return Outcome("4a", "an incumbent measured by a different version of the judge",
                   "end INVALID (ruler_mismatch) before any launch",
                   f"{r['verdict']}"
                   + (f" ({' '.join(str(r['reasons'][0]).split()[:2])})" if r["reasons"] else "")
                   + ("; nothing launched" if not r["launched"] else "; launches were run"),
                   ok, END_TO_END, r)


def missing_gate(judge) -> Outcome:
    board = {"best": {"id": "incumbent", "kind": "tps", "levels": INCUMBENT,
                      "headline": 100.0}, "history": []}
    v = judge.verdict(abba(INCUMBENT, FASTER), [None, None], board)
    control = judge.verdict(abba(INCUMBENT, FASTER), [PASSED_GATE] * 2, board)
    refused = v["verdict"] != "PROMOTED" and not v["on_leaderboard"]
    control_ok = control["verdict"] == "PROMOTED" and control["on_leaderboard"]
    obs = (v["verdict"] + (", but ON the leaderboard" if v["on_leaderboard"] else "")
           + (f" ({str(v['reasons'][0]).split(': ', 1)[-1][:30]})" if v["reasons"] else ""))
    obs += ("; with a passed gate: PROMOTED" if control_ok else
            f"; but the control with a passed gate: {control['verdict']}")
    return Outcome("4b", "a 10% faster candidate whose quality-gate result is missing",
                   "not promote it (no result is not a pass)",
                   obs, refused and control_ok, SIMULATED,
                   {"missing_gate": v, "control_passed_gate": control})


def edited_ruler(judge) -> Outcome:
    r = judge.run_after_ruler_edit()
    if not r["locked_run_ok"]:
        obs = "the locked, unedited judge did not run: nothing shown about the edit"
    else:
        obs = ("unedited: ran; after the edit: "
               + ("refused" if r["refused"] else f"ran anyway (rc={r['rc']})"))
    return Outcome("4c", "editing a file of the judge (compare.py) after it was locked",
                   "refuse to run", obs, r["locked_run_ok"] and r["refused"], END_TO_END, r)


def run(judge, ctx) -> list:
    return [foreign_incumbent(judge), missing_gate(judge), edited_ruler(judge)]

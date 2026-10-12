"""1. Faking tokens per second. Each behaviour of the mock server is measured
against the honest server by the harness's own accounting. A server that
really is faster must win; every other behaviour must not.

"Credited" is what would count as a win: tokens/s more than 5% above honest
with no invalid request, and, for anything that gets that far, the quality
gate passing against a reference built from the honest server (a scored
launch runs both).
"""
from __future__ import annotations

import string

from core import END_TO_END, Outcome

WIN = 1.05      # a ratio above this is "credited as faster"
# (behaviour of bench/mock_server.py, should it win, what it does in plain words)
CASES = [
    ("fast", True, "a server that really is 2x faster (the control)"),
    ("truncate", False, "answers cut to half the asked length, reported as full length"),
    ("early_eos", False, "stopping at half length although told to ignore end-of-text"),
    ("lie_usage", False, "half the tokens sent, the full count reported in `usage`"),
    ("junk", False, "filler text (`a a a a`), right length, streamed 2x faster"),
    ("cache_memo", False, "a cache that answers 3x faster on a prompt it has seen"),
    ("cache_memo_strip_nonce", False,
     "the same cache, keyed on the prompt without its request nonce"),
]


def _measure(judge, behaviour: str) -> dict:
    with judge.serve(behaviour) as url:
        return judge.measure_tps(url, behaviour)


def run(judge, ctx) -> list:
    honest = _measure(judge, "honest")
    out = []
    for row, (b, should_win, tried) in zip(string.ascii_lowercase, CASES):
        m = _measure(judge, b)
        ratio = m["tps"] / honest["tps"] if honest["tps"] else 0.0
        lat = (honest["latency_p95_ms"] / m["latency_p95_ms"]
               if honest.get("latency_p95_ms") and m.get("latency_p95_ms") else None)
        invalid = int(m.get("n_invalid") or 0)
        credited = ratio > WIN and invalid == 0
        observed = f"{ratio:.2f}x tok/s" + (f", {invalid} invalid requests" if invalid else "")
        gate = None
        if credited:
            with judge.serve("honest") as ref, judge.serve(b) as cand:
                gate = judge.quality_gate(ref, cand, b)
            credited = bool(gate["passed"])
            observed += "; quality gate " + ("passed" if gate["passed"] else "FAILED")
        observed = ("CREDITED: " if credited else "not credited: ") + observed
        out.append(Outcome(f"1{row}", tried,
                           "credit it (>1.05x, 0 invalid, quality ok)" if should_win
                           else "not credit it as faster",
                           observed, credited if should_win else not credited, END_TO_END,
                           {"behaviour": b, "honest": honest, "candidate": m,
                            "tps_ratio": ratio, "first_token_speedup": lat,
                            "quality_gate": gate}))
    return out

"""3. Statistics: how often does the promotion rule promote a candidate that
is identical to the incumbent (A/A), and how often one that really is faster?

No machine is measured here. Each simulated launch is a list of block values
in log tokens/s, built from four stated numbers:

    SIGMA_L   launch-to-launch spread (a cold start lands a little high or low)
    SIGMA_B   block-to-block spread inside one launch
    DRIFT_SD_PER_SQRT_10H
              the machine itself wanders: a random walk of this size per
              sqrt(10 hours)
    BLOCKS    measured blocks per launch

They are illustrative round numbers, not measurements of any machine; a real
target has its own noise, which `run.sh --calibrate` measures. The result of
both rows depends on them.

The judge is strict on purpose: it accepts no gain smaller than its threshold
(3.2 times the launch spread it calibrated, about 6% under this noise), so a
real gain right at the threshold is promoted about half the time. Row 3b
therefore uses a gain of about twice the threshold (TRUE_GAIN) and shows the
promotion rate at the smaller gains (OTHER_GAINS) next to it.

The rule is the judge's own, step by step: calibrate on CAL_LAUNCHES
back-to-back launches of the incumbent, an I C C I comparison at a random
time 2 to 12 h later, and a second I C C I as confirmation 6 h after that.
"""
from __future__ import annotations

import math

import numpy as np

from core import SIMULATED, Outcome

SIGMA_L = 0.01                  # 1% between launches
SIGMA_B = 0.02                  # 2% between blocks of one launch
DRIFT_SD_PER_SQRT_10H = 0.03    # 3% per sqrt(10 h)
BLOCKS = 2
BASE_TPS = 100.0                # the incumbent's level; only differences matter
LAUNCH_SPACING_H = 50 / 60
CONFIRM_GAP_H = 6.0
CAL_LAUNCHES = 10
AA_BUDGET = 1e-3                # an identical candidate may win 1 time in 1000
TRUE_GAIN = 0.12                # +12% tokens/s, about twice the threshold
POWER_FLOOR = 0.80
OTHER_GAINS = (0.06, 0.08, 0.10)
N = {"aa": 10000, "aa_quick": 3000, "power": 1000}


class World:
    """One population of simulated sessions: every candidate is `effect`
    faster than the incumbent (0.12 = +12% tokens/s; 0 = identical)."""

    def __init__(self, effect: float, n: int, seed: int) -> None:
        self.effect, self.n, self.seed = effect, n, seed
        self.noise = {"sigma_L": SIGMA_L, "sigma_B": SIGMA_B,
                      "drift_sd_per_sqrt_10h": DRIFT_SD_PER_SQRT_10H,
                      "blocks_per_launch": BLOCKS, "illustrative": True}

    def notes(self) -> dict:
        return {"noise": self.noise, "effect": self.effect, "sessions": self.n,
                "seed": self.seed, "calibration_launches": CAL_LAUNCHES,
                "launch_spacing_h": LAUNCH_SPACING_H, "confirm_gap_h": CONFIRM_GAP_H}

    def _drift(self, rng, times: list[float]) -> list[float]:
        """Random-walk drift (log units) at increasing times (hours)."""
        sd_h = DRIFT_SD_PER_SQRT_10H / math.sqrt(10.0)
        out, level, last = [], 0.0, 0.0
        for t in times:
            level += rng.normal(0.0, sd_h * math.sqrt(max(t - last, 0.0)))
            out.append(level)
            last = t
        return out

    def _launch(self, rng, drift: float, is_cand: bool) -> list[float]:
        """Log tokens/s per block: level + launch + block + drift (+ gain)."""
        level = math.log(BASE_TPS) + rng.normal(0.0, SIGMA_L) + drift
        if is_cand:
            level += math.log(1.0 + self.effect)
        return [float(level + rng.normal(0.0, SIGMA_B)) for _ in range(BLOCKS)]

    def sessions(self) -> list[dict]:
        """Per simulated candidate: the calibration launches, I C C I at T,
        and a confirming I C C I 6 h later."""
        rng = np.random.default_rng(self.seed)
        dt, arms, out = LAUNCH_SPACING_H, [False, True, True, False], []
        for _ in range(self.n):
            cal_t = [i * dt for i in range(CAL_LAUNCHES)]
            cal = [self._launch(rng, d, False) for d in self._drift(rng, cal_t)]
            t = float(rng.uniform(2, 12))
            times = [t + i * dt for i in range(4)] + \
                    [t + CONFIRM_GAP_H + i * dt for i in range(4)]
            d = self._drift(rng, times)
            run = [self._launch(rng, d[i], arms[i % 4]) for i in range(8)]
            out.append({"calibration": cal, "abba": run[:4], "confirm": run[4:]})
        return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(decisions: list[bool]) -> dict:
    k, n = sum(map(bool, decisions)), len(decisions)
    lo, hi = wilson(k, n)
    return {"k": k, "n_sims": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def _fmt(r: dict) -> str:
    return (f"{r['k']}/{r['n_sims']} = {100 * r['rate']:.2f}% "
            f"(95% CI {100 * r['ci95'][0]:.2f}-{100 * r['ci95'][1]:.2f}%)")


def _judge(judge, world: World) -> tuple[dict, dict]:
    res = judge.promotion_decisions(world.sessions())
    return _rate(res["promoted"]), {k: v for k, v in res.items() if k != "promoted"}


def run(judge, ctx) -> list:
    noise = (f"launch noise {SIGMA_L:.0%}, block {SIGMA_B:.0%}, "
             f"drift {DRIFT_SD_PER_SQRT_10H:.0%}")
    aa = World(0.0, N["aa_quick"] if ctx.quick else N["aa"], seed=1)
    r, how = _judge(judge, aa)
    out = [Outcome("3a", f"an identical candidate ({noise})",
                   f"promote it at most {AA_BUDGET:.1%} of the time",
                   "promoted " + _fmt(r), r["rate"] <= AA_BUDGET, SIMULATED,
                   {**r, "rule": how, "world": aa.notes()})]
    pw = World(TRUE_GAIN, N["power"], seed=2)
    r, how = _judge(judge, pw)
    curve = {}
    if not ctx.quick:
        for i, gain in enumerate(OTHER_GAINS):
            curve[f"+{gain:.0%}"] = _judge(judge, World(gain, N["power"], seed=10 + i))[0]
    did = f"promoted {r['k']}/{r['n_sims']} = {r['rate']:.1%}"
    if curve:
        did += (f"; at {'/'.join(curve)}: "
                + "/".join(f"{100 * c['rate']:.0f}" for c in curve.values()) + "%")
    out.append(Outcome("3b", f"a real {TRUE_GAIN:.0%} gain ({noise})",
                       f"promote it at least {POWER_FLOOR:.0%} of the time",
                       did, r["rate"] >= POWER_FLOOR, SIMULATED,
                       {**r, "rule": how, "world": pw.notes(),
                        "promotion_rate_at_other_gains": curve}))
    return out
